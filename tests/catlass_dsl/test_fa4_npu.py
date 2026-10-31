"""Real Ascend 950 accuracy through the FA4 public dispatch boundary."""

from pathlib import Path
import subprocess
import sys

import pytest
import torch
from catlass import tla

from flash_attn_npu_4 import flash_attn_func, flash_attn_varlen_func
from flash_attn_npu_4.catlass_dsl import compute_block_sparsity, simt
from flash_attn_npu_4.catlass_dsl.block_sparsity import transpose_block_sparsity

from .test_graph import SCORE, SCORE_BWD, _score_ref
from .test_utils import check_attention, make_inputs

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("packed", [False, True])
def test_first_compile_initializes_dsl_without_affecting_ordinary_fa4(device, packed):
    # A subprocess is necessary: importing this test module already loads the
    # callback helpers. Here the first native dispatch must happen in Dynamo.
    script = """
import importlib.abc
from importlib import metadata
import sys
import torch
import torch_npu

device = torch.device("npu", int(sys.argv[2]))
packed = sys.argv[1] == "True"
original = metadata.version
def version(name):
    if name == "ascend-catlass-dsl":
        raise AssertionError("Ordinary FA4 checked the optional DSL dependency")
    return original(name)
class NoCatlass(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "catlass" or fullname.startswith("catlass."):
            raise AssertionError("Ordinary FA4 imported CATLASS")
metadata.version = version
blocker = NoCatlass()
sys.meta_path.insert(0, blocker)
import flash_attn_npu_4 as fa4
assert "flash_attn_npu_4.catlass_dsl.graph" not in sys.modules
ordinary = [torch.randn(1, 17, 2, 128, device=device, dtype=torch.float16,
                       requires_grad=True) for _ in range(3)]
out = fa4.flash_attn_func(*ordinary)
out.sum().backward()
assert all(t.grad is not None for t in ordinary)
assert "catlass" not in sys.modules
metadata.version = original
sys.meta_path.remove(blocker)

shape = (17, 2, 128) if packed else (1, 17, 2, 128)
inputs = [torch.randn(shape, device=device, dtype=torch.float16,
                      requires_grad=True) for _ in range(3)]
cu = torch.tensor([0, 17], device=device, dtype=torch.int32) if packed else None
def fn(q, k, v, cu):
    if packed:
        return fa4.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                         aux_scalars=(1.0,), return_lse=True)
    return fa4.flash_attn_func(q, k, v, aux_scalars=(1.0,), return_lse=True)
assert "flash_attn_npu_4.catlass_dsl.graph" not in sys.modules
actual = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)(*inputs, cu)
assert "flash_attn_npu_4.catlass_dsl.graph" in sys.modules
expected = fn(*inputs, cu)
for a, b in zip(actual, expected):
    torch.testing.assert_close(a, b)
tangents = tuple(torch.ones_like(t) for t in actual)
got = torch.autograd.grad(actual, inputs, tangents)
wanted = torch.autograd.grad(expected, inputs, tangents)
for a, b in zip(got, wanted):
    torch.testing.assert_close(a, b)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(packed), str(device.index)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@simt
@tla.jit
def mask(b, h, q, k, info, aux, scalars):
    return (k <= q) & ((q - k) < 129)


def mask_ref(b, h, q, k, info, aux, scalars):
    return (k <= q) & ((q - k) < 129)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("qp,kp", [(False, False), (True, False), (False, True), (True, True)])
def test_public_eager_score_and_mask(device, dtype, qp, kp):
    cpu, inputs, cq, ck, options = make_inputs(device, dtype, qp=qp, kp=kp, sq=131, sk=137, kh=1)
    tokens = inputs[1].shape[0] if kp else inputs[1].shape[1]
    bias = torch.linspace(-0.1, 0.1, tokens)
    kwargs = dict(
        mask_mod=mask,
        score_mod=SCORE,
        score_mod_bwd=SCORE_BWD,
        aux_tensors=[bias.to(device)],
        aux_scalars=(0.5,),
        return_lse=True,
    )
    if qp or kp:
        result = flash_attn_varlen_func(*inputs, **options, **kwargs)
        assert len(result) == 4
        for empty in result[2:]:
            assert empty.numel() == 0 and empty.dtype == torch.float32
            assert empty.requires_grad and empty.device == inputs[0].device
    else:
        result = flash_attn_func(*inputs, **kwargs)
        assert len(result) == 2
    check_attention(
        cpu,
        inputs,
        *result[:2],
        cq=cq,
        ck=ck,
        loss="both",
        mask_mod=mask_ref,
        score_mod=_score_ref,
        aux_tensors=(bias,),
        aux_scalars=(0.5,),
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("window", [(-1, 3), (3, -1)])
def test_public_one_sided_window(device, packed, window):
    cpu, inputs, cq, ck, options = make_inputs(device, torch.bfloat16, qp=packed, kp=packed)
    kwargs = dict(window_size=window, aux_scalars=(1.0,), return_lse=True)
    result = (
        flash_attn_varlen_func(*inputs, **options, **kwargs)
        if packed
        else flash_attn_func(*inputs, **kwargs)
    )
    left, right = window

    def expected_mask(b, h, q, k, info, aux, scalars):
        center = q + info[1] - info[0]
        return ((k >= center - left) if left >= 0 else True) & (
            (k <= center + right) if right >= 0 else True
        )

    check_attention(cpu, inputs, *result[:2], cq=cq, ck=ck, loss="both", mask_mod=expected_mask)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_inductor_sparse_dynamic_updates(device, packed, dtype):
    from torch._inductor import config

    assert config.fx_graph_cache, "Acceptance must retain the default FX graph cache"

    def fn(q, k, v, cq, ck, bias):
        batch = q.shape[0] if cq is None else cq.shape[0] - 1
        heads = q.shape[2] if cq is None else q.shape[1]
        sq, sk = (q.shape[1], k.shape[1]) if cq is None else (q.shape[0], k.shape[0])
        blocks = compute_block_sparsity(
            128, 128, batch, heads, sq, sk, mask, None, q.device, cu_seqlens_q=cq, cu_seqlens_k=ck
        )
        transposed = transpose_block_sparsity(
            blocks,
            batch_size=batch,
            num_heads=heads,
            seqlen_q=sq,
            seqlen_k=sk,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
        )
        kwargs = dict(
            mask_mod=mask,
            block_sparse_tensors=blocks,
            score_mod=SCORE,
            score_mod_bwd=SCORE_BWD,
            aux_tensors=[bias],
            aux_scalars=(0.5,),
            return_lse=True,
        )
        if packed:
            result = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cq, cu_seqlens_k=ck, **kwargs)
        else:
            result = flash_attn_func(q, k, v, block_sparse_tensors_bwd=transposed, **kwargs)
        return result, transposed.mask_block_cnt

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        # Repeat a shape while changing Q/K/V, aux values and packed lengths,
        # then change the shape as well. Cached programs must bind fresh inputs.
        for step, length in enumerate((131, 131, 259)):
            cpu, inputs, cq, ck, options = make_inputs(
                device,
                dtype,
                qp=packed,
                kp=packed,
                sq=length,
                sk=length + 6,
                q_lengths=(length - 3 - step, length),
                k_lengths=(length + 6, length - step),
                seed=1909 + step,
            )
            tokens = inputs[1].shape[0] if packed else inputs[1].shape[1]
            bias = torch.linspace(-0.1 * (step + 1), 0.1 * (step + 1), tokens)
            result, counts = compiled(
                *inputs, options["cu_seqlens_q"], options["cu_seqlens_k"], bias.to(device)
            )
            assert len(result) == (4 if packed else 2)
            assert counts.dtype == torch.int32
            check_attention(
                cpu,
                inputs,
                *result[:2],
                cq=cq,
                ck=ck,
                loss="both",
                mask_mod=mask_ref,
                score_mod=_score_ref,
                aux_tensors=(bias,),
                aux_scalars=(0.5,),
            )
    torch._dynamo.reset()
