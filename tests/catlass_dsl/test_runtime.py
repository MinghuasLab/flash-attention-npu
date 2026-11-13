"""Executor rebinding and producer/consumer streams through the public API."""

from unittest.mock import Mock

import catlass.tla as tla
import pytest
import torch

from .test_utils import DTYPES, check, check_attention, copy_aux_to_device, make_inputs, oracle

from flash_attn_npu_4.catlass_dsl import builtin, flash_attn_func, flash_attn_varlen_func, runtime
from flash_attn_npu_4.catlass_dsl.cache_utils import get_jit_cache

pytestmark = pytest.mark.npu


@tla.jit
def score(s, b, h, q, k, info, aux, scalars):
    return s * scalars[0] + aux[0][q]


@tla.jit
def score_bwd(g, s, b, h, q, k, info, aux, scalars):
    return g * scalars[0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_cache_rebinds_inputs_aux_and_scalars(device, dtype):
    previous = []
    keys = None
    for seed, value in ((701, 0.3), (702, 1.2)):
        cpu, inputs, _, _, _ = make_inputs(device, dtype, seed=seed)
        data = torch.linspace(-value, value, 34).reshape(17, 2)[:, 1]
        aux = copy_aux_to_device(data, device)
        out, lse = flash_attn_func(
            *inputs,
            score_mod=score,
            score_mod_bwd=score_bwd,
            aux_tensors=[aux],
            aux_scalars=(value,),
            return_lse=True,
        )
        reference = lambda s, b, h, q, k, info, buffers, scalars: s * scalars[0] + buffers[0][q]
        check_attention(
            cpu, inputs, out, lse, score_mod=reference, aux_tensors=[data], aux_scalars=(value,)
        )
        if previous:
            assert set(runtime.compile_cache) == keys
            assert all(old.data_ptr() != new.data_ptr() for old, new in zip(previous, inputs))
        keys = set(runtime.compile_cache)
        previous = inputs


@pytest.mark.parametrize("dtype", DTYPES)
def test_producer_consumer_stream(device, dtype):
    producer = torch.npu.Stream(device=device)
    consumers = [torch.npu.Stream(device=device) for _ in range(2)]
    keys = None
    for step, consumer in enumerate(consumers):
        cpu, _, _, _, _ = make_inputs("cpu", dtype, sq=128, sk=128, seed=1909 + step)
        with torch.npu.stream(producer):
            inputs = [t.detach().to(device, non_blocking=True).requires_grad_() for t in cpu]
        with torch.npu.stream(consumer):
            consumer.wait_stream(producer)
            out, lse = flash_attn_func(*inputs, return_lse=True)
            check_attention(cpu, inputs, out, lse, loss="both")
        torch.npu.current_stream(device).wait_stream(consumer)
        if keys is not None:
            assert set(runtime.compile_cache) == keys
        keys = set(runtime.compile_cache)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", [False, True], ids=["bsnd", "tnd"])
def test_cached_plan_rebinds_scale_and_segment_boundaries(device, dtype, packed):
    """A warm plan must read fresh scales, aux values and packed boundaries."""
    mask = builtin.causal(alignment="bottom_right")
    keys = None
    for step, (scale, value) in enumerate(((0.0, 0.3), (-0.125, 1.2))):
        cpu, inputs, cq, ck, options = make_inputs(
            device,
            dtype,
            qp=packed,
            kp=packed,
            sq=17,
            sk=23,
            q_lengths=(5, 12) if step == 0 else (8, 9),
            k_lengths=(13, 10) if step == 0 else (9, 14),
            seed=701 + step,
        )
        data = torch.linspace(-value, value, 34).reshape(17, 2)[:, 1]
        out, lse = flash_attn_varlen_func(
            *inputs,
            **options,
            max_seqlen_q=17,
            max_seqlen_k=23,
            score_mod=score,
            score_mod_bwd=score_bwd,
            mask_mod=mask,
            softmax_scale=scale,
            aux_tensors=[copy_aux_to_device(data, device)],
            aux_scalars=(value,),
            return_lse=True,
        )
        check_attention(
            cpu,
            inputs,
            out,
            lse,
            cq=cq,
            ck=ck,
            loss="both",
            scale=scale,
            score_mod=lambda s, b, h, q, k, info, aux, scalars: s * scalars[0] + aux[0][q],
            mask_mod=lambda b, h, q, k, info, aux, scalars: k <= q + info[1] - info[0],
            aux_tensors=(data,),
            aux_scalars=(value,),
        )
        if keys is not None:
            assert set(runtime.compile_cache) == keys
        keys = set(runtime.compile_cache)


def test_attention_plan_eviction_and_clear(device, monkeypatch):
    """Both item eviction and clear must discard the prepared executor."""
    cache = get_jit_cache()
    monkeypatch.setattr(runtime, "compile_cache", cache)
    compiler = Mock(wraps=runtime._compile_kernel)
    monkeypatch.setattr(runtime, "_compile_kernel", compiler)
    for step, expected_compilations in enumerate((1, 1, 2, 3)):
        if step == 2:
            cache.pop(next(iter(cache)))
        elif step == 3:
            runtime.clear_compile_cache()
            assert not cache
        cpu, inputs, _, _, _ = make_inputs(device, torch.float16, seed=903 + step)
        with torch.no_grad():
            out, lse = flash_attn_func(*inputs, return_lse=True)
            expected, expected_lse = oracle(cpu)
        check(out, expected)
        check(lse, expected_lse)
        assert compiler.call_count == expected_compilations
