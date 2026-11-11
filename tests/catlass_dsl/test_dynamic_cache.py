"""Cross-length compilation reuse, including changing launch/aux geometry."""

from unittest.mock import Mock

import catlass.tla as tla
import pytest
import torch

from flash_attn_npu_4.catlass_dsl import (
    compute_block_sparsity,
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_varlen_func_with_quant,
    runtime,
)
from flash_attn_npu_4.catlass_dsl.cache_utils import JITCache
from .test_utils import DTYPES, check_attention, copy_aux_to_device, make_inputs
from .test_runtime import score, score_bwd

pytestmark = pytest.mark.npu


@pytest.fixture
def compiler(monkeypatch):
    monkeypatch.setattr(runtime, "compile_cache", JITCache())
    compiler = Mock(wraps=runtime._compile_kernel)
    monkeypatch.setattr(runtime, "_compile_kernel", compiler)
    return compiler


def test_broadcast_aux_metadata_uses_wide_indices():
    # A zero stride keeps the storage tiny even when a logical extent needs
    # 64 bits. Metadata width must cover shapes as well as memory offsets.
    auxiliary = torch.empty(1).expand(2**31)
    assert runtime._aux_specs((auxiliary,))[0][3] is False
    _, metadata = runtime._aux((auxiliary,), ())
    assert metadata[0].value == 2**31
    assert type(metadata[0]) is tla.Int64


@pytest.mark.parametrize("kind", ["flat", "sparse", "strided-aux", "uint32", "fp8", "empty"])
@pytest.mark.parametrize("prepared", [False, True])
def test_launch_descriptor_matches_catlass_abi(device, monkeypatch, kind, prepared):
    monkeypatch.setattr(runtime, "_prepared_tensor_binding", lambda: prepared)
    dtype = torch.uint32 if kind == "uint32" else torch.float16
    tensor = torch.zeros(100, dtype=dtype).to(device)[5:85]
    if kind == "strided-aux":
        tensor = tensor[::3]
    if kind == "empty":
        tensor = tensor[:0]
    if kind == "fp8":
        tensor = tensor.to(torch.float8_e4m3fn)
    tensor = runtime._buffer(tensor)
    pointer = kind in ("strided-aux", "uint32")
    shape = (8, 10) if kind == "sparse" else (1,) if pointer else (tensor.numel(),)
    raw = (
        tensor.view(torch.int32)
        if kind == "uint32"
        else tensor.view(torch.uint8)
        if kind == "fp8"
        else tensor
    )
    options = {"element_type": tla.Float8E4M3FN} if kind == "fp8" else {}
    expected = tla.from_dlpack(raw, origin_shape=shape, layout_tag=tla.arch.RowMajor, **options)
    if not pointer:
        expected.mark_layout_dynamic()
    binding = runtime._tensor(tensor, shape, logical=pointer, fp8=kind == "fp8")
    assert tuple(binding.__c_pointers__()) == (tensor.data_ptr(),)
    assert binding.build_memref_launch_fields() == expected.build_memref_launch_fields()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", [False, True])
def test_forward_backward_reuse_across_lengths(device, compiler, dtype, packed):
    first = None
    for step, (sq, sk) in enumerate(((17, 23), (33, 45), (65, 67), (17, 23))):
        cpu, inputs, cq, ck, options = make_inputs(
            device,
            dtype,
            sq=sq,
            sk=sk,
            qp=packed,
            kp=packed,
            q_lengths=(sq - 3, 3),
            k_lengths=(5, sk - 5),
            seed=1120 + step,
        )
        # Both storage offset and non-unit stride change without changing ABI.
        stride = 2 + step
        data = torch.linspace(-0.2, 0.3, sq * stride).reshape(sq, stride)[:, -1]
        out, lse = flash_attn_varlen_func(
            *inputs,
            **options,
            max_seqlen_q=sq,
            max_seqlen_k=sk,
            score_mod=score,
            score_mod_bwd=score_bwd,
            aux_tensors=(copy_aux_to_device(data, device),),
            aux_scalars=(0.7 + step / 10,),
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
            score_mod=lambda s, b, h, q, k, i, a, c: s * c[0] + a[0][q],
            aux_tensors=(data,),
            aux_scalars=(0.7 + step / 10,),
        )
        owners = tuple(runtime.compile_cache.values())
        if first is None:
            first = owners
        assert owners == first
        assert compiler.call_count == 2  # One forward and one backward.


@tla.jit
def causal_mask(b, h, q, k, info, aux, scalars):
    return q >= k


@pytest.mark.parametrize("sampled", [False, True])
def test_mask_reuses_dynamic_sparse_capacity(device, compiler, sampled):
    from .reference import reference_block_sparsity
    from .test_block_sparsity import active_lists

    for sq, sk in ((17, 23), (129, 257), (257, 129), (17, 23)):
        result = compute_block_sparsity(
            128,
            128,
            1,
            1,
            sq,
            sk,
            causal_mask,
            aux_tensors=None,
            device=device,
            use_fast_sampling=sampled,
        )
        expected = reference_block_sparsity(1, 1, sq, sk, lambda b, h, q, k, i, a, c: q >= k)
        assert active_lists(result) == active_lists(expected)
        assert active_lists(result, full=True) == active_lists(expected, full=True)
        assert compiler.call_count == 1


def test_specializations_do_not_alias(device, compiler):
    for dtype in (torch.float16, torch.bfloat16, torch.float16):
        cpu, inputs, _, _, _ = make_inputs(device, dtype, sq=17, sk=23)
        out, lse = flash_attn_func(*inputs, return_lse=True)
        check_attention(cpu, inputs, out, lse, loss="both")
    assert compiler.call_count == 4


def _constant_score(factor):
    @tla.jit
    def callback(s, b, h, q, k, info, aux, scalars):
        return s * factor + aux[0][q]

    return callback


def test_callback_constants_and_aux_abi_isolation(device, compiler):
    from .test_utils import check, oracle

    programs = {factor: _constant_score(factor) for factor in (0.5, 0.75)}
    for step, (factor, stride, expected_compiles) in enumerate(
        ((0.5, 1, 1), (0.75, 1, 2), (0.5, 2, 3), (0.5, 3, 3), (0.5, 1, 3))
    ):
        length = 17 + step * 16
        cpu, inputs, _, _, _ = make_inputs(device, torch.float16, sq=length, sk=length + 6)
        aux = torch.linspace(-0.1, 0.1, length * stride)[::stride]
        with torch.no_grad():
            out, lse = flash_attn_func(
                *inputs,
                score_mod=programs[factor],
                aux_tensors=(copy_aux_to_device(aux, device),),
                return_lse=True,
            )
            expected, logs = oracle(
                cpu,
                score_mod=lambda s, b, h, q, k, i, a, c: s * factor + a[0][q],
                aux_tensors=(aux,),
            )
        check(out, expected)
        check(lse, logs)
        assert compiler.call_count == expected_compiles


def test_alignment_specialization_reuses_prior_executor(device, compiler):
    for length, compiles in ((128, 2), (129, 4), (256, 4), (257, 4)):
        cpu, inputs, _, _, _ = make_inputs(
            device, torch.float16, sq=length, sk=length, heads=2, kh=2
        )
        out, lse = flash_attn_func(*inputs, return_lse=True)
        check_attention(cpu, inputs, out, lse, loss="both")
        assert compiler.call_count == compiles


@pytest.mark.parametrize("packed", [False, True])
def test_mxfp8_reuses_dynamic_lengths_and_scales(device, compiler, monkeypatch, packed):
    from .mxfp8_reference import make_case, reference, to_device, check_quantized_lse
    from .test_utils import check

    compile_fp8 = Mock(wraps=runtime._compile_mxfp8)
    monkeypatch.setattr(runtime, "_compile_mxfp8", compile_fp8)
    first = None
    for step, (sq, sk) in enumerate(((129, 193), (257, 321), (385, 449))):
        case = make_case(
            q_lengths=(sq,),
            k_lengths=(sk,),
            q_heads=4,
            kv_heads=2,
            q_packed=packed,
            k_packed=packed,
            v_scale_capacity=20 + step if packed else None,
            seed=1176 + step,
        )
        expected, logs = reference(case)
        inputs, options = to_device(case, device)
        out, lse = flash_attn_varlen_func_with_quant(*inputs, **options, return_lse=True)
        check(out, expected)
        check_quantized_lse(lse.cpu(), logs)
        owners = tuple(runtime.compile_cache.values())
        if first is None:
            first = owners
        assert owners == first
        assert compile_fp8.call_count == 1


def test_sparse_capacities_and_batch_rebind_independently(device, compiler):
    from .reference import reference_block_sparsity
    from .test_mxfp8 import _device_sparse

    first = None
    for step, (batch, sq, sk) in enumerate(((1, 129, 513), (2, 257, 769), (1, 129, 513))):
        sparse = reference_block_sparsity(1, 1, sq, sk, lambda b, h, q, k, i, a, c: q >= k)
        # Different partial/full row strides, both changing between launches.
        sparse = sparse._replace(
            mask_block_idx=sparse.mask_block_idx[..., : (1, 2, 1)[step]].contiguous(),
            full_block_idx=sparse.full_block_idx[..., : step + 2].contiguous(),
        )
        cpu, inputs, _, _, _ = make_inputs(
            device, torch.float16, sq=sq, sk=sk, q_lengths=(sq,) * batch, k_lengths=(sk,) * batch
        )
        out, lse = flash_attn_func(
            *inputs,
            mask_mod=causal_mask,
            block_sparse_tensors=_device_sparse(sparse, device),
            return_lse=True,
        )
        check_attention(
            cpu, inputs, out, lse, loss="both", mask_mod=lambda b, h, q, k, i, a, c: q >= k
        )
        owners = tuple(runtime.compile_cache.values())
        first = owners if first is None else first
        assert owners == first
        assert compiler.call_count == 2


@tla.jit
def batch_labels(b, h, q, k, info, aux, scalars):
    return aux[0][b, q] == aux[1][b, k]


@pytest.mark.parametrize("tail", [0, 1], ids=["full-aux-cache", "tile-aux-cache"])
def test_mxfp8_aux_cache_rebinds_geometry(device, compiler, monkeypatch, tail):
    from .mxfp8_reference import make_case, reference, to_device, check_quantized_lse
    from .test_mxfp8 import _device_sparse
    from .test_utils import check
    from .reference import reference_block_sparsity

    compile_fp8 = Mock(wraps=runtime._compile_mxfp8)
    monkeypatch.setattr(runtime, "_compile_mxfp8", compile_fp8)

    def mask_ref(b, h, q, k, info, aux, scalars):
        return aux[0][b, q] == aux[1][b, k]

    first = None
    for step, length in enumerate((1024 + tail, 1280 + tail)):
        case = make_case(q_lengths=(length,), k_lengths=(length,), q_heads=2, kv_heads=1)
        labels = (torch.arange(length, dtype=torch.int32) // 192)[None, :]
        if step:
            labels = labels.as_strided(labels.shape, (length * 3, 1))
            assert labels.is_contiguous()  # The batch stride is unused.
        auxiliary = (labels, labels.clone())
        sparse = reference_block_sparsity(1, 1, length, length, mask_ref, aux_tensors=auxiliary)
        sparse = sparse._replace(
            mask_block_idx=sparse.mask_block_idx[..., : 3 + step].contiguous(),
            full_block_idx=sparse.full_block_idx[..., : 4 + step].contiguous(),
        )
        inputs, options = to_device(case, device)
        rows = [0, 63, 127, 191, 255, length - 1]
        expected, logs = reference(case, mask_mod=mask_ref, aux=auxiliary, query_rows=[rows])
        out, lse = flash_attn_varlen_func_with_quant(
            *inputs,
            **options,
            return_lse=True,
            mask_mod=batch_labels,
            aux_tensors=tuple(copy_aux_to_device(t, device) for t in auxiliary),
            block_sparse_tensors=_device_sparse(sparse, device),
        )
        check(out[:, rows], expected[:, rows])
        check_quantized_lse(lse.cpu()[:, :, rows], logs[:, :, rows])
        assert compile_fp8.call_count == 1
        metadata = compile_fp8.call_args.args[0][-3]
        assert metadata and (metadata[0][0] < 0) == bool(tail)
        owners = tuple(runtime.compile_cache.values())
        first = owners if first is None else first
        assert owners == first


@pytest.mark.parametrize("kind", ["fp16", "bf16", "mxfp8"])
def test_layout_isolation_and_bnsd_length_reuse(device, compiler, monkeypatch, kind):
    from flash_attn_npu_4.catlass_dsl.interface import _flash
    from .mxfp8_reference import make_case, reference, to_device, check_quantized_lse
    from .test_utils import check

    fp8_compiler = Mock(wraps=runtime._compile_mxfp8)
    monkeypatch.setattr(runtime, "_compile_mxfp8", fp8_compiler)
    options = dict(
        softmax_scale=None,
        causal=False,
        window_size=(None, None),
        softcap=0.0,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        aux_tensors=None,
        aux_scalars=None,
        block_sparse_tensors=None,
        block_sparse_tensors_bwd=None,
        return_lse=True,
    )
    owners = {}
    for layout, sq in (("BNSD", 17), ("BNSD", 33), ("BSND", 17), ("BSND", 33), ("BNSD", 17)):
        if kind == "mxfp8":
            case = make_case(q_lengths=(sq,), k_lengths=(sq + 6,), layout=layout)
            inputs, extra = to_device(case, device)
            out, lse = _flash(
                *inputs,
                **options,
                layout=layout,
                **{name: extra[name] for name in ("q_descale", "k_descale", "v_descale")},
            )
            expected, logs = reference(case)
            check(out, expected)
            check_quantized_lse(lse.cpu(), logs)
        else:
            dtype = torch.float16 if kind == "fp16" else torch.bfloat16
            cpu, inputs, _, _, _ = make_inputs(device, dtype, sq=sq, sk=sq + 6)
            reordered = (
                [t.transpose(1, 2).contiguous() for t in inputs] if layout == "BNSD" else inputs
            )
            out, lse = _flash(*reordered, **options, layout=layout)
            check_attention(
                cpu, inputs, out.transpose(1, 2) if layout == "BNSD" else out, lse, loss="both"
            )
        current = tuple(runtime.compile_cache.values())
        if layout in owners:
            assert all(any(owner is entry for entry in current) for owner in owners[layout])
        owners[layout] = current
        assert (fp8_compiler.call_count if kind == "mxfp8" else compiler.call_count) == len(
            owners
        ) * (1 if kind == "mxfp8" else 2)
