# Adapted from CATLASS DSL tests.
"""CPU-only public contracts and compiled-kernel reuse."""

from contextlib import contextmanager
import sys
from types import SimpleNamespace

from importlib.util import find_spec

import pytest

# Only an absent optional runtime is skipped; a broken installed runtime fails.
if find_spec("catlass") is None:
    pytest.skip("CATLASS DSL runtime is not installed", allow_module_level=True)

from flash_attn_npu_dsl import interface, simd

if find_spec("torch") is None:
    pytest.skip("PyTorch is not installed", allow_module_level=True)
import torch


class _LabelledTensor(torch.Tensor):
    @property
    def device(self):
        return SimpleNamespace(type="npu", index=0)


def _labelled(shape, dtype=torch.float16):
    return torch.empty(shape, dtype=dtype).as_subclass(_LabelledTensor)


def _inputs(dim=128, dtype=torch.float16):
    return (
        _labelled((1, 17, 4, dim), dtype),
        _labelled((1, 65, 2, dim), dtype),
        _labelled((1, 65, 2, dim), dtype),
    )


def causal_mask(b, h, q, kv, info, tensors):
    return q >= kv


@simd
def vector_mask(b, h, q, kv, info, tensors):
    return interface.tla.cmp(q, kv, "ge")


def scale_score(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return value * 0.75


@simd
def vector_score(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return value * 0.75


def test_public_exports():
    import flash_attn_npu_dsl as flex_attention

    assert set(flex_attention.__all__) == {
        "BlockSparseTensorsTorch",
        "compute_block_sparsity",
        "flash_attn_func",
        "flash_attn_varlen_func",
        "simd",
    }


def test_bsnd_varlen_without_lengths_reuses_fixed_entry(host):
    inputs = tuple(
        torch.empty(s, dtype=torch.float16)
        for s in ((2, 65, 4, 64), (2, 129, 2, 64), (2, 129, 2, 96))
    )
    interface.flash_attn_func(*inputs, causal=True, return_lse=True)
    interface.flash_attn_varlen_func(*inputs, causal=True, return_lse=True)
    assert len(host.compiled) == 1
    assert all(call[0][8] is None for call in host.launched)


def test_bsnd_effective_lengths_bind_current_metadata_without_readback(host, monkeypatch):
    inputs = tuple(
        torch.empty(s, dtype=torch.float16)
        for s in ((2, 129, 4, 64), (2, 257, 2, 64), (2, 257, 2, 96))
    )
    used_q, used_k = (torch.tensor(x, dtype=torch.int32) for x in ([65, 1], [129, 0]))

    def no_read(*args, **kwargs):
        raise AssertionError("effective lengths must stay on device")

    monkeypatch.setattr(torch.Tensor, "item", no_read)
    monkeypatch.setattr(torch.Tensor, "tolist", no_read)
    for update in range(2):
        if update:
            used_q, used_k = used_q.clone(), used_k.clone()
            used_q.zero_()
        out, lse = interface.flash_attn_varlen_func(
            *inputs,
            seqused_q=used_q,
            seqused_k=used_k,
            max_seqlen_q=65,
            max_seqlen_k=129,
            return_lse=True,
        )
        assert out.shape == (2, 129, 4, 96) and lse.shape == (2, 4, 129)
        metadata = host.launched[-1][0][8][0]
        assert metadata[:2] == (None, None)
        assert metadata[2] is used_q and metadata[3] is used_k
    assert len(host.compiled) == 1
    assert host.compiled[0][11:13] == (65, 129)
    assert host.compiled[0][-1] == (129, 257)


@pytest.mark.parametrize("option", ["cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"])
def test_bsnd_sequence_contract_rejects_invalid_static_metadata(host, option):
    q = torch.empty(2, 129, 4, 64, dtype=torch.float16)
    k = torch.empty(2, 257, 2, 64, dtype=torch.float16)
    value = torch.zeros(3, dtype=torch.int32) if option.startswith("cu_") else 1
    with pytest.raises(ValueError, match=option):
        interface.flash_attn_varlen_func(q, k, k, **{option: value})
    assert not host.compiled and not host.launched


@pytest.mark.parametrize(
    "total_q,used,expected_dn",
    [
        (254, False, False),
        (256, False, True),
        (258, False, True),
        (1024, True, False),
    ],
)
def test_varlen_scalar_dn_uses_known_rows_not_only_upper_bound(host, total_q, used, expected_dn):
    from flash_attn_npu_dsl import BlockSparseTensorsTorch

    q = torch.empty(total_q, 4, 64, dtype=torch.float16)
    k = torch.empty(1026, 2, 64, dtype=torch.float16)
    cuq = torch.tensor([0, total_q // 2, total_q], dtype=torch.int32)
    cuk = torch.tensor([0, 513, 1026], dtype=torch.int32)
    blocks = BlockSparseTensorsTorch(
        torch.zeros(1, 8, dtype=torch.int32),
        torch.zeros(1, 40, dtype=torch.int32),
        cu_total_m_blocks=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_block_idx_offsets=torch.tensor([0, 20, 40], dtype=torch.int32),
        block_size=(128, 128),
    )
    interface.flash_attn_varlen_func(
        q,
        k,
        k,
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        max_seqlen_q=512,
        max_seqlen_k=513,
        seqused_q=torch.tensor([1, 1], dtype=torch.int32) if used else None,
        mask_mod=causal_mask,
        block_sparse_tensors=blocks,
    )
    assert host.compiled[0][24:26] == (expected_dn, False)
    sequences, pages, partial_stats = host.launched[0][0][8]
    assert sequences[0] is cuq and sequences[1] is cuk
    assert pages is None and partial_stats is None


@pytest.mark.parametrize(
    "storage_format,address,error",
    [(0, 16, None), (2, 16, None), (29, 16, ValueError), (2, 0, RuntimeError)],
)
def test_static_launch_requires_dense_bound_storage(monkeypatch, storage_format, address, error):
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(get_npu_format=lambda tensor: storage_format)
    )
    tensor = SimpleNamespace(data_ptr=lambda: address)
    if error is None:
        interface._check_launch_tensors((None, tensor))
    else:
        with pytest.raises(error, match="storage|not bound"):
            interface._check_launch_tensors((tensor,))


@pytest.mark.parametrize(
    "dim,dtype",
    [
        (1, torch.float16),
        (59, torch.bfloat16),
        (129, torch.float16),
        (192, torch.bfloat16),
        (255, torch.float16),
        (256, torch.bfloat16),
        (8, torch.float16),
        (16, torch.bfloat16),
        (32, torch.float16),
        (64, torch.float16),
        (80, torch.bfloat16),
        (96, torch.bfloat16),
        (128, torch.float16),
    ],
)
def test_supported_shapes(dim, dtype):
    interface._validate_inputs(*_inputs(dim, dtype))


@pytest.mark.parametrize(
    "shape,error",
    [
        ((17, 4, 128), "rank-4"),
        ((1, 0, 4, 128), "nonempty"),
        ((1, 17, 4, 257), "head dimension"),
        ((1, 17, 4, 64), "dimensions must match"),
        ((1, 17, 3, 128), "divisible"),
        ((2, 17, 4, 128), "batch sizes"),
    ],
)
def test_invalid_shapes(shape, error):
    _, k, v = _inputs()
    with pytest.raises(ValueError, match=error):
        interface._validate_inputs(_labelled(shape), k, v)


def test_inputs_reject_cpu_dtype_stride_and_grad():
    q, k, v = _inputs()
    for bad, error, message in (
        (torch.empty(q.shape), ValueError, "same NPU"),
        (_labelled(q.shape, torch.float32), TypeError, "FP16 or BF16"),
        (q.transpose(1, 2), ValueError, "contiguous"),
        (q.requires_grad_(), ValueError, "inference forward only"),
    ):
        with pytest.raises(error, match=message):
            interface._validate_inputs(bad, k, v)


@pytest.mark.parametrize(
    "tensors,scalars", [(None, None), ([], ()), ([], []), ([torch.ones(2)], (0.25,))]
)
def test_aux_none_empty_and_dynamic_values(tensors, scalars):
    actual_tensors, actual_scalars = interface._validate_aux(tensors, scalars, "cpu")
    assert (actual_tensors is None) == (tensors is None)
    assert actual_scalars == (tuple(scalars) if scalars else None)
    if tensors:
        assert actual_tensors[0] is tensors[0]


@pytest.mark.parametrize(
    "tensors,scalars,error",
    [
        (object(), None, "tuple or a list"),
        ([object()], None, "Tensor"),
        (None, [torch.tensor(1)], "scalar"),
    ],
)
def test_invalid_aux(tensors, scalars, error):
    with pytest.raises(TypeError, match=error):
        interface._validate_aux(tensors, scalars, "cpu")


@pytest.fixture
def host(monkeypatch):
    state = SimpleNamespace(compiled=[], launched=[], converted=[], current="previous")

    class HostTensor:
        def __init__(self, tensor, *, layout_tag):
            self.owner = tensor
            self.dynamic_mode = None
            state.converted.append(tensor)

        def mark_compact_shape_dynamic(self, mode):
            self.dynamic_mode = mode
            return self

        def data_ptr(self):
            return self.owner.data_ptr()

    @contextmanager
    def guard(device):
        previous, state.current = state.current, device
        try:
            yield
        finally:
            state.current = previous

    def compile(kernel, *args, **kwargs):
        assert kwargs == {"options": "--npu-arch 3510"}
        state.compiled.append(args)
        return lambda *args, **kwargs: state.launched.append((args, kwargs))

    monkeypatch.setattr(interface, "_validate_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(interface, "_check_launch_tensors", lambda tensors: None)
    monkeypatch.setattr(interface, "from_dlpack", HostTensor)
    monkeypatch.setattr(interface.tla, "compile", compile)
    monkeypatch.setenv("CATLASS_DSL_CACHE", "1")
    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "0")
    interface._compiled_kernels.clear()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device=guard,
            get_device_properties=lambda index: SimpleNamespace(cube_core_num=2),
        ),
        raising=False,
    )
    state.inputs = (
        torch.empty(1, 17, 4, 96, dtype=torch.bfloat16),
        torch.empty(1, 65, 2, 96, dtype=torch.bfloat16),
        torch.empty(1, 65, 2, 96, dtype=torch.bfloat16),
    )
    yield state
    interface._compiled_kernels.clear()


@pytest.mark.parametrize(
    "lse,tensors,scalars",
    [(False, None, None), (True, [], ()), (True, [torch.ones(2)], (0.25,))],
)
def test_public_call_preserves_logical_aux_and_optional_lse(host, lse, tensors, scalars):
    out, result_lse = interface.flash_attn_func(
        *host.inputs, aux_tensors=tensors, aux_scalars=scalars, return_lse=lse
    )
    assert len(host.compiled) == len(host.launched) == 1
    assert len(host.compiled[0]) == 32
    assert host.compiled[0][-1] is None  # No separate BSND capacity on the ordinary path.
    args, kwargs = host.launched[0]
    assert len(args) == 9 and kwargs == {"block_num": 2}
    assert args[8] is None  # BSND has no packed sequence metadata.
    assert (args[4] is None) == (not lse)
    assert args[5] is None
    assert (args[6] is None) == (tensors is None)
    assert args[7] == (tuple(scalars) if scalars else None)
    if tensors is not None:
        assert all(actual is original for actual, original in zip(args[6], tensors))
        assert all(
            sample.owner is original for sample, original in zip(host.compiled[0][6], tensors)
        )
    dynamic = tensors is None and not scalars
    assert all(tensor.dynamic_mode == (0 if dynamic else None) for tensor in host.compiled[0][:4])
    assert host.compiled[0][11:13] == ((None, None) if dynamic else (34, 65))
    assert host.compiled[0][21:24] == (96, None, 2)
    assert out.shape == host.inputs[0].shape
    assert (result_lse is None) == (not lse)
    assert host.current == "previous"


def test_deterministic_inference_reuses_the_same_compiled_call(host):
    for deterministic in (False, True, False):
        interface.flash_attn_func(*host.inputs, deterministic=deterministic)
    assert len(host.compiled) == 1
    assert len(host.launched) == 3


@pytest.mark.parametrize("dynamic", [False, True])
def test_value_dimension_controls_output_binding_and_cache(host, dynamic):
    q, k, _ = host.inputs
    kwargs = {} if dynamic else {"mask_mod": causal_mask}
    for dim in (64, 128, 64):
        v = torch.empty(*k.shape[:-1], dim, dtype=k.dtype)
        out, lse = interface.flash_attn_func(q, k, v, **kwargs)
        assert out.shape == (*q.shape[:-1], dim)
        assert lse is None
        runtime_v, runtime_out = host.launched[-1][0][2:4]
        if dynamic:
            assert runtime_v.columns == k.shape[-2] * dim
            assert runtime_out.columns == q.shape[-2] * dim
        else:
            assert runtime_v is v and runtime_out is out
    assert len(host.compiled) == 2
    assert [args[28] for args in host.compiled] == [64, 128]


def test_public_call_restores_device_guard_on_compile_error(host, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(interface.tla, "compile", fail)
    with pytest.raises(RuntimeError, match="compile failed"):
        interface.flash_attn_func(*host.inputs)
    assert not host.launched and host.current == "previous"


def test_public_call_checks_external_storage_not_new_outputs(host, monkeypatch):
    checked = []
    monkeypatch.setattr(interface, "_check_launch_tensors", lambda tensors: checked.extend(tensors))
    aux = torch.ones(4)
    out, lse = interface.flash_attn_func(*host.inputs, aux_tensors=[aux], return_lse=True)
    expected = (*host.inputs, aux)
    assert len(checked) == len(expected) + 1
    assert checked[3] is None  # No page-table input in a contiguous call.
    assert all(
        actual is original
        for actual, original in zip((x for x in checked if x is not None), expected)
    )
    assert all(tensor is not out and tensor is not lse for tensor in checked)


@pytest.mark.parametrize(
    "batch,q_len,heads,kv_heads,cores,sparse,expected",
    [
        (650, 2, 49, 1, 32, None, 49),
        (2, 64, 6, 1, 32, None, 1),
        (32, 129, 33, 1, 32, None, 1),
        (32, 256, 33, 1, 32, None, 1),
        (32, 2, 48, 3, 32, None, 16),
        (13, 35, 4, 1, 28, None, 1),
        (14, 35, 4, 1, 28, None, 4),
        (32, 2, 49, 1, 32, SimpleNamespace(strides=((0, 1, 1),)), 1),
    ],
)
def test_packed_head_policy_preserves_parallelism(
    batch, q_len, heads, kv_heads, cores, sparse, expected
):
    packed_heads, _, _ = interface._select_kernel_options(
        batch,
        q_len,
        257,
        heads,
        kv_heads,
        cores,
        sparse,
        has_simt_mod=False,
        has_score_mod=False,
        has_aux=False,
        has_mod=False,
        has_window=False,
        scale=0.125,
    )
    assert packed_heads == expected


@pytest.mark.parametrize(
    "options", [{}, {"causal": True}, {"window_size": (7, 3)}, {"window_size": (0, None)}]
)
def test_packed_public_call_preserves_layout_and_reuses_lengths(host, options):
    k = torch.empty(32, 129, 1, 64, dtype=torch.float16)
    for q_len in (5, 7, 32, 33):
        q = torch.empty(32, q_len, 33, 64, dtype=torch.float16)
        out, lse = interface.flash_attn_func(q, k, k, return_lse=True, **options)
        assert out.shape == q.shape and out.is_contiguous()
        assert lse.shape == (32, 33, q_len) and lse.is_contiguous()
        launch = host.launched[-1][0]
        assert launch[0].tensor.shape == (32, q_len * 33, 1, 64)
        assert launch[0].tensor.data_ptr() == q.data_ptr()
        assert q.shape == (32, q_len, 33, 64)
    assert len(host.compiled) == 1
    assert host.compiled[0][23] == 33


def test_pack_gqa_false_disables_packing_without_creating_another_interface(host):
    q = torch.empty(32, 3, 33, 64, dtype=torch.float16)
    k = torch.empty(32, 129, 1, 64, dtype=torch.float16)
    for choice, expected in ((None, 33), (False, 1), (None, 33)):
        out, lse = interface.flash_attn_func(q, k, k, pack_gqa=choice)
        assert out.shape == q.shape and lse is None
        argument = host.launched[-1][0][0]
        assert argument.tensor.shape[1:3] == (3 * expected, 33 // expected)
    assert [args[23] for args in host.compiled] == [33, 1]
    with pytest.raises(NotImplementedError, match="pack_gqa"):
        interface.flash_attn_func(q, k, k, pack_gqa=True)


@pytest.mark.parametrize(
    "mask,score,q_len,packed_heads",
    [
        (causal_mask, None, 5, 33),
        (None, scale_score, 5, 33),
        (causal_mask, vector_score, 5, 33),
        (vector_mask, scale_score, 5, 33),
        (vector_mask, vector_score, 5, 33),
        (causal_mask, scale_score, 32, 33),
        (causal_mask, None, 33, 1),
        (None, scale_score, 63, 1),
        (vector_mask, scale_score, 64, 1),
        (vector_mask, vector_score, 64, 33),
    ],
)
def test_callback_packing_limits_simt_to_short_queries(host, mask, score, q_len, packed_heads):
    q = torch.empty(32, q_len, 33, 64, dtype=torch.float16)
    k = torch.empty(32, 17, 1, 64, dtype=torch.float16)
    interface.flash_attn_func(q, k, k, mask_mod=mask, score_mod=score)
    assert host.compiled[0][23] == packed_heads


@pytest.mark.parametrize(
    "q_len,kv_len,options,expected",
    [
        (127, 512, {}, (False, False)),
        (128, 255, {}, (False, False)),
        (128, 256, {}, (True, True)),
        (128, 257, {}, (True, False)),
        (129, 512, {}, (False, False)),
        (191, 512, {}, (False, False)),
        (192, 255, {}, (False, False)),
        (192, 256, {}, (True, False)),
        (399, 512, {}, (True, False)),
        (511, 512, {}, (True, False)),
        (513, 511, {}, (True, False)),
        (512, 512, {}, (True, True)),
        (513, 512, {}, (True, False)),
        (513, 513, {"mask_mod": vector_mask}, (True, False)),
        (513, 513, {"mask_mod": causal_mask}, (False, False)),
        (513, 513, {"aux_scalars": (1,)}, (False, False)),
        (128, 256, {"mask_mod": vector_mask}, (True, False)),
        (128, 256, {"score_mod": vector_score}, (True, False)),
        (128, 256, {"mask_mod": causal_mask}, (False, False)),
        (128, 256, {"mask_mod": vector_mask, "score_mod": scale_score}, (False, False)),
        (128, 256, {"aux_scalars": (1,)}, (False, False)),
        (128, 256, {"aux_tensors": []}, (False, False)),
        (128, 256, {"softcap": 5.0}, (True, False)),
        (128, 256, {"softmax_scale": 0.0}, (True, False)),
        (128, 256, {"softmax_scale": -0.125}, (True, False)),
    ],
)
def test_public_score_layout_selection(host, q_len, kv_len, options, expected):
    q = torch.empty(1, q_len, 4, 64, dtype=torch.float16)
    k = torch.empty(1, kv_len, 2, 64, dtype=torch.float16)
    interface.flash_attn_func(q, k, k, **options)
    assert host.compiled[0][24:26] == expected
    assert len(host.launched[0][0]) == 9


@pytest.mark.parametrize(
    "average_q_len,has_simt_mod,has_aux,expected",
    [
        (None, False, False, False),
        (191, False, False, False),
        (192, False, False, True),
        (255, True, False, False),
        (255, False, True, False),
    ],
)
def test_short_variable_query_dn_requires_known_utilization(
    average_q_len, has_simt_mod, has_aux, expected
):
    _, use_dn, full_dn = interface._select_kernel_options(
        4,
        255,
        512,
        4,
        4,
        28,
        None,
        has_simt_mod=has_simt_mod,
        has_aux=has_aux,
        has_mod=True,
        has_window=False,
        scale=0.125,
        has_score_mod=False,
        variable_lengths=True,
        average_q_len=average_q_len,
    )
    assert use_dn is expected
    assert full_dn is False


@pytest.mark.parametrize(
    "q_len,options,packed_heads,expected",
    [
        (4, {}, 32, (True, True)),
        (17, {}, 32, (False, False)),
        (4, {"causal": True}, 32, (False, False)),
        (4, {"window_size": (128, 32)}, 32, (False, False)),
        (4, {"window_size": (0, None)}, 32, (False, False)),
        (4, {"causal": True, "score_mod": vector_score}, 32, (False, False)),
        (32, {"causal": True}, 32, (False, False)),
        (33, {"causal": True}, 32, (False, False)),
        (64, {"window_size": (128, 32)}, 32, (False, False)),
        (33, {"causal": True, "score_mod": vector_score}, 32, (False, False)),
        (33, {"causal": True, "score_mod": scale_score}, 1, (False, False)),
    ],
)
def test_packed_window_layout_and_query_limit(host, q_len, options, packed_heads, expected):
    q = torch.empty(32, q_len, 32, 64, dtype=torch.float16)
    k = torch.empty(32, 256, 1, 64, dtype=torch.float16)
    interface.flash_attn_func(q, k, k, **options)
    assert host.compiled[0][23] == packed_heads
    assert host.compiled[0][24:26] == expected


def test_dynamic_cache_separates_score_layouts_and_reuses_within_each(host):
    for q_len, kv_len, count in (
        (128, 256, 1),
        (256, 384, 1),
        (128, 257, 2),
        (256, 513, 2),
        (513, 512, 2),
        (577, 1025, 2),
        (129, 257, 3),
        (191, 512, 3),
        (192, 512, 3),
        (127, 513, 3),
        (128, 256, 3),
    ):
        q = torch.empty(1, q_len, 4, 64, dtype=torch.float16)
        k = torch.empty(1, kv_len, 2, 64, dtype=torch.float16)
        interface.flash_attn_func(q, k, k)
        assert len(host.compiled) == count


def test_public_call_reuses_compilation_with_current_buffers_and_scalars(host, monkeypatch):
    aux = torch.ones(2)
    first, _ = interface.flash_attn_func(*host.inputs, aux_tensors=[aux], aux_scalars=(0.25,))
    inputs, updated_aux = tuple(t.clone() for t in host.inputs), aux.clone()
    with monkeypatch.context() as warm:
        warm.setattr(
            interface, "from_dlpack", lambda *a, **kw: pytest.fail("hot DLPack conversion")
        )
        second, _ = interface.flash_attn_func(
            *inputs, aux_tensors=[updated_aux], aux_scalars=(0.75,)
        )
    assert len(host.compiled) == 1 and len(host.launched) == 2
    args = host.launched[-1][0]
    assert args[0].data_ptr() == inputs[0].data_ptr()
    assert args[0].data_ptr() != host.launched[0][0][0].data_ptr()
    assert args[3].data_ptr() == second.data_ptr() != first.data_ptr()
    assert args[6][0] is updated_aux and args[7] == (0.75,)

    variants = (
        (inputs, {"softmax_scale": 0.5}),
        (inputs, {"return_lse": True}),
        (inputs, {"aux_tensors": None}),
        (inputs, {"aux_tensors": []}),
        (inputs, {"aux_scalars": None}),
        (inputs, {"aux_scalars": (1,)}),
        (tuple(t.to(torch.float16) for t in inputs), {}),
        (tuple(t[..., :64].contiguous() for t in inputs), {}),
    )
    for expected, (current, overrides) in enumerate(variants, start=2):
        options = dict(aux_tensors=[updated_aux], aux_scalars=(0.75,))
        interface.flash_attn_func(*current, **(options | overrides))
        assert len(host.compiled) == expected
        assert len(host.launched) == expected + 1


@pytest.mark.parametrize("options", [{}, {"causal": True}, {"window_size": (7, 3)}])
def test_public_call_reuses_dynamic_lengths_with_current_buffers(host, monkeypatch, options):
    def unexpected_conversion(*args, **kwargs):
        pytest.fail("cache hits must not construct DLPack compilation samples")

    seen_modes = set()
    for q_len, kv_len in ((512, 257), (768, 513), (1024, 1024), (512, 257)):
        mode = (True, not options and kv_len % 128 == 0)
        inputs = (
            torch.empty(1, q_len, 4, 96, dtype=torch.bfloat16),
            torch.empty(1, kv_len, 2, 96, dtype=torch.bfloat16),
            torch.empty(1, kv_len, 2, 96, dtype=torch.bfloat16),
        )
        with monkeypatch.context() as context:
            if mode in seen_modes:
                context.setattr(interface, "from_dlpack", unexpected_conversion)
            out, lse = interface.flash_attn_func(*inputs, **options)
        seen_modes.add(mode)
        assert len(host.compiled) == len(seen_modes)
        assert host.compiled[0][11:13] == (None, None)
        assert all(tensor.dynamic_mode == 0 for tensor in host.compiled[0][:4])
        args = host.launched[-1][0]
        for argument, tensor in zip(args[:4], (*inputs, out)):
            assert argument.tensor is tensor
            assert argument.build_memref_launch_fields()[0] == tensor.data_ptr()
        assert args[0].build_memref_launch_fields()[3:5] == (q_len, 4 * 96)
        assert args[1].build_memref_launch_fields()[3:5] == (kv_len, 2 * 96)
        assert out.shape == inputs[0].shape and lse is None
        assert len(host.converted) == 4 * len(seen_modes)

    out, lse = interface.flash_attn_func(*inputs, return_lse=True, **options)
    assert len(host.compiled) == len(seen_modes) + 1
    assert lse.shape == (1, 4, q_len)
    args = host.launched[-1][0]
    assert args[4].tensor is lse
    assert args[4].build_memref_launch_fields()[0] == lse.data_ptr()
    assert args[4].build_memref_launch_fields()[3:5] == (4 * q_len, 1)
    # Short queries select ND, independent of the DN executables above.
    interface.flash_attn_func(*host.inputs, return_lse=True, **options)
    expected_variants = len(seen_modes) + 2
    assert len(host.compiled) == expected_variants
    assert len(host.converted) == 4 * len(seen_modes) + 10
    with monkeypatch.context() as context:
        context.setattr(interface, "from_dlpack", unexpected_conversion)
        interface.flash_attn_func(*host.inputs, return_lse=True, **options)
    assert len(host.compiled) == expected_variants


def test_dynamic_length_cache_keeps_batch_heads_dimension_and_dtype(host):
    variants = (
        (1, 4, 2, 96, torch.bfloat16),
        (2, 4, 2, 96, torch.bfloat16),
        (1, 8, 2, 96, torch.bfloat16),
        (1, 4, 1, 96, torch.bfloat16),
        (1, 4, 2, 64, torch.bfloat16),
        (1, 4, 2, 96, torch.float16),
    )
    for count, (batch, q_heads, kv_heads, dim, dtype) in enumerate(variants, start=1):
        q = torch.empty(batch, 17, q_heads, dim, dtype=dtype)
        k = torch.empty(batch, 65, kv_heads, dim, dtype=dtype)
        interface.flash_attn_func(q, k, torch.empty_like(k))
        assert len(host.compiled) == count
        argument = host.launched[-1][0][0]
        assert argument.tensor is q
        assert argument.build_memref_launch_fields()[3:5] == (batch * 17, q_heads * dim)


@pytest.mark.parametrize(
    "cache,force,compilations",
    [(" 0 ", "0", 2), ("1", " 1 ", 2), ("1", " 0 ", 1)],
)
def test_public_call_respects_cache_environment(host, monkeypatch, cache, force, compilations):
    monkeypatch.setenv("CATLASS_DSL_CACHE", cache)
    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", force)
    interface.flash_attn_func(*host.inputs)
    interface.flash_attn_func(*host.inputs)
    assert len(host.compiled) == compilations and len(host.launched) == 2
    assert len(host.converted) == 4 * compilations


@pytest.mark.parametrize(
    "mask,score,window,expected,modes",
    [
        (None, None, (-1, -1), (None, 0), (False, False)),
        (None, None, (7, 3), (7, 0), (False, False)),
        (causal_mask, None, (7, 3), (None, None), (False, False)),
        (vector_mask, None, (7, 3), (None, None), (True, False)),
        (None, scale_score, (7, 3), (7, 0), (False, False)),
        (None, vector_score, (7, 3), (7, 0), (False, True)),
        (causal_mask, scale_score, (7, 3), (None, None), (False, False)),
        (causal_mask, vector_score, (7, 3), (None, None), (False, True)),
        (vector_mask, scale_score, (7, 3), (None, None), (True, False)),
        (vector_mask, vector_score, (7, 3), (None, None), (True, True)),
    ],
)
def test_public_callbacks_select_independent_modes_and_mask_overrides_window(
    host, mask, score, window, expected, modes
):
    interface.flash_attn_func(
        *host.inputs, mask_mod=mask, score_mod=score, causal=True, window_size=window
    )
    args = host.compiled[0]
    assert args[15] is mask and args[16] is score
    assert args[17:19] == expected and args[19:21] == modes
    assert args[21:24] == (96, None, 2)
    assert args[11:13] == ((None, None) if mask is None and score is None else (34, 65))
    assert host.launched[0][0][5] is None


@pytest.mark.parametrize("scale", [float("nan"), float("inf")])
def test_nonfinite_scale_rejected(host, scale):
    with pytest.raises(ValueError, match="finite"):
        interface.flash_attn_func(*host.inputs, softmax_scale=scale)
    assert not host.compiled and not host.launched


@pytest.mark.parametrize(
    "window,expected",
    [
        ((-1, 3), (None, 3)),
        ((7, -1), (7, None)),
        ((None, -1), (None, None)),
        ((0, 0), (0, 0)),
        ((None, 3), (None, 3)),
    ],
)
def test_window_sides_normalize_independently(host, window, expected):
    interface.flash_attn_func(*host.inputs, window_size=window)
    assert host.compiled[0][17:19] == expected


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"causal": True}, (None, None)),
        ({"window_size": (None, 3)}, (None, None)),
        ({"window_size": (7, 0)}, (7, None)),
        ({"causal": True, "window_size": (0, None)}, (0, None)),
        ({"causal": True, "score_mod": vector_score}, (None, None)),
        ({"causal": True, "window_size": (7, 3), "mask_mod": vector_mask}, (None, None)),
    ],
)
def test_single_query_removes_only_redundant_right_bound(host, options, expected):
    q = torch.empty(32, 1, 33, 64, dtype=torch.float16)
    k = torch.empty(32, 129, 1, 64, dtype=torch.float16)
    interface.flash_attn_func(q, k, k, **options)
    assert host.compiled[0][17:19] == expected
    assert host.compiled[0][15] is options.get("mask_mod")
    assert host.compiled[0][16] is options.get("score_mod")


def test_single_query_causal_shares_dense_cache_not_longer_causal(host):
    k = torch.empty(32, 129, 1, 64, dtype=torch.float16)
    for q_len, causal, variants in ((1, True, 1), (1, False, 1), (2, True, 2), (2, False, 2)):
        q = torch.empty(32, q_len, 33, 64, dtype=torch.float16)
        interface.flash_attn_func(q, k, k, causal=causal)
        assert len(host.compiled) == variants
    assert host.compiled[0][17:19] == (None, None)
    assert host.compiled[1][17:19] == (None, 0)


@pytest.mark.parametrize("option,value", [("qv", object()), ("num_splits", 2)])
def test_unsupported_options_rejected_before_compile(host, option, value):
    with pytest.raises(NotImplementedError, match=option):
        interface.flash_attn_func(*host.inputs, **{option: value})
    assert not host.compiled and not host.launched


@pytest.mark.parametrize("cap", [-1.0, float("nan"), float("inf")])
def test_invalid_softcap_rejected(host, cap):
    with pytest.raises(ValueError, match="softcap"):
        interface.flash_attn_func(*host.inputs, softcap=cap)
    assert not host.compiled and not host.launched


def test_softcap_uses_stable_vector_callback_and_separate_variants(host):
    interface.flash_attn_func(*host.inputs, softcap=2.0)
    interface.flash_attn_func(*host.inputs, softcap=2.0)
    assert len(host.compiled) == 1
    callback = host.compiled[0][16]
    assert callback._use_simd is True
    interface.flash_attn_func(*host.inputs, softcap=50.0)
    assert len(host.compiled) == 2 and host.compiled[1][16] is not callback
    interface.flash_attn_func(*host.inputs, softcap=0.0)
    assert len(host.compiled) == 3 and host.compiled[2][16] is None
    with pytest.raises(ValueError, match="softcap and score_mod"):
        interface.flash_attn_func(*host.inputs, softcap=2.0, score_mod=scale_score)
