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
        "simd",
    }


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
        ((1, 17, 4, 72), "head dimension"),
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

    monkeypatch.setattr(interface, "_validate_inputs", lambda *args: None)
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
    assert len(host.compiled[0]) == 24
    args, kwargs = host.launched[0]
    assert len(args) == 8 and kwargs == {"block_num": 2}
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
    assert host.compiled[0][-3:] == (96, None, 2)
    assert out.shape == host.inputs[0].shape
    assert (result_lse is None) == (not lse)
    assert host.current == "previous"


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
    assert len(checked) == len(expected)
    assert all(actual is original for actual, original in zip(checked, expected))
    assert all(tensor is not out and tensor is not lse for tensor in checked)


@pytest.mark.parametrize(
    "batch,q_len,heads,kv_heads,cores,sparse,window,expected",
    [
        (650, 2, 49, 1, 32, None, (None, None), 49),
        (2, 64, 6, 1, 32, None, (None, None), 1),
        (32, 129, 33, 1, 32, None, (None, None), 1),
        (32, 256, 33, 1, 32, None, (None, None), 1),
        (32, 2, 48, 3, 32, None, (None, None), 16),
        (32, 2, 49, 1, 32, SimpleNamespace(strides=((0, 1, 1),)), (None, None), 1),
        (32, 2, 49, 1, 32, None, (None, 0), 1),
    ],
)
def test_packed_head_policy_preserves_parallelism(
    batch, q_len, heads, kv_heads, cores, sparse, window, expected
):
    assert (
        interface._select_packed_heads(batch, q_len, heads, kv_heads, cores, sparse, *window)
        == expected
    )


def test_packed_public_call_preserves_layout_and_reuses_lengths(host):
    k = torch.empty(32, 129, 1, 64, dtype=torch.float16)
    for q_len in (5, 7):
        q = torch.empty(32, q_len, 33, 64, dtype=torch.float16)
        out, lse = interface.flash_attn_func(q, k, k, return_lse=True)
        assert out.shape == q.shape and out.is_contiguous()
        assert lse.shape == (32, 33, q_len) and lse.is_contiguous()
        launch = host.launched[-1][0]
        assert launch[0].tensor.shape == (32, q_len * 33, 1, 64)
        assert launch[0].tensor.data_ptr() == q.data_ptr()
        assert q.shape == (32, q_len, 33, 64)
    assert len(host.compiled) == 1
    assert host.compiled[0][-1] == 33


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
    assert host.compiled[0][-1] == packed_heads


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

    for q_len, kv_len in ((512, 257), (768, 513), (1024, 1024), (512, 257)):
        inputs = (
            torch.empty(1, q_len, 4, 96, dtype=torch.bfloat16),
            torch.empty(1, kv_len, 2, 96, dtype=torch.bfloat16),
            torch.empty(1, kv_len, 2, 96, dtype=torch.bfloat16),
        )
        with monkeypatch.context() as context:
            if host.compiled:
                context.setattr(interface, "from_dlpack", unexpected_conversion)
            out, lse = interface.flash_attn_func(*inputs, **options)
        assert len(host.compiled) == 1
        assert host.compiled[0][11:13] == (None, None)
        assert all(tensor.dynamic_mode == 0 for tensor in host.compiled[0][:4])
        args = host.launched[-1][0]
        for argument, tensor in zip(args[:4], (*inputs, out)):
            assert argument.tensor is tensor
            assert argument.build_memref_launch_fields()[0] == tensor.data_ptr()
        assert args[0].build_memref_launch_fields()[3:5] == (q_len, 4 * 96)
        assert args[1].build_memref_launch_fields()[3:5] == (kv_len, 2 * 96)
        assert out.shape == inputs[0].shape and lse is None
        assert len(host.converted) == 4

    out, lse = interface.flash_attn_func(*inputs, return_lse=True, **options)
    assert len(host.compiled) == 2
    assert lse.shape == (1, 4, q_len)
    args = host.launched[-1][0]
    assert args[4].tensor is lse
    assert args[4].build_memref_launch_fields()[0] == lse.data_ptr()
    assert args[4].build_memref_launch_fields()[3:5] == (4 * q_len, 1)
    # A short dense GQA now selects the packed specialization; window paths do not.
    interface.flash_attn_func(*host.inputs, return_lse=True, **options)
    expected_variants = 3 if not options else 2
    assert len(host.compiled) == expected_variants
    assert len(host.converted) == (14 if not options else 9)
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
    assert args[21:] == (96, None, 2 if mask is not None else 1)
    assert args[11:13] == (
        (None, None) if mask is None and score is None else (34 if mask is not None else 17, 65)
    )
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
