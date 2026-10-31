"""Public signatures and the essential validation/autograd contracts."""

import ast
import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from .test_utils import check, metadata_tensor, oracle

from flash_attn_npu_4.catlass_dsl import (
    compute_block_sparsity,
    flash_attn_func,
    flash_attn_varlen_func,
    interface,
    runtime,
)

ENTRIES = (flash_attn_func, flash_attn_varlen_func)

# Expected public signatures, stored locally for independent contract checks.
SIGNATURES = {
    "compute_block_sparsity": """
        tile_m,
        tile_n,
        batch_size,
        num_heads,
        seqlen_q,
        seqlen_k,
        mask_mod: Callable,
        aux_tensors: Optional[list],
        device,
        aux_scalars: Optional[tuple] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        cu_total_m_blocks: Optional[torch.Tensor] = None,
        cu_block_idx_offsets: Optional[torch.Tensor] = None,
        compute_full_blocks: bool = True,
        use_fast_sampling: bool = False
    """,
    "flash_attn_func": """
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
        block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
        return_lse: bool = False,
        gather_bwd_recompute_p: bool = False,
        gather_bwd_token_chunk: Optional[int] = None
    """,
    "flash_attn_varlen_func": """
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        min_seqlen_k: Optional[int] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        return_lse: bool = False,
        scheduler_metadata: Optional[SchedulerMetadataTensorsTorch] = None,
        seqlen_k_per_split: Optional[int] = None,
        disable_scheduler_metadata: bool = False,
        gather_bwd_recompute_p: bool = False,
        gather_bwd_token_chunk: Optional[int] = None
    """,
}


@pytest.mark.parametrize("entry", (*ENTRIES, compute_block_sparsity))
def test_public_signature(entry):
    actual = inspect.signature(entry)
    source = ast.parse(Path(inspect.getfile(entry)).read_text())
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == entry.__name__
    )
    result = " -> BlockSparseTensorsTorch" if entry is compute_block_sparsity else ""
    expected = ast.parse(f"def entry({SIGNATURES[entry.__name__]}){result}: pass").body[0]
    assert ast.dump(function.args, include_attributes=False) == ast.dump(
        expected.args, include_attributes=False
    )
    if expected.returns is not None:
        assert ast.dump(function.returns, include_attributes=False) == ast.dump(
            expected.returns, include_attributes=False
        )
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in actual.parameters.values()
    )


@pytest.mark.parametrize("parameter", ["mask_mod_ref", "mask_mod_simd"])
def test_sparse_builder_rejects_extensions(parameter):
    with pytest.raises(TypeError, match=parameter):
        compute_block_sparsity(128, 128, 1, 1, 17, 23, None, None, "cpu", **{parameter: None})


@pytest.mark.parametrize("missing", ["aux_tensors", "device"])
def test_sparse_builder_requires_explicit_aux_and_device(missing):
    keywords = {"aux_tensors": None, "device": "cpu"}
    del keywords[missing]
    with pytest.raises(TypeError, match=missing):
        compute_block_sparsity(128, 128, 1, 1, 17, 23, None, **keywords)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "parameter", ["layout", "score_mod_simd", "score_mod_bwd_simd", "mask_mod_simd"]
)
def test_removed_extensions(entry, parameter):
    q = torch.empty(1, 7, 2, 128)
    with pytest.raises(TypeError, match=parameter):
        entry(q, q, q, **{parameter: None})


def test_varlen_rejects_explicit_backward_lists():
    q = torch.empty(1, 7, 2, 128)
    with pytest.raises(TypeError, match="block_sparse_tensors_bwd"):
        flash_attn_varlen_func(q, q, q, block_sparse_tensors_bwd=None)


@pytest.mark.parametrize("entry", ENTRIES)
def test_positional_and_keyword_binding(entry, monkeypatch):
    captured = []
    monkeypatch.setattr(
        interface,
        "_flash",
        lambda *args, **kwargs: captured.append((args, kwargs)) or (args[0], None),
    )
    q = torch.empty(1, 7, 2, 128)
    signature = inspect.signature(entry)
    arguments = {name: p.default for name, p in signature.parameters.items()}
    arguments.update(q=q, k=q, v=q)
    assert entry(*arguments.values())[0] is q
    assert entry(**arguments)[0] is q
    assert captured[0] == captured[1]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "options", [{"gather_bwd_recompute_p": True}, {"gather_bwd_token_chunk": 2}]
)
def test_gather_options_require_indices(entry, options):
    q = torch.empty(1, 7, 2, 128)
    with pytest.raises(ValueError, match="require gather_kv_indices"):
        entry(q, q, q, **options)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("chunk", [True, 0, 1.5])
def test_invalid_gather_chunk_precedes_capability_check(entry, chunk):
    q = torch.empty(1, 7, 2, 128)
    with pytest.raises(ValueError, match="positive int"):
        entry(q, q, q, gather_kv_indices=torch.empty(1), gather_bwd_token_chunk=chunk)


@pytest.fixture
def cpu_launch(monkeypatch):
    """Exercise real wrappers/autograd with an explicitly substituted test oracle."""
    validate = interface._validate

    def cpu_validate(q, k, v, aux, scalars, scale, cq, ck, mq, mk, **kw):
        cfg = validate(
            *(metadata_tensor(t.shape, t.dtype) for t in (q, k, v)),
            aux,
            scalars,
            scale,
            None if cq is None else metadata_tensor(cq.shape, cq.dtype),
            None if ck is None else metadata_tensor(ck.shape, ck.dtype),
            mq,
            mk,
            **kw,
        )
        cfg.update(cu_q=cq, cu_k=ck)
        return cfg

    def launch(q, k, v, out, lse, config, **kw):
        if "dout" not in kw:
            wanted, logsum = oracle(
                (q, k, v), cu_q=config["cu_q"], cu_k=config["cu_k"], scale=config["scale"]
            )
            out.copy_(wanted)
            lse.copy_(logsum)
            return
        with torch.enable_grad():
            inputs = [t.detach().requires_grad_() for t in (q, k, v)]
            wanted, logsum = oracle(
                inputs, cu_q=config["cu_q"], cu_k=config["cu_k"], scale=config["scale"]
            )
            outputs, upstream = [wanted], [kw["dout"]]
            if kw.get("dlse") is not None:
                outputs.append(logsum)
                upstream.append(kw["dlse"])
            grads = torch.autograd.grad(outputs, inputs, upstream)
        for name, grad in zip(("dq", "dk", "dv"), grads):
            kw[name].copy_(grad)

    monkeypatch.setattr(interface, "_validate", cpu_validate)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device=lambda *args: nullcontext()), raising=False
    )
    monkeypatch.setattr(runtime, "launch_attention", launch)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_return_matrix(cpu_launch, entry, requires_grad, grad_enabled, return_lse):
    q = torch.randn(1, 2, 7, 128, dtype=torch.float16, requires_grad=requires_grad).transpose(1, 2)
    with torch.set_grad_enabled(grad_enabled):
        out, lse = entry(q, q, q, return_lse=return_lse)
    assert out.requires_grad == (requires_grad and grad_enabled)
    if out.requires_grad:
        expected_node = (
            "FlashAttnFuncBackward" if entry is flash_attn_func else "FlashAttnVarlenFuncBackward"
        )
        assert type(out.grad_fn).__name__ == expected_node
    assert (lse is not None) == (return_lse or requires_grad)
    if lse is not None:
        assert lse.shape == (1, 2, 7) and lse.dtype == torch.float32
        assert lse.requires_grad == (requires_grad and grad_enabled)


@pytest.mark.parametrize(
    "return_lse,mode", [(True, "out"), (True, "lse"), (True, "both"), (False, "lse")]
)
@pytest.mark.parametrize("entry", ENTRIES)
def test_cpu_lse_backward_plumbing(cpu_launch, return_lse, mode, entry):
    generator = torch.Generator().manual_seed(281)
    inputs = [
        torch.randn(1, 7, 2, 128, generator=generator, dtype=torch.float16).requires_grad_()
        for _ in range(3)
    ]
    reference = [t.detach().requires_grad_() for t in inputs]
    out, lse = entry(*inputs, return_lse=return_lse)
    expected, el = oracle(reference)
    dout = torch.randn_like(out) if mode != "lse" else torch.zeros_like(out)
    dlse = torch.randn_like(lse) if mode != "out" else torch.zeros_like(lse)
    targets = [out] if mode == "out" else [lse] if mode == "lse" else [out, lse]
    upstream = [dout] if mode == "out" else [dlse] if mode == "lse" else [dout, dlse]
    actual = torch.autograd.grad(targets, inputs, upstream)
    wanted = torch.autograd.grad(
        (expected, el), reference, (dout, dlse if return_lse else torch.zeros_like(dlse))
    )
    for a, b in zip(actual, wanted):
        torch.testing.assert_close(a, b)
    if mode == "lse":
        assert not actual[2].count_nonzero()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("hint", [None, 12, "tensor"])
def test_maximum_hints_without_device_readback(monkeypatch, packed, hint):
    q = torch.empty((9, 2, 128) if packed else (2, 7, 2, 128))
    cu = torch.tensor([0, 4, 9], dtype=torch.int32) if packed else None
    hint = torch.empty((), device="meta") if hint == "tensor" else hint
    captured = []
    monkeypatch.setattr(interface, "_flash", lambda *a, **kw: captured.append(kw) or (a[0], None))
    flash_attn_varlen_func(
        q, q, q, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=hint, max_seqlen_k=hint
    )
    expected = (12 if isinstance(hint, int) else 9) if packed else 7
    assert captured[0]["max_q"] == expected and captured[0]["max_k"] == expected


@pytest.mark.parametrize(
    "entry,options",
    [
        (flash_attn_func, {"qv": torch.empty(1)}),
        (flash_attn_func, {"gather_kv_indices": torch.empty(1)}),
        (flash_attn_func, {"learnable_sink": torch.empty(1)}),
        (flash_attn_func, {"num_splits": 2}),
        (flash_attn_func, {"pack_gqa": True}),
        (flash_attn_func, {"deterministic": True}),
        (flash_attn_varlen_func, {"page_table": torch.empty(1)}),
        (flash_attn_varlen_func, {"seqused_q": torch.empty(1)}),
        (flash_attn_varlen_func, {"seqused_k": torch.empty(1)}),
        (flash_attn_varlen_func, {"min_seqlen_k": 1}),
        (flash_attn_varlen_func, {"scheduler_metadata": object()}),
        (flash_attn_varlen_func, {"seqlen_k_per_split": 1}),
        (flash_attn_varlen_func, {"disable_scheduler_metadata": True}),
    ],
)
def test_unsupported_options(entry, options):
    q = torch.empty(1, 7, 2, 128)
    with pytest.raises(NotImplementedError):
        entry(q, q, q, **options)


def test_cpu_attention_requires_npu():
    q = torch.empty(1, 7, 2, 128)
    with pytest.raises(ValueError, match="requires an NPU"):
        flash_attn_func(q, q, q)


@pytest.mark.parametrize(
    "kind,message",
    [
        ("dtype", "float16 or bfloat16"),
        ("head_dim", "dimensions must be 128"),
        ("head_count", "divisible"),
        ("kv_shape", "k/v shapes must match"),
    ],
)
def test_invalid_inputs(cpu_launch, kind, message):
    shape = (1, 7, 3 if kind == "head_count" else 4, 64 if kind == "head_dim" else 128)
    dtype = torch.float32 if kind == "dtype" else torch.float16
    q = torch.empty(shape, dtype=dtype)
    k = torch.empty((1, 9, 2, shape[-1]), dtype=dtype)
    v = k[:, :8].contiguous() if kind == "kv_shape" else k
    with pytest.raises(ValueError, match=message):
        flash_attn_func(q, k, v)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"softcap": 1e-50}, "remain nonzero"),
        ({"softcap": float("inf")}, "finite float32"),
        ({"softcap": 2.0, "score_mod": lambda s: s}, "cannot be used together"),
    ],
)
def test_invalid_softcap(options, message):
    q = torch.empty(1, 7, 2, 128)
    with pytest.raises(ValueError, match=message):
        flash_attn_func(q, q, q, **options)


@pytest.mark.parametrize(
    "dtype,tolerance", [(torch.float16, 2**-10), (torch.bfloat16, 2**-7), (torch.float32, 2**-13)]
)
@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_mixed_tolerance_dtype_threshold(dtype, tolerance, factor):
    actual = torch.tensor([factor * tolerance], dtype=dtype)
    expected = torch.zeros(1, dtype=torch.float64)
    if factor < 1:
        check(actual, expected)
    else:
        with pytest.raises(AssertionError, match="matched_ratio"):
            check(actual, expected)


@pytest.mark.parametrize(
    "outliers,error,passes", [(1, 0.002, True), (2, 0.002, False), (1, 0.02, False)]
)
def test_mixed_tolerance_ratio_and_absolute_limit(outliers, error, passes):
    actual = torch.zeros(100, dtype=torch.float32)
    actual[:outliers] = error
    expected = torch.zeros_like(actual)
    if passes:
        check(actual, expected)
    else:
        with pytest.raises(AssertionError):
            check(actual, expected)


@pytest.mark.parametrize(
    "case", ["dynamic-pass", "dynamic-fail", "error-point-anchor", "largest-finite"]
)
def test_mixed_tolerance_ulp_anchor(case):
    if case.startswith("dynamic"):
        expected = torch.tensor([1e8], dtype=torch.float64)
        actual = (expected + (128 if case == "dynamic-pass" else 512)).float()
    elif case == "error-point-anchor":
        expected = torch.zeros(100, dtype=torch.float64)
        expected[0] = 1e8
        actual = expected.float()
        actual[-1] = 0.02
    else:
        expected = torch.full((100,), 65504.0, dtype=torch.float64)
        actual = expected.half()
        actual[-1] -= 1056  # 33 ULP: exceeds the finite 32-ULP hard limit.
    if case == "dynamic-pass":
        check(actual, expected)
    else:
        with pytest.raises(AssertionError, match="max_abs_error"):
            check(actual, expected)


def test_mixed_tolerance_empty_and_nonfinite_values():
    check(torch.empty(0, dtype=torch.float16), torch.empty(0, dtype=torch.float64))
    check(
        torch.tensor([torch.inf, -torch.inf], dtype=torch.float16),
        torch.tensor([70000.0, -70000.0]),
    )
    with pytest.raises(AssertionError):
        check(torch.tensor([-torch.inf]), torch.tensor([torch.inf]))
    with pytest.raises(AssertionError):
        check(torch.tensor([torch.nan]), torch.tensor([torch.nan]))
