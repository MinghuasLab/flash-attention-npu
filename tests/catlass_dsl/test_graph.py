"""Fullgraph plumbing on CPU oracles, plus independent Ascend precision tests.

CPU fixtures explicitly substitute the device launcher and sparse builder; they
do not provide a production fallback and do not establish NPU kernel accuracy.
"""

from contextlib import nullcontext

import pytest
import torch
from catlass import tla
from .reference import reference_block_sparsity
from .test_utils import check_attention, make_inputs, metadata_tensor, oracle

from flash_attn_npu_4.catlass_dsl import block_sparsity as sparsity
from flash_attn_npu_4.catlass_dsl import (
    builtin,
    compute_block_sparsity,
    flash_attn_func,
    flash_attn_varlen_func,
    graph,
    interface,
    runtime,
    simt,
)
from flash_attn_npu_4.catlass_dsl.block_sparsity import transpose_block_sparsity
from flash_attn_npu_4.catlass_dsl.mods import Mod, reference_callable


@tla.jit
def _score(s, b, h, q, k, info, aux, scalars):
    return s * scalars[0] + aux[0][k + info[3]]


@tla.jit
def _score_bwd(g, s, b, h, q, k, info, aux, scalars):
    return g * scalars[0]


def _score_ref(s, b, h, q, k, info, aux, scalars):
    return s * scalars[0] + aux[0][k + info[3]]


SCORE = Mod(_score, torch_ref=_score_ref)
SCORE_BWD = Mod(_score_bwd, torch_ref=lambda g, s, b, h, q, k, i, a, c: g * c[0])


@pytest.fixture
def cpu_runtime(monkeypatch):
    """Use real validation/configuration with a clearly substituted CPU kernel."""
    validate = interface._validate

    def meta(t):
        return (
            None if t is None else metadata_tensor(t.shape, t.dtype).requires_grad_(t.requires_grad)
        )

    def cpu_validate(q, k, v, aux, scalars, scale, cq, ck, mq, mk, **kw):
        config = validate(
            meta(q),
            meta(k),
            meta(v),
            tuple(meta(t) for t in aux),
            scalars,
            scale,
            meta(cq),
            meta(ck),
            mq,
            mk,
            **kw,
        )
        config.update(cu_q=cq, cu_k=ck, aux_tensors=aux)
        return config

    def mask_for(config):
        callback = reference_callable(config["mask_mod"], "mask")
        blocks = config["block_sparse_tensors"]
        if blocks is None:
            return callback

        def mask(b, h, q, k, info, aux, scalars):
            def contains(counts, indices):
                bi = 0 if counts.shape[0] == 1 else b
                hi = torch.zeros_like(h) if counts.shape[1] == 1 else h
                count = counts[bi, hi, q // 128]
                values = indices[bi, hi, q // 128]
                slots = torch.arange(values.shape[-1])
                return ((slots < count[..., None]) & (values == (k // 128)[..., None])).any(-1)

            partial = contains(blocks.mask_block_cnt, blocks.mask_block_idx)
            full = contains(blocks.full_block_cnt, blocks.full_block_idx)
            keep = True if callback is None else callback(b, h, q, k, info, aux, scalars)
            return full | (partial & keep)

        return mask

    def evaluate(inputs, config):
        score = config["score_mod"]
        if score is not None and score.torch_ref is None:
            score = score.simt
        return oracle(
            inputs,
            cu_q=config["cu_q"],
            cu_k=config["cu_k"],
            scale=config["scale"],
            score_mod=score,
            mask_mod=mask_for(config),
            aux_tensors=config["aux_tensors"],
            aux_scalars=config["aux_scalars"],
        )

    # Custom-op kernels execute below Autograd in the dispatcher. Restore the
    # ordinary CPU key sets only inside this test's differentiable oracle.
    included = torch._C._dispatch_tls_local_include_set()
    excluded = torch._C._dispatch_tls_local_exclude_set()

    def launch(q, k, v, out, lse, config, **kw):
        if "dout" not in kw:
            wanted, logsum = evaluate((q, k, v), config)
            out.copy_(wanted)
            lse.copy_(logsum)
            return
        with torch._C._ForceDispatchKeyGuard(included, excluded), torch.enable_grad():
            inputs = [t.detach().requires_grad_() for t in (q, k, v)]
            wanted, logsum = evaluate(inputs, config)
            dlse = kw.get("dlse")
            grads = torch.autograd.grad(
                (wanted, logsum),
                inputs,
                (kw["dout"], torch.zeros_like(logsum) if dlse is None else dlse),
            )
        for name, grad in zip(("dq", "dk", "dv"), grads):
            kw[name].copy_(grad)

    def builder(tile_m, tile_n, batch, heads, sq, sk, callback, aux, device, **kw):
        metadata = reference_block_sparsity(
            batch,
            heads,
            sq,
            sk,
            callback,
            aux_tensors=aux,
            aux_scalars=kw["aux_scalars"],
            cu_seqlens_q=kw["cu_seqlens_q"],
            cu_seqlens_k=kw["cu_seqlens_k"],
            compute_full_blocks=kw["compute_full_blocks"],
            use_fast_sampling=kw["use_fast_sampling"],
        )
        return metadata._replace(
            **{
                name: kw[name]
                for name in ("cu_total_m_blocks", "cu_block_idx_offsets")
                if kw.get(name) is not None
            }
        )

    monkeypatch.setattr(interface, "_validate", cpu_validate)
    # FA4 imports torch_npu up front. Keep its AMP/Dynamo integration intact
    # while isolating CPU-oracle tests from device initialization.
    monkeypatch.setattr(torch.npu, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.npu, "is_available", lambda: False)
    monkeypatch.setattr(runtime, "launch_attention", launch)
    monkeypatch.setattr(sparsity, "compute_block_sparsity", builder)


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def compile_counted(fn, *, dynamic=True):
    from torch._dynamo.backends.registry import lookup_backend

    graphs = []

    def backend(gm, inputs):
        # Float specialization can restart the initial trace. Count only
        # successful compilations, so shape reuse is measured independently.
        result = lookup_backend("aot_eager")(gm, inputs)
        graphs.append(gm)
        return result

    return torch.compile(fn, backend=backend, fullgraph=True, dynamic=dynamic), graphs


def compare_call(fn, inputs, **kwargs):
    reference = [t.detach().requires_grad_(t.requires_grad) for t in inputs]
    actual = fn(*inputs, **kwargs)
    expected = flash_attn_varlen_func(*reference, **kwargs)
    for a, b in zip(actual, expected):
        if b is None:
            assert a is None
        else:
            torch.testing.assert_close(a, b)
    if inputs[0].requires_grad and torch.is_grad_enabled():
        targets = [t for t in actual if t is not None]
        upstream = [torch.randn_like(t) for t in targets]
        grads = torch.autograd.grad(targets, inputs, upstream)
        wanted = torch.autograd.grad([t for t in expected if t is not None], reference, upstream)
        for a, b in zip(grads, wanted):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("entry", [flash_attn_func, flash_attn_varlen_func])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fullgraph_dynamic_forward_backward(cpu_runtime, entry, return_lse):
    compiled, graphs = compile_counted(entry)
    for length in (17, 29, 37):
        _, inputs, _, _, _ = make_inputs("cpu", torch.float16, sq=length, sk=length + 6)
        compare_call(compiled, inputs, causal=True, return_lse=return_lse)
    assert len(graphs) == 1
    assert any(
        node.target == torch.ops.flash_attn_npu_4_catlass.forward.default
        for node in graphs[0].graph.nodes
    )


@pytest.mark.parametrize("requires_grad,grad_enabled", [(False, True), (True, False), (True, True)])
def test_fullgraph_return_contract(cpu_runtime, requires_grad, grad_enabled):
    fn = torch.compile(flash_attn_func, backend="aot_eager", fullgraph=True)
    inputs = [
        torch.randn(2, 7, 4, 128, dtype=torch.float16, requires_grad=requires_grad)
        for _ in range(3)
    ]
    with torch.set_grad_enabled(grad_enabled):
        compare_call(fn, inputs)


@pytest.mark.parametrize("loss", ["out", "lse", "both"])
def test_fullgraph_score_aux_and_lse_gradients(cpu_runtime, loss):
    def fn(q, k, v, bias, multiplier):
        out, lse = flash_attn_func(
            q,
            k,
            v,
            score_mod=SCORE,
            score_mod_bwd=SCORE_BWD,
            aux_tensors=[bias],
            aux_scalars=(multiplier,),
            return_lse=True,
        )
        return (out,) if loss == "out" else (lse,) if loss == "lse" else (out, lse)

    compiled, graphs = compile_counted(fn)
    for value in (0.2, 0.7):
        _, inputs, _, _, _ = make_inputs("cpu", torch.float16)
        reference = [t.detach().requires_grad_() for t in inputs]
        bias = torch.linspace(-value, value, 23)
        actual = compiled(*inputs, bias, 0.5)
        expected = fn(*reference, bias, 0.5)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        upstream = [torch.randn_like(t) for t in actual]
        got = torch.autograd.grad(actual, inputs, upstream)
        wanted = torch.autograd.grad(expected, reference, upstream)
        for a, b in zip(got, wanted):
            torch.testing.assert_close(a, b)
        if loss == "lse":
            assert not got[2].count_nonzero()
    assert len(graphs) == 1


@pytest.mark.parametrize("qp,kp", [(True, True), (True, False), (False, True)])
def test_fullgraph_varlen_dynamic(cpu_runtime, qp, kp):
    compiled, graphs = compile_counted(flash_attn_varlen_func)
    for length in (17, 29):
        _, inputs, _, _, options = make_inputs(
            "cpu", torch.float16, qp=qp, kp=kp, sq=length, sk=length + 6
        )
        compare_call(compiled, inputs, **options, return_lse=True)
    assert len(graphs) == 1


def test_fullgraph_noncontiguous_and_unaligned(cpu_runtime):
    compiled = torch.compile(flash_attn_func, backend="aot_eager", fullgraph=True)
    for unaligned in (False, True):
        if unaligned:
            inputs = [
                torch.randn(2 * 7 * 4 * 128 + 1, dtype=torch.float16)[1:]
                .view(2, 7, 4, 128)
                .requires_grad_()
                for _ in range(3)
            ]
        else:
            inputs = [
                torch.randn(2, 4, 7, 128, dtype=torch.float16).transpose(1, 2).requires_grad_()
                for _ in range(3)
            ]
        compare_call(compiled, inputs, return_lse=True)


def test_graph_sparse_build_transpose_and_attention(cpu_runtime):
    mask = builtin.causal()

    def fn(q, k, v):
        batch, sq, heads, _ = q.shape
        sk = k.shape[1]
        blocks = compute_block_sparsity(128, 128, batch, heads, sq, sk, mask, None, q.device)
        backward = transpose_block_sparsity(
            blocks, batch_size=batch, num_heads=heads, seqlen_q=sq, seqlen_k=sk
        )
        return flash_attn_func(
            q,
            k,
            v,
            mask_mod=mask,
            block_sparse_tensors=blocks,
            block_sparse_tensors_bwd=backward,
            return_lse=True,
        )

    compiled, graphs = compile_counted(fn)
    for length in (131, 259):
        _, inputs, _, _, _ = make_inputs("cpu", torch.float16, sq=length, sk=length + 6)
        out, lse = compiled(*inputs)
        expected = flash_attn_func(*inputs, mask_mod=mask, return_lse=True)
        for a, b in zip((out, lse), expected):
            torch.testing.assert_close(a, b)
        upstream = (torch.randn_like(out), torch.randn_like(lse))
        actual_grad = torch.autograd.grad((out, lse), inputs, upstream)
        expected_grad = torch.autograd.grad(expected, inputs, upstream)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)
    assert len(graphs) == 1


@pytest.mark.parametrize("full", [False, True])
def test_graph_packed_sparse_dynamic_sizes(cpu_runtime, full):
    mask = builtin.causal()

    def fn(q, k, v, cq, ck):
        batch, heads = cq.shape[0] - 1, q.shape[1]
        blocks = compute_block_sparsity(
            128,
            128,
            batch,
            heads,
            q.shape[0],
            k.shape[0],
            mask,
            None,
            q.device,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            compute_full_blocks=full,
        )
        backward = transpose_block_sparsity(
            blocks,
            batch_size=batch,
            num_heads=heads,
            seqlen_q=q.shape[0],
            seqlen_k=k.shape[0],
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
        )
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            mask_mod=mask,
            block_sparse_tensors=blocks,
            return_lse=True,
        )
        return out, lse, blocks.mask_block_cnt, backward.mask_block_cnt

    compiled, graphs = compile_counted(fn)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        for lengths in ((129, 5), (2, 132), (0, 134)):
            _, inputs, cq, ck, options = make_inputs(
                "cpu", torch.float16, qp=True, kp=True, q_lengths=lengths, k_lengths=lengths
            )
            out, lse, counts, backward_counts = compiled(*inputs, cq, ck)
            expected = flash_attn_varlen_func(*inputs, **options, mask_mod=mask, return_lse=True)
            for a, b in zip((out, lse), expected):
                torch.testing.assert_close(a, b)
            assert counts.shape[1] == sum((n + 127) // 128 for n in lengths)
            assert backward_counts.shape == counts.shape
            upstream = (torch.randn_like(out), torch.randn_like(lse))
            actual_grad = torch.autograd.grad((out, lse), inputs, upstream)
            expected_grad = torch.autograd.grad(expected, inputs, upstream)
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b)
    assert len(graphs) == 1


def test_graph_input_mutation_and_missing_vjp(cpu_runtime):
    compiled = torch.compile(flash_attn_func, backend="aot_eager", fullgraph=True)
    _, inputs, _, _, _ = make_inputs("cpu", torch.float16)
    bias = torch.zeros(23)
    options = {
        "score_mod": SCORE,
        "score_mod_bwd": SCORE_BWD,
        "aux_tensors": [bias],
        "aux_scalars": (0.5,),
    }
    out, _ = compiled(*inputs, **options)
    bias.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()
    options.pop("score_mod_bwd")
    with pytest.raises(ValueError, match="score_mod_bwd is required"):
        compiled(*inputs, **options)


def test_native_callbacks_rebinding_and_scalar_types(cpu_runtime):
    marked_score, marked_bwd = simt(_score), simt(_score_bwd)
    fn = torch.compile(flash_attn_func, backend="aot_eager", fullgraph=True, dynamic=True)
    bias = torch.zeros(23)
    for score, grad, multiplier in [
        (_score, _score_bwd, 0.5),
        (marked_score, marked_bwd, 2),
        (SCORE, SCORE_BWD, True),
        (None, None, 1.0),
    ]:
        _, inputs, _, _, _ = make_inputs("cpu", torch.float16)
        bias.copy_(torch.linspace(-0.2, 0.4, 23) * multiplier)
        compare_call(
            fn,
            inputs,
            score_mod=score,
            score_mod_bwd=grad,
            aux_tensors=[bias],
            aux_scalars=(multiplier,),
            return_lse=True,
        )


def test_dynamic_batch_and_head_counts(cpu_runtime):
    fn, graphs = compile_counted(flash_attn_func)
    for batch, heads, kv_heads in [(3, 8, 4), (5, 6, 3)]:
        q = torch.randn(batch, 17, heads, 128, dtype=torch.float16, requires_grad=True)
        k, v = [
            torch.randn(batch, 23, kv_heads, 128, dtype=torch.float16, requires_grad=True)
            for _ in range(2)
        ]
        compare_call(fn, (q, k, v), return_lse=True)
    assert len(graphs) == 1


@pytest.mark.parametrize("batch,sq,sk", [(0, 7, 9), (2, 0, 9), (2, 7, 0)])
def test_graph_empty_inputs(cpu_runtime, batch, sq, sk):
    fn = torch.compile(flash_attn_func, backend="aot_eager", fullgraph=True)
    shapes = [(batch, sq, 4, 128), (batch, sk, 2, 128), (batch, sk, 2, 128)]
    inputs = [torch.randn(s, dtype=torch.float16, requires_grad=True) for s in shapes]
    compare_call(fn, inputs, return_lse=True)


def test_graph_sparse_runtime_rebinding(cpu_runtime):
    fn, graphs = compile_counted(flash_attn_func)
    blocks = reference_block_sparsity(2, 4, 17, 23)
    for visible in (True, False, True):
        blocks.full_block_cnt.fill_(int(visible))
        _, inputs, _, _, _ = make_inputs("cpu", torch.float16)
        compare_call(fn, inputs, block_sparse_tensors=blocks, return_lse=True)
    assert len(graphs) == 1
    _, inputs, _, _, _ = make_inputs("cpu", torch.float16)
    out, _ = fn(*inputs, block_sparse_tensors=blocks)
    blocks.full_block_idx.zero_()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


def test_graph_varlen_length_rebinding(cpu_runtime):
    fn, graphs = compile_counted(flash_attn_varlen_func)
    for lengths in ((7, 17), (9, 15), (0, 24)):
        _, inputs, _, _, options = make_inputs(
            "cpu", torch.float16, qp=True, kp=True, q_lengths=lengths, k_lengths=(13, 17)
        )
        compare_call(
            fn,
            inputs,
            **options,
            causal=True,
            return_lse=True,
            max_seqlen_q=torch.tensor(100),
            max_seqlen_k=torch.tensor(100),
        )
    assert len(graphs) == 1


def test_operator_schema_fake_and_autograd(cpu_runtime):
    _, inputs, _, _, _ = make_inputs("cpu", torch.float16)
    callbacks = graph._register_callbacks(*(graph._callback_spec(None) for _ in range(6)))
    sparse, spec = graph._pack_sparse(None, None)
    args = (
        *inputs,
        None,
        None,
        sparse,
        [],
        spec,
        callbacks,
        None,
        False,
        None,
        None,
        0.0,
        [],
        None,
        None,
        "BSND",
        True,
        True,
    )
    torch.library.opcheck(graph._attention_fwd, args)


def test_sparse_operator_schema_and_empty_packed(cpu_runtime):
    blocks = reference_block_sparsity(1, 1, 129, 137)
    tensors, spec = graph._pack_sparse(blocks)
    torch.library.opcheck(graph._transpose_sparse, (tensors, spec, 2, 4, 129, 137, None, None))

    callback = graph._register_callbacks(graph._callback_spec(None))
    cu = torch.zeros(3, dtype=torch.int32)
    offsets = torch.zeros_like(cu)
    args = (
        2,
        4,
        129,
        137,
        callback,
        [],
        [],
        torch.device("cpu"),
        cu,
        cu,
        offsets,
        offsets,
        True,
        False,
    )
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        torch.library.opcheck(graph._build_sparse, args)
    result = graph._build_sparse(*args)
    assert result[0].shape == result[1].shape == (4, 0)
    assert result[4].data_ptr() != offsets.data_ptr()
    assert result[5].data_ptr() != result[4].data_ptr()


@pytest.mark.parametrize("packed", [False, True])
def test_cpu_inductor_fullgraph(cpu_runtime, packed):
    mask = builtin.causal()

    def fn(q, k, v, cq, ck):
        batch = q.shape[0] if cq is None else cq.shape[0] - 1
        sq, sk = (q.shape[1], k.shape[1]) if cq is None else (q.shape[0], k.shape[0])
        heads = q.shape[2] if cq is None else q.shape[1]
        blocks = compute_block_sparsity(
            128, 128, batch, heads, sq, sk, mask, None, q.device, cu_seqlens_q=cq, cu_seqlens_k=ck
        )
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            mask_mod=mask,
            block_sparse_tensors=blocks,
            return_lse=True,
        )

    compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        for length in (17, 29):
            _, inputs, cq, ck, options = make_inputs(
                "cpu", torch.float16, qp=packed, kp=packed, sq=length, sk=length + 6
            )
            actual = compiled(*inputs, cq, ck)
            expected = flash_attn_varlen_func(*inputs, **options, mask_mod=mask, return_lse=True)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            tangents = tuple(torch.randn_like(t) for t in actual)
            for a, b in zip(
                torch.autograd.grad(actual, inputs, tangents),
                torch.autograd.grad(expected, inputs, tangents),
            ):
                torch.testing.assert_close(a, b)


@pytest.mark.npu
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_fullgraph_precision(device, dtype, packed, backend):
    def fn(q, k, v, cq, ck, bias):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            score_mod=SCORE,
            score_mod_bwd=SCORE_BWD,
            aux_tensors=[bias],
            aux_scalars=(0.5,),
            return_lse=True,
        )

    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
    for length in (17, 131):
        cpu, inputs, cq, ck, options = make_inputs(
            device, dtype, qp=packed, kp=packed, sq=length, sk=length + 6
        )
        tokens = inputs[1].shape[0] if packed else inputs[1].shape[1]
        bias = torch.linspace(-0.1, 0.1, tokens)
        out, lse = compiled(
            *inputs, options["cu_seqlens_q"], options["cu_seqlens_k"], bias.to(device)
        )
        check_attention(
            cpu,
            inputs,
            out,
            lse,
            cq=cq,
            ck=ck,
            loss="both",
            score_mod=_score_ref,
            aux_tensors=(bias,),
            aux_scalars=(0.5,),
        )


@pytest.mark.npu
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("packed", [False, True])
def test_npu_graph_sparse_precision(device, packed, backend):
    mask = builtin.causal()

    def fn(q, k, v, cq, ck):
        batch = q.shape[0] if cq is None else cq.shape[0] - 1
        sq, sk = (q.shape[1], k.shape[1]) if cq is None else (q.shape[0], k.shape[0])
        heads = q.shape[2] if cq is None else q.shape[1]
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
        if cq is None:
            out, lse = flash_attn_func(
                q,
                k,
                v,
                mask_mod=mask,
                block_sparse_tensors=blocks,
                block_sparse_tensors_bwd=transposed,
                return_lse=True,
            )
        else:
            # The varlen API accepts only forward lists and prepares backward
            # metadata internally, matching its eager contract.
            out, lse = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cq,
                cu_seqlens_k=ck,
                mask_mod=mask,
                block_sparse_tensors=blocks,
                return_lse=True,
            )
        return out, lse, transposed.mask_block_cnt

    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        for length in (131, 259):
            cpu, inputs, cq, ck, options = make_inputs(
                device, torch.float16, qp=packed, kp=packed, sq=length, sk=length + 6
            )
            out, lse, _ = compiled(*inputs, options["cu_seqlens_q"], options["cu_seqlens_k"])
            check_attention(cpu, inputs, out, lse, cq=cq, ck=ck, loss="both", mask_mod=mask)
