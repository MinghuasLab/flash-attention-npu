# Copyright (c) 2026, Minghua Shen.
import os

import torch

from tests.common.golden_cache import get_or_compute_golden, register_retry
"""
  Shared FlashAttention NPU reference / golden implementation.

  - Forward golden:
    ref_flash_attention computes golden out/lse values for
    the v2/v3/v4 forward tests.
"""

"""
Small-op FlashAttention forward golden implementation.
"""


def softmax1(
    qk_result,
    is_first,
    gm,
    interm_dtype = torch.float16,
    rescale_threshold = 0.0,
    ):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    if is_first:
        hm = lm
        dm = torch.zeros_like(lm)
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
        if rescale_threshold > 0:
            hm = torch.maximum(gm, lm - rescale_threshold)
            dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.exp(sim_sub.to(interm_dtype))
    row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm


def qk_scores(query, key, scale, reorder_ops):
    """Compute BSND QK scores without flattening the batch dimension."""
    q = query.permute(0, 2, 1, 3)  # B, H, Q, D
    k = key.permute(0, 2, 3, 1)  # B, Hkv, D, K
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_heads == num_kv_heads:
        return torch.matmul(q * scale, k) if not reorder_ops else torch.matmul(q, k * scale)
    group_num = num_heads // num_kv_heads
    q = q.reshape(q.shape[0], num_kv_heads, group_num, q.shape[2], q.shape[3])
    k = k.unsqueeze(2)
    scores = torch.matmul(q * scale, k) if not reorder_ops else torch.matmul(q, k * scale)
    return scores.reshape(scores.shape[0], num_heads, scores.shape[-2], scores.shape[-1])


def pv_out(prob, value):
    """Compute BSND probability-times-value without materializing repeated KV."""
    batch, num_heads, q_len, _ = prob.shape
    num_kv_heads = value.shape[2]
    value = value.permute(0, 2, 1, 3)  # B, Hkv, K, D
    if num_heads == num_kv_heads:
        return torch.matmul(prob, value)
    group_num = num_heads // num_kv_heads
    prob = prob.reshape(batch, num_kv_heads, group_num, q_len, prob.shape[-1])
    out = torch.matmul(prob, value.unsqueeze(2))
    return out.reshape(batch, num_heads, q_len, value.shape[-1])


def apply_attention_dropout(prob, drop_mask, dropout_p):
    """Apply an FA-provided dropout mask without changing the softmax LSE."""
    if drop_mask is None or dropout_p <= 0.0:
        return prob
    if dropout_p >= 1.0:
        raise ValueError(f"dropout_p must be less than 1, got {dropout_p}")
    drop_mask = torch.as_tensor(drop_mask, device=prob.device, dtype=prob.dtype)
    if drop_mask.dim() == 3:
        drop_mask = drop_mask.unsqueeze(0)
    if drop_mask.dim() != 4:
        raise ValueError(
            f"drop_mask must be (B,H,Q,K) or (H,Q,K), got {tuple(drop_mask.shape)}"
        )
    return prob * drop_mask / (1.0 - dropout_p)


def alibi_bias(alibi_slopes, q_len, k_len, q_seqlens=None, kv_seqlens=None):
    """Build the additive ALiBi bias [B, H, Q, K] (float32, non-positive).

    alibi_slopes is fp32 shaped (H,) or (B, H). Positions use bottom-right
    alignment: query i and key j get -slope * |i + max(0, k - q) - j|. With
    per-batch q_seqlens/kv_seqlens (padded varlen goldens) the alignment offset
    becomes per-batch; padding positions keep bias 0 so the boolean mask stays
    authoritative.
    """
    slopes = torch.as_tensor(alibi_slopes, dtype=torch.float32)
    if slopes.dim() == 1:
        slopes = slopes.unsqueeze(0)
    batch, num_heads = slopes.shape
    row = torch.arange(q_len, dtype=torch.float32)
    col = torch.arange(k_len, dtype=torch.float32)
    if q_seqlens is None or kv_seqlens is None:
        row_offset = float(max(0, k_len - q_len))
        dist = (row_offset + row[:, None] - col[None, :]).abs()
        return -slopes.reshape(batch, num_heads, 1, 1) * dist.reshape(1, 1, q_len, k_len)
    q_seqlens = torch.as_tensor(q_seqlens)
    kv_seqlens = torch.as_tensor(kv_seqlens)
    bias = torch.zeros(batch, num_heads, q_len, k_len)
    for b in range(batch):
        valid_row = row < float(q_seqlens[b])
        valid_col = col < float(kv_seqlens[b])
        row_offset = float(max(0, int(kv_seqlens[b]) - int(q_seqlens[b])))
        dist = (row_offset + row[:, None] - col[None, :]).abs()
        dist = dist * valid_row[:, None] * valid_col[None, :]
        bias[b] = -slopes[b].reshape(num_heads, 1, 1) * dist.reshape(1, q_len, k_len)
    return bias


def softmax_with_sink(scores, sink_matrix, value_dtype):
    """Compute attention probabilities and LSE with an extra sink term."""
    sink_matrix = torch.as_tensor(sink_matrix, device=scores.device)
    if sink_matrix.dim() == 3:
        sink_matrix = sink_matrix.unsqueeze(0)
    expected_shape = (scores.shape[0], scores.shape[1], scores.shape[2], 1)
    if tuple(sink_matrix.shape) != expected_shape:
        raise ValueError(
            f"sink_matrix shape {tuple(sink_matrix.shape)} must be {expected_shape}"
        )
    sink_matrix = sink_matrix.to(scores.dtype)
    row_max = torch.maximum(scores.amax(dim=-1, keepdim=True), sink_matrix)
    row_max_high = row_max.to(torch.float64)
    score_exp = torch.exp(scores.to(torch.float64) - row_max_high)
    sink_exp = torch.exp(sink_matrix.to(torch.float64) - row_max_high)
    denominator = score_exp.sum(dim=-1, keepdim=True) + sink_exp
    probability = (score_exp / denominator).to(value_dtype)
    lse = (torch.log(denominator) + row_max_high).squeeze(-1)
    return probability, lse


def ref_flash_attention(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap=0.0,
    rescale_threshold=0.0,
    upcast=True,
    reorder_ops=False,
    sink_matrix=None,
    drop_mask=None,
    dropout_p=0.0,
    alibi_slopes=None,
    q_seqlens=None,
    kv_seqlens=None,
):
    """BSND reference implementation that computes all batches together."""
    dtype_og = query.dtype
    if upcast:
        query, key, value = query.float(), key.float(), value.float()
    interm_dtype = value.dtype
    scale = torch.tensor(scale, device=query.device, dtype=query.dtype)
    qk_result = qk_scores(query, key, scale, reorder_ops).to(interm_dtype)

    if softcap > 0.0:
        qk_result = torch.tanh(qk_result / softcap) * softcap
    if alibi_slopes is not None:
        bias = alibi_bias(
            alibi_slopes, qk_result.shape[-2], qk_result.shape[-1], q_seqlens, kv_seqlens
        ).to(device=qk_result.device)
        qk_result = qk_result + bias.to(qk_result.dtype)
    if mask is not None:
        mask = mask.to(device=qk_result.device, dtype=torch.bool)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        qk_result = qk_result.masked_fill(mask[:, None, :, :], -1e4)

    if sink_matrix is not None and rescale_threshold and rescale_threshold > 0.0:
        raise NotImplementedError("sink_matrix is not supported with rescale_threshold")

    if rescale_threshold and rescale_threshold > 0.0:
        context_size = 512
        gm = None
        gl = None
        go = None
        for kv_start in range(0, qk_result.shape[-1], context_size):
            qk_chunk = qk_result[..., kv_start:kv_start + context_size]
            p_chunk, row_sum, dm, gm = softmax1(
                qk_chunk,
                kv_start == 0,
                gm,
                interm_dtype,
                rescale_threshold,
            )
            p_chunk = apply_attention_dropout(
                p_chunk,
                None if drop_mask is None else drop_mask[..., kv_start:kv_start + context_size],
                dropout_p,
            )
            # Match the NPU kernel's rescale-O path by accumulating go/lo/gl in fp32.
            # With long KV and windowing, the unnormalized exp-weighted sum of
            # valid keys can exceed the fp16 limit (65504). fp16 accumulation
            # would overflow to NaN, whereas fp32 accumulation remains safe.
            lo = pv_out(
                p_chunk.to(torch.float32),
                value[:, kv_start:kv_start + context_size].to(torch.float32),
            )
            if kv_start == 0:
                gl = row_sum.to(torch.float32)
                go = lo
            else:
                dm = torch.exp(dm)
                gl = gl * dm + row_sum.to(torch.float32)
                go = go * dm + lo
        out = (go / gl).permute(0, 2, 1, 3)
        lse = torch.squeeze(torch.log(gl) + gm, dim=-1).to(torch.float32)
    else:
        if sink_matrix is None:
            lse = torch.logsumexp(qk_result.to(torch.float32), dim=-1)
            prob = torch.softmax(qk_result, dim=-1).to(value.dtype)
        else:
            prob, lse = softmax_with_sink(qk_result, sink_matrix, value.dtype)
        prob = apply_attention_dropout(prob, drop_mask, dropout_p)
        out = pv_out(prob, value).permute(0, 2, 1, 3)
    return out.to(dtype_og), lse


def _ref_flash_attention_pair(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap=0.0,
    rescale_threshold=None,
    sink_matrix=None,
    drop_mask=None,
    dropout_p=0.0,
    alibi_slopes=None,
    q_seqlens=None,
    kv_seqlens=None,
):
    """Return the two BSND golden references used by the comparator."""
    kwargs = {} if rescale_threshold is None else {"rescale_threshold": rescale_threshold}
    out_ref, lse_ref = ref_flash_attention(
        query, key, value, scale, mask, data_type, softcap,
        upcast=True, reorder_ops=False, sink_matrix=sink_matrix,
        drop_mask=drop_mask, dropout_p=dropout_p,
        alibi_slopes=alibi_slopes, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens,
        **kwargs,
    )
    out_pt, lse_pt = ref_flash_attention(
        query, key, value, scale, mask, data_type, softcap,
        upcast=False, reorder_ops=True, sink_matrix=sink_matrix,
        drop_mask=drop_mask, dropout_p=dropout_p,
        alibi_slopes=alibi_slopes, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens,
        **kwargs,
    )
    return out_ref, lse_ref, out_pt, lse_pt


def _golden_metadata(query, key, value, scale, mask, data_type, softcap, extra=None):
    metadata = {
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
        "query_dtype": str(query.dtype),
        "key_dtype": str(key.dtype),
        "value_dtype": str(value.dtype),
        "data_type": str(data_type),
        "scale": scale,
        "softcap": softcap,
    }
    if extra:
        metadata.update(extra)
    return metadata


def cached_ref_flash_attention_pair(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap=0.0,
    *,
    nodeid=None,
    rescale_threshold=None,
    sink_matrix=None,
    drop_mask=None,
    dropout_p=0.0,
    alibi_slopes=None,
    q_seqlens=None,
    kv_seqlens=None,
    metadata=None,
):
    """Cached wrapper for the two forward reference implementations."""
    case_metadata = _golden_metadata(
        query, key, value, scale, mask, data_type, softcap, metadata
    )
    case_metadata["rescale_threshold"] = rescale_threshold
    case_metadata["dropout_p"] = dropout_p
    case_metadata["alibi_slopes_shape"] = (
        None if alibi_slopes is None else list(alibi_slopes.shape)
    )
    case_metadata["q_seqlens"] = None if q_seqlens is None else list(q_seqlens)
    case_metadata["kv_seqlens"] = None if kv_seqlens is None else list(kv_seqlens)
    inputs = {
        "query": query,
        "key": key,
        "value": value,
        "mask": mask,
        "sink": sink_matrix,
        "drop_mask": drop_mask,
        "alibi_slopes": alibi_slopes,
        "q_seqlens": q_seqlens,
        "kv_seqlens": kv_seqlens,
    }

    def compute():
        values = _ref_flash_attention_pair(
            query, key, value, scale, mask, data_type, softcap,
            rescale_threshold=rescale_threshold, sink_matrix=sink_matrix,
            drop_mask=drop_mask, dropout_p=dropout_p,
            alibi_slopes=alibi_slopes, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens,
        )
        return dict(zip(("out_ref", "lse_ref", "out_pt", "lse_pt"), values))

    cache_args = dict(
        nodeid=nodeid or _current_nodeid(), metadata=case_metadata, inputs=inputs,
        compute_fn=compute, expected_keys=("out_ref", "lse_ref", "out_pt", "lse_pt"),
        source_files=[__file__], test_source_files=_golden_test_source_files(),
    )
    values, status = get_or_compute_golden(**cache_args, return_status=True)
    if status == "hit":
        register_retry(values, lambda: get_or_compute_golden(**cache_args, force_refresh=True))
    return tuple(values[name] for name in ("out_ref", "lse_ref", "out_pt", "lse_pt"))


def cached_autograd_grads(nodeid, outputs, refs, dout, *, metadata=None, inputs=None):
    """Cache an existing pair of reference autograd results.

    ``outputs`` and ``refs`` are the exact tensors already built by the test,
    so this helper also works for packed/varlen views whose gradient shape is
    different from the padded reference input.
    """
    query, key, value = refs
    case_metadata = {
        "gradient": True,
        "metadata": _json_metadata(metadata),
        "output_shapes": [list(outputs[0].shape), list(outputs[1].shape)],
    }
    caller_inputs = inputs if inputs is not None else {
        "query": query,
        "key": key,
        "value": value,
        "dout": dout,
    }
    cache_inputs = {
        "caller_inputs": caller_inputs,
        "out_ref": outputs[0],
        "out_pt": outputs[1],
    }

    def compute():
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(
            outputs[0], refs, dout.detach().cpu(), retain_graph=True
        )
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(
            outputs[1], refs, dout.detach().cpu()
        )
        return {
            "dq_ref": dq_ref, "dk_ref": dk_ref, "dv_ref": dv_ref,
            "dq_pt": dq_pt, "dk_pt": dk_pt, "dv_pt": dv_pt,
        }

    cache_args = dict(
        nodeid=nodeid, metadata=case_metadata, inputs=cache_inputs,
        compute_fn=compute,
        expected_keys=("dq_ref", "dk_ref", "dv_ref", "dq_pt", "dk_pt", "dv_pt"),
        source_files=[__file__], test_source_files=_golden_test_source_files(),
    )
    values, status = get_or_compute_golden(**cache_args, return_status=True)
    if status == "hit":
        register_retry(values, lambda: get_or_compute_golden(**cache_args, force_refresh=True))
    return tuple(values[name] for name in ("dq_ref", "dk_ref", "dv_ref", "dq_pt", "dk_pt", "dv_pt"))


def _json_metadata(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _golden_test_source_files():
    test_file = os.environ.get("GOLDEN_CACHE_TEST_FILE")
    return [test_file] if test_file else []


def _current_nodeid():
    return os.environ.get("GOLDEN_CACHE_NODEID", "standalone")


def ref_flash_attention_pair(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap=0.0,
    rescale_threshold=None,
    sink_matrix=None,
    drop_mask=None,
    dropout_p=0.0,
    alibi_slopes=None,
    q_seqlens=None,
    kv_seqlens=None,
):
    # Backward tests need the live CPU graph to remain available when their
    # separate gradient artifact is missing or being refreshed.  Forward-only
    # cases use the persistent cache below.
    if any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad
        for tensor in (query, key, value)
    ):
        return _ref_flash_attention_pair(
            query, key, value, scale, mask, data_type, softcap,
            rescale_threshold=rescale_threshold, sink_matrix=sink_matrix,
            drop_mask=drop_mask, dropout_p=dropout_p,
            alibi_slopes=alibi_slopes, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens,
        )
    return cached_ref_flash_attention_pair(
        query, key, value, scale, mask, data_type, softcap,
        nodeid=_current_nodeid(),
        rescale_threshold=rescale_threshold, sink_matrix=sink_matrix,
        drop_mask=drop_mask, dropout_p=dropout_p,
        alibi_slopes=alibi_slopes, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens,
    )
