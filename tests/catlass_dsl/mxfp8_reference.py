"""Independent CPU MXFP8 inputs and online-attention oracle (tests only).

No production quantizer, layout adapter, or kernel helper is imported here.
The probability oracle uses float64 exponent selection and PyTorch's CPU RNE
FP8 conversion, independently of the device bit-manipulation implementation.
"""

import torch


from flash_attn_npu_4.catlass_dsl import TensorwiseScales

SCALE_NAMES = ("q_descale", "k_descale", "v_descale")


def quantize_group32(values, axis=-1):
    """Return E4M3FN values and one E8M0 byte per last-axis group32.

    Values retain their original shape; scales have the moved-axis shape with
    the last extent replaced by ceil(length / 32). Input must be finite.
    """
    values = values.detach().cpu().float().movedim(axis, -1)
    if not torch.isfinite(values).all():
        raise ValueError("the test input quantizer requires finite values")
    length = values.shape[-1]
    groups = (length + 31) // 32
    padded = torch.zeros((*values.shape[:-1], groups * 32), dtype=torch.float32)
    padded[..., :length] = values
    grouped = padded.reshape(*values.shape[:-1], groups, 32)
    amax = grouped.abs().amax(-1)
    # float64 avoids underflow in amax/448 and erroneous log2 threshold rounding.
    exponent = torch.ceil(torch.log2(amax.double() / 448.0)).clamp(-127, 127)
    exponent = torch.where(amax == 0, 0, exponent).to(torch.int32)
    multiplier = torch.ldexp(torch.ones_like(amax, dtype=torch.float64), exponent)
    rounded = (grouped.double() / multiplier[..., None]).clamp(-448, 448).float()
    encoded = rounded.to(torch.float8_e4m3fn).reshape(*values.shape[:-1], groups * 32)
    return encoded[..., :length].movedim(-1, axis).contiguous(), (exponent + 127).byte()


def decode_scale(scale):
    raw = scale.detach().cpu().view(torch.uint8)
    if (raw == 255).any():
        raise ValueError("E8M0 255 is invalid input")
    return torch.ldexp(torch.ones_like(raw, dtype=torch.float32), raw.int() - 127)


def quantize_qk(values):
    q8, scale = quantize_group32(values)
    return q8, scale.reshape(*values.shape[:-1], 2, 2)


def quantize_v(sequence):
    """Quantize one [S,H,128] sequence; V scale groups restart at S=0."""
    length, heads, dim = sequence.shape
    extent = (length + 63) // 64
    padded = torch.zeros((extent * 64, heads, dim), dtype=torch.float32)
    padded[:length] = sequence
    encoded, scales = quantize_group32(padded, axis=0)
    scales = scales.reshape(heads, dim, extent, 2).permute(2, 0, 1, 3).contiguous()
    return encoded[:length].contiguous(), scales


def make_case(
    *,
    q_lengths=(17, 17),
    k_lengths=(23, 23),
    q_heads=4,
    kv_heads=2,
    q_packed=False,
    k_packed=False,
    layout="BSND",
    seed=42,
    v_scale_capacity=None,
    native_scales=False,
):
    """Build CPU inputs; fixed sides require equal lengths across the batch."""
    if len(q_lengths) != len(k_lengths):
        raise ValueError("Q/K lengths describe different batches")
    if not q_packed and len(set(q_lengths)) > 1:
        raise ValueError("fixed Q requires uniform lengths")
    if not k_packed and len(set(k_lengths)) > 1:
        raise ValueError("fixed KV requires uniform lengths")
    rng = torch.Generator().manual_seed(seed)

    def random_values(length, heads):
        data = torch.randn((length, heads, 128), generator=rng) * 0.2
        # Unequal adjacent group scales expose accidental per-head scaling.
        powers = torch.randint(-2, 3, (length, heads, 4), generator=rng)
        return torch.ldexp(data, powers.repeat_interleave(32, -1))

    q_parts, k_parts, v_parts, qs_parts, ks_parts, vs_parts = [], [], [], [], [], []
    for qn, kn in zip(q_lengths, k_lengths):
        q, qs = quantize_qk(random_values(qn, q_heads))
        k, ks = quantize_qk(random_values(kn, kv_heads))
        v, vs = quantize_v(random_values(kn, kv_heads))
        for parts, value in zip(
            (q_parts, k_parts, v_parts, qs_parts, ks_parts, vs_parts), (q, k, v, qs, ks, vs)
        ):
            parts.append(value)

    def combine(parts, packed, trailing, dtype):
        if parts:
            # CPU cat/stack do not implement every float8 operation.
            raw = [p.view(torch.uint8) if p.dtype == torch.float8_e4m3fn else p for p in parts]
            result = torch.cat(raw) if packed else torch.stack(raw)
            result = result.view(dtype)
        else:
            result = torch.empty(((0,) if packed else (0, 0)) + trailing, dtype=dtype)
        if not packed and layout == "BNSD":
            result = result.transpose(1, 2).contiguous()
        return result

    q = combine(q_parts, q_packed, (q_heads, 128), torch.float8_e4m3fn)
    k = combine(k_parts, k_packed, (kv_heads, 128), torch.float8_e4m3fn)
    v = combine(v_parts, k_packed, (kv_heads, 128), torch.float8_e4m3fn)
    qs = combine(qs_parts, q_packed, (q_heads, 2, 2), torch.uint8)
    ks = combine(ks_parts, k_packed, (kv_heads, 2, 2), torch.uint8)
    vs = combine(vs_parts, k_packed, (kv_heads, 128, 2), torch.uint8)
    if k_packed and v_scale_capacity is not None:
        if v_scale_capacity < vs.shape[0]:
            raise ValueError("requested V scale capacity is too small")
        capacity = torch.full((v_scale_capacity, kv_heads, 128, 2), 255, dtype=torch.uint8)
        capacity[: vs.shape[0]] = vs
        vs = capacity  # Poison unused capacity; it must never be read.
    kwargs = dict(zip(SCALE_NAMES, (qs, ks, vs)))
    if native_scales:
        kwargs = {name: value.view(torch.float8_e8m0fnu) for name, value in kwargs.items()}
    for name, lengths, packed in (
        ("cu_seqlens_q", q_lengths, q_packed),
        ("cu_seqlens_k", k_lengths, k_packed),
    ):
        if packed:
            kwargs[name] = torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
            )
    kwargs.update(max_seqlen_q=max(q_lengths, default=0), max_seqlen_k=max(k_lengths, default=0))
    return dict(
        qkv=(q, k, v),
        kwargs=kwargs,
        q_lengths=tuple(q_lengths),
        k_lengths=tuple(k_lengths),
        layout=layout,
    )


def to_device(case, device):
    options = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in case["kwargs"].items()
        if name not in SCALE_NAMES
    }
    options["scales"] = TensorwiseScales(*(case["kwargs"][name].to(device) for name in SCALE_NAMES))
    return tuple(t.to(device) for t in case["qkv"]), options


def decode_sequences(case):
    q, k, v = (t.detach().cpu() for t in case["qkv"])
    qs, ks, vs = (case["kwargs"][name].detach().cpu() for name in SCALE_NAMES)
    qoff = koff = soff = 0
    decoded = []
    for b, (qn, kn) in enumerate(zip(case["q_lengths"], case["k_lengths"])):
        # Scale and input ranks differ, so use the corresponding Q/K layout.
        qpart = q[qoff : qoff + qn] if q.ndim == 3 else q[b]
        kpart = k[koff : koff + kn] if k.ndim == 3 else k[b]
        vpart = v[koff : koff + kn] if v.ndim == 3 else v[b]
        qsp = qs[qoff : qoff + qn] if q.ndim == 3 else qs[b]
        ksp = ks[koff : koff + kn] if k.ndim == 3 else ks[b]
        vsp = vs[soff : soff + (kn + 63) // 64] if k.ndim == 3 else vs[b]
        if case["layout"] == "BNSD":
            if q.ndim == 4:
                qpart, qsp = qpart.transpose(0, 1), qsp.transpose(0, 1)
            if k.ndim == 4:
                kpart, vpart, ksp, vsp = (t.transpose(0, 1) for t in (kpart, vpart, ksp, vsp))
        qdecoded = qpart.float() * decode_scale(qsp).flatten(-2).repeat_interleave(32, -1)
        kdecoded = kpart.float() * decode_scale(ksp).flatten(-2).repeat_interleave(32, -1)
        vscale = decode_scale(vsp).permute(0, 3, 1, 2).flatten(0, 1).repeat_interleave(32, 0)[:kn]
        decoded.append((qdecoded, kdecoded, vpart.float() * vscale))
        qoff += qn
        koff += kn
        soff += (kn + 63) // 64
    return decoded


def decoded_inputs(case, dtype=torch.bfloat16):
    sequences = decode_sequences(case)
    values = []
    for i, original in enumerate(case["qkv"]):
        parts = [s[i] for s in sequences]
        if not parts:
            values.append(torch.empty_like(original, dtype=dtype))
            continue
        value = torch.cat(parts) if original.ndim == 3 else torch.stack(parts)
        if original.ndim == 4 and case["layout"] == "BNSD":
            value = value.transpose(1, 2)
        values.append(value.to(dtype).contiguous())
    return tuple(values)


def reference(
    case,
    *,
    scale=None,
    score_mod=None,
    mask_mod=None,
    causal=False,
    window_size=(None, None),
    softcap=0.0,
    sparse=None,
    aux=(),
    scalars=(),
    quantize_p=True,
    query_rows=None,
    normalize_quantized=False,
):
    """CPU online golden; sparse lists use the existing partial-then-full order.

    Optional query_rows is a list of local row-index lists, one per sequence.
    Unchecked rows remain NaN, allowing large benchmark cases to sample rows.
    normalize_quantized also accumulates the decoded P8 row sum for output
    normalization. LSE stays an independent, unquantized score reference.
    """
    original_q = case["qkv"][0]
    result = torch.full(original_q.shape, float("nan"), dtype=torch.float32)
    hq = (
        original_q.shape[-2]
        if original_q.ndim == 3 or case["layout"] == "BSND"
        else original_q.shape[1]
    )
    lshape = (
        (hq, original_q.shape[0])
        if original_q.ndim == 3
        else (original_q.shape[0], hq, original_q.shape[1 if case["layout"] == "BSND" else 2])
    )
    lse = torch.full(lshape, float("nan"), dtype=torch.float32)
    scale = 128**-0.5 if scale is None else scale
    qoff = koff = 0
    for b, (q, k, v) in enumerate(decode_sequences(case)):
        qn, kn, kh = q.shape[0], k.shape[0], k.shape[1]
        info = (
            qn,
            kn,
            qoff if original_q.ndim == 3 else 0,
            koff if case["qkv"][1].ndim == 3 else 0,
        )
        selected = list(range(qn)) if query_rows is None else list(query_rows[b])
        for qb in sorted({row // 128 for row in selected}):
            rows = torch.tensor([row for row in selected if row // 128 == qb], dtype=torch.long)
            for h in range(hq):
                kvh = h // (hq // kh)
                m = torch.full((len(rows),), -torch.inf)
                denominator = torch.zeros(len(rows))
                quantized_denominator = torch.zeros(len(rows))
                accum = torch.zeros((len(rows), 128))
                blocks = [
                    (kb, mask_mod is None and not causal and window_size == (None, None))
                    for kb in range((kn + 127) // 128)
                ]
                if sparse is not None:
                    blocks = []
                    for full, counts, indices in (
                        (False, sparse.mask_block_cnt, sparse.mask_block_idx),
                        (True, sparse.full_block_cnt, sparse.full_block_idx),
                    ):
                        if counts is None:
                            continue
                        bi, hi = b % counts.shape[0], h % counts.shape[1]
                        blocks += [
                            (int(kb), full) for kb in indices[bi, hi, qb, : int(counts[bi, hi, qb])]
                        ]
                for kb, full in blocks:
                    start, end = kb * 128, min(kn, (kb + 1) * 128)
                    keys = torch.arange(start, end)
                    scores = (q[rows, h] @ k[start:end, kvh].T) * scale
                    if softcap:
                        scores = softcap * torch.tanh(scores / softcap)
                    if score_mod is not None:
                        scores = score_mod(
                            scores, b, h, rows[:, None], keys[None, :], info, aux, scalars
                        )
                    if not full:
                        center = rows[:, None] + kn - qn
                        if mask_mod is not None:
                            visible = mask_mod(
                                b, h, rows[:, None], keys[None, :], info, aux, scalars
                            )
                        else:
                            left, right = window_size
                            if left is not None and right is not None and left < 0 and right < 0:
                                left = right = None
                            if causal:
                                right = 0
                            visible = torch.ones_like(scores, dtype=torch.bool)
                            if left is not None:
                                visible &= keys[None, :] >= center - left
                            if right is not None:
                                visible &= keys[None, :] <= center + right
                        scores = torch.where(visible, scores, -torch.inf)
                    updated_m = torch.maximum(m, scores.amax(-1))
                    empty = torch.isneginf(updated_m)
                    alpha = torch.exp(torch.where(empty, 0.0, m - updated_m))
                    p = torch.exp(
                        torch.where(empty[:, None], -torch.inf, scores - updated_m[:, None])
                    )
                    denominator = alpha * denominator + p.sum(-1)
                    pv = p
                    if quantize_p:
                        invalid_row = ~torch.isfinite(p).all(-1, keepdim=True)
                        encoded, pscale = quantize_group32(torch.where(torch.isfinite(p), p, 0.0))
                        pv = (
                            encoded.float()
                            * decode_scale(pscale).repeat_interleave(32, -1)[..., : p.shape[-1]]
                        )
                        pv = torch.where(invalid_row, torch.nan, pv)
                    accum = alpha[:, None] * accum + pv @ v[start:end, kvh]
                    quantized_denominator = alpha * quantized_denominator + pv.sum(-1)
                    m = updated_m
                empty = denominator == 0
                output_denominator = quantized_denominator if normalize_quantized else denominator
                out = torch.where(empty[:, None], 0.0, accum / output_denominator[:, None])
                logsum = torch.where(empty, -torch.inf, m + denominator.log())
                if original_q.ndim == 3:
                    result[qoff + rows, h] = out
                    lse[h, qoff + rows] = logsum
                elif case["layout"] == "BNSD":
                    result[b, h, rows] = out
                    lse[b, h, rows] = logsum
                else:
                    result[b, rows, h] = out
                    lse[b, h, rows] = logsum
        qoff += qn
        koff += kn
    return result, lse


def check_quantized_lse(actual, expected):
    """Bound LSE rounding independently of the unchanged BF16 output check.

    E4M3 RNE has at most 1/16 relative error for normal probabilities. Its
    log-sum error is bounded by -log(1-1/16) < 0.065. The fast-path denominator
    guard bounds the additional subnormal contribution; 0.07 leaves room for
    it. The relative term covers FP16 score rounding. NaN and infinity classes
    must match exactly. This is for the approximate MX forward path only.
    """
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=0.07, rtol=2**-11, equal_nan=False
    )


def error_metrics(actual, expected):
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    finite = torch.isfinite(expected)
    error = (actual[finite] - expected[finite]).abs()
    return dict(
        elements=int(error.numel()),
        rms=float(error.square().mean().sqrt()) if error.numel() else 0.0,
        max_abs=float(error.max()) if error.numel() else 0.0,
        p50_abs=float(error.quantile(0.5)) if error.numel() else 0.0,
        p99_abs=float(error.quantile(0.99)) if error.numel() else 0.0,
    )
