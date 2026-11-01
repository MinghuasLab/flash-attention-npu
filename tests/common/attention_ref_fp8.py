# Copyright (c) 2026, Minghua Shen.
"""Pure-PyTorch FP8 forward references for the 950 V3 tests.

Windows use bottom-right alignment; negative sides are unlimited.
The low reference models one unsplit KV stream, independently of host tiling,
metadata layout and FD scheduling. ATK owns exact schedule-aware validation.
"""

import os
from dataclasses import asdict, dataclass

import torch
from tests.common.golden_cache import get_or_compute_golden, register_retry

P_SCALE = 256.0


@dataclass(frozen=True)
class AttentionCase:
    entry: str
    q_lengths: tuple
    kv_lengths: tuple
    q_heads: int
    kv_heads: int
    mask: str
    window_size: tuple
    page_size: int
    num_splits: int
    metadata: bool
    return_lse: bool
    softmax_scale: float
    softcap: float

    @property
    def tnd(self):
        # 950 V3 由入口和 cu_seqlens_q 选择 layout，没有 input_layout 参数。
        # varlen、分页 cache，以及每个 batch 的 Q 长度不同的 cache 使用 TND。
        return (
            self.entry == "varlen"
            or self.page_size != 0
            or (self.entry == "cache" and len(set(self.q_lengths)) > 1)
        )

    @property
    def windows(self):
        return self.window_size if self.mask == "window" else (-1, -1)


def read_inputs(kwargs, entry, metadata=False):
    # 输入已经是 generator 指定的接口布局；只为 CPU golden 还原逻辑 BSND。
    native = {
        name: value.cpu() if isinstance(value, torch.Tensor) else value
        for name, value in kwargs.items()
    }
    q = native["q"].to(torch.float8_e4m3fn)
    k = native["k_cache" if entry == "cache" else "k"].to(torch.float8_e4m3fn)
    v = native["v_cache" if entry == "cache" else "v"].to(torch.float8_e4m3fn)
    native["q"] = q
    native["k_cache" if entry == "cache" else "k"] = k
    native["v_cache" if entry == "cache" else "v"] = v
    cuq = native.get("cu_seqlens_q")
    cuk = native.get("cu_seqlens_k")
    if cuq is None:
        q_lengths = (q.shape[1],) * q.shape[0]
    else:
        q_lengths = tuple(int(n) for n in cuq.diff())
    if entry == "cache":
        kv_lengths = tuple(int(n) for n in native["cache_seqlens"])
    elif cuk is not None:
        kv_lengths = tuple(int(n) for n in cuk.diff())
    else:
        kv_lengths = (k.shape[1],) * len(q_lengths)
    table = native.get("page_table")
    # Keep the API bounds verbatim: visible() applies them to each actual
    # sequence independently, without copying the host normalization logic.
    window_left, window_right = native["window_size"]
    if native["causal"]:
        window_right = 0
    if window_left < 0 and window_right == 0:
        mask = "causal"
    elif window_left >= 0 or window_right >= 0:
        mask = "window"
    else:
        mask = "none"
    windows = (window_left, window_right)
    case = AttentionCase(
        entry=entry,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        q_heads=q.shape[-2],
        kv_heads=k.shape[-2],
        mask=mask,
        window_size=windows,
        page_size=k.shape[1] if table is not None else 0,
        num_splits=native["num_splits"],
        metadata=metadata,
        return_lse=native["return_softmax_lse" if entry == "cache" else "return_attn_probs"],
        softmax_scale=native["softmax_scale"],
        softcap=native["softcap"],
    )
    batch = len(q_lengths)
    logical_q = torch.zeros(batch, max(q_lengths), case.q_heads, 128, dtype=q.dtype)
    logical_k = torch.zeros(batch, max(kv_lengths), case.kv_heads, 128, dtype=k.dtype)
    logical_v = torch.zeros_like(logical_k)
    q_start = k_start = 0
    for b, (qn, kn) in enumerate(zip(q_lengths, kv_lengths)):
        logical_q[b, :qn] = q[q_start : q_start + qn] if q.dim() == 3 else q[b, :qn]
        if table is not None:
            for page in range((kn + case.page_size - 1) // case.page_size):
                start = page * case.page_size
                count = min(case.page_size, kn - start)
                physical = int(table[b, page])
                logical_k[b, start : start + count] = k[physical, :count]
                logical_v[b, start : start + count] = v[physical, :count]
        else:
            logical_k[b, :kn] = k[k_start : k_start + kn] if k.dim() == 3 else k[b, :kn]
            logical_v[b, :kn] = v[k_start : k_start + kn] if v.dim() == 3 else v[b, :kn]
        q_start += qn
        k_start += kn
    inputs = dict(
        q=logical_q,
        k=logical_k,
        v=logical_v,
        native=native,
        descales=[native[name] for name in ("q_descale", "k_descale", "v_descale")],
    )
    return case, inputs


def scale_at(inputs, index, batch, head):
    scale = inputs["descales"][index]
    return 1.0 if scale is None else float(scale[batch, head])


def visible(case, batch, q0, rows, k0, cols):
    q = torch.arange(q0, q0 + rows)[:, None] + case.kv_lengths[batch] - case.q_lengths[batch]
    k = torch.arange(k0, k0 + cols)[None, :]
    if case.mask == "causal":
        return k <= q
    if case.mask == "window":
        left, right = case.window_size
        return ((k >= q - left) if left >= 0 else torch.ones_like(k, dtype=torch.bool)) & (
            (k <= q + right) if right >= 0 else torch.ones_like(k, dtype=torch.bool)
        )
    return torch.ones(rows, cols, dtype=torch.bool)


def high_reference(case, inputs):
    outputs, lses = [], []
    for batch, (qn, kn) in enumerate(zip(case.q_lengths, case.kv_lengths)):
        output = torch.empty(qn, case.q_heads, 128, dtype=torch.float64)
        lse = torch.empty(case.q_heads, qn, dtype=torch.float64)
        for head in range(case.q_heads):
            kv_head = head // (case.q_heads // case.kv_heads)
            q = inputs["q"][batch, :qn, head].double() * scale_at(inputs, 0, batch, kv_head)
            k = inputs["k"][batch, :kn, kv_head].double() * scale_at(inputs, 1, batch, kv_head)
            v = inputs["v"][batch, :kn, kv_head].double() * scale_at(inputs, 2, batch, kv_head)
            for q0 in range(0, qn, 128):
                score = (q[q0 : q0 + 128] @ k.T) * case.softmax_scale
                if case.softcap:
                    score = case.softcap * torch.tanh(score / case.softcap)
                score.masked_fill_(~visible(case, batch, q0, len(score), 0, kn), -torch.inf)
                row_lse = torch.logsumexp(score, -1)
                probability = torch.softmax(score, -1).nan_to_num()
                output[q0 : q0 + len(score), head] = probability @ v
                lse[head, q0 : q0 + len(score)] = row_lse.masked_fill(
                    ~torch.isfinite(row_lse), torch.inf
                )
        outputs.append(output)
        lses.append(lse)
    return torch.cat(outputs), torch.cat(lses, dim=1)


def dn_sum(exponential):
    chains = exponential.reshape(exponential.shape[0], 4, 64)
    sums = torch.zeros(exponential.shape[0], 4)
    for index in range(64):
        sums = sums + chains[:, :, index]
    return (sums[:, 0] + sums[:, 2]) + (sums[:, 1] + sums[:, 3])


def softcap_reference(score, cap):
    exponential = torch.exp(torch.clamp(score * -2.0, max=17.6))
    # Axpy rounds cap - exponential * cap once, matching the Vector instruction.
    numerator = (cap.double() - exponential.double() * cap.double()).float()
    return numerator / (1.0 + exponential)


def partial_reference(case, inputs, batch, head, q0, rows, begin, end):
    kv_head = head // (case.q_heads // case.kv_heads)
    q = inputs["q"][batch, q0 : q0 + rows, head].float()
    k = inputs["k"][batch, : case.kv_lengths[batch], kv_head].float()
    v = inputs["v"][batch, : case.kv_lengths[batch], kv_head].float()
    scale = torch.tensor(case.softmax_scale, dtype=torch.float32)
    if case.softcap:
        cap = torch.tensor(case.softcap, dtype=torch.float32)
        scale = scale / cap
    scale = (scale * scale_at(inputs, 0, batch, kv_head)) * scale_at(inputs, 1, batch, kv_head)
    maximum = torch.full((rows,), torch.finfo(torch.float32).min)
    total, accumulator = torch.zeros(rows), torch.zeros(rows, 128)
    first = True
    for tile in range(begin, end):
        k0, count = tile * 256, min(256, len(k) - tile * 256)
        keep = visible(case, batch, q0, rows, k0, count)
        if not keep.any():
            continue
        raw_score = cube_matmul(q, k[k0 : k0 + count].T)
        score = raw_score * scale
        if case.softcap:
            score = softcap_reference(score, cap)
        score.masked_fill_(~keep, torch.finfo(torch.float32).min)
        if case.softcap or not keep.all():
            local_max = score.max(-1).values
        else:
            local_max = torch.clamp(
                raw_score.max(-1).values * scale, min=torch.finfo(torch.float32).min
            )
        current = torch.maximum(local_max, maximum)
        padded = torch.full((rows, 256), torch.finfo(torch.float32).min)
        padded[:, :count] = score
        correction = torch.exp(maximum - current)
        exponential = torch.exp(
            padded - (current[:, None] - torch.tensor(5.5451774444795623, dtype=torch.float32))
        )
        if not keep.all():
            columns = keep.any(0).nonzero().flatten()
            first_column, last_column = int(columns[0]), int(columns[-1])
            if last_column < 64:
                kv_begin, kv_end = 0, 64
            elif first_column >= 192:
                kv_begin, kv_end = 192, 256
            elif last_column < 128:
                kv_begin, kv_end = 0, 128
            elif first_column >= 128:
                kv_begin, kv_end = 128, 256
            elif last_column < 192:
                kv_begin, kv_end = 0, 192
            elif first_column >= 64:
                kv_begin, kv_end = 64, 256
            else:
                kv_begin, kv_end = 0, 256
            exponential[:, :kv_begin] = 0
            exponential[:, kv_end:] = 0
        row_sum = dn_sum(exponential)
        probability = exponential.to(torch.float8_e4m3fn).float()
        value = torch.zeros(256, 128)
        value[:count] = v[k0 : k0 + count]
        k_round = (count + 31) // 32 * 32
        temporary = cube_matmul(probability[:, :k_round], value[:k_round])
        accumulator = temporary if first else accumulator * correction[:, None] + temporary
        total = row_sum if first else total * correction + row_sum
        maximum, first = current, False
    output = (accumulator / total[:, None]) * scale_at(inputs, 2, batch, kv_head)
    output = torch.where((total > 0)[:, None], output, 0)
    lse = torch.log(total * torch.tensor(1 / P_SCALE, dtype=torch.float32)) + maximum
    return output, torch.where(total > 0, lse, torch.inf)


def cube_matmul(a, b):
    """FP8 dot products: align each group of 16 products before FP32 addition.

    The same Cube rounding model is used in the ATK low reference. Here all
    operations are CPU PyTorch; no compiler or production source is needed.
    """
    output = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)
    for start in range(0, a.shape[1], 16):
        products = a[:, None, start : start + 16] * b[start : start + 16].T[None, :, :]
        bits = products.contiguous().view(torch.int32)
        exponents = (bits & 0x7F800000) >> 23
        shifts = exponents.max(-1, keepdim=True).values - exponents + 5
        quantum = torch.ones_like(shifts) << shifts.clamp(max=24)
        aligned = bits & -quantum
        aligned = torch.where(shifts >= 24, bits & -0x80000000, aligned)
        output = output + aligned.view(torch.float32).sum(-1)
    return output


def low_reference(case, inputs):
    """Quantized online attention over the full KV range, without a planner."""
    outputs, lses = [], []
    for batch, (qn, kn) in enumerate(zip(case.q_lengths, case.kv_lengths)):
        output = torch.empty(qn, case.q_heads, 128, dtype=torch.bfloat16)
        lse = torch.empty(case.q_heads, qn, dtype=torch.float32)
        for head in range(case.q_heads):
            for q0 in range(0, qn, 128):
                rows = min(128, qn - q0)
                result, row_lse = partial_reference(
                    case, inputs, batch, head, q0, rows, 0, (kn + 255) // 256
                )
                valid_rows = visible(case, batch, q0, rows, 0, kn).any(-1)
                output[q0 : q0 + rows, head] = torch.where(valid_rows[:, None], result, 0).to(
                    torch.bfloat16
                )
                lse[head, q0 : q0 + rows] = torch.where(valid_rows, row_lse, torch.inf)
        outputs.append(output)
        lses.append(lse)
    return torch.cat(outputs), torch.cat(lses, dim=1)


def ref_flash_attention_pair(case, inputs):
    """Return the two FP8 references, using the shared golden cache."""
    names = ("high_o", "high_lse", "low_o", "low_lse")

    def compute():
        return dict(zip(names, (*high_reference(case, inputs), *low_reference(case, inputs))))

    api_name = {
        "dense": "flash_attn_func",
        "varlen": "flash_attn_varlen_func",
        "cache": "flash_attn_with_kvcache",
    }[case.entry]
    cache_args = dict(
        nodeid=os.environ.get("GOLDEN_CACHE_NODEID", "standalone"),
        api_name=api_name,
        metadata=asdict(case),
        inputs={name: inputs[name] for name in ("q", "k", "v", "descales")},
        compute_fn=compute,
        expected_keys=names,
        source_files=[__file__],
    )
    values, status = get_or_compute_golden(**cache_args, return_status=True)
    if status == "hit":
        register_retry(values, lambda: get_or_compute_golden(**cache_args, force_refresh=True))
    return (values["high_o"], values["high_lse"]), (values["low_o"], values["low_lse"])
