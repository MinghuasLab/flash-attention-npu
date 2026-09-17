# Copyright (c) 2026, Minghua Shen.

"""Shared data, paged-KV, and mask construction utilities for attention tests."""

import torch

CASE_SEED = 42


def make_alibi_slopes(batch_size, num_heads):
    """Create the standard ALiBi slope table [batch_size, num_heads] on CPU fp32."""
    slopes = torch.tensor([0.5 / (2 ** h) for h in range(num_heads)], dtype=torch.float32)
    return slopes.unsqueeze(0).repeat(batch_size, 1)


def make_random_tensor(
    shape,
    data_type,
    *,
    low=-5.0,
    high=5.0,
    generator=None,
    device=None,
    requires_grad=False,
):
    """Create a random test tensor with consistent dtype, range, device, and grad settings.

    Random values are generated on CPU before moving to the target device to
    preserve the existing random-number order and reproducibility.
    """
    tensor = low + (high - low) * torch.rand(shape, generator=generator)
    tensor = tensor.to(data_type)
    if device is not None:
        if device == "npu":
            tensor = tensor.npu()
        else:
            tensor = tensor.to(device)
    return tensor.requires_grad_(requires_grad)


def make_attention_inputs(
    query_shape,
    key_shape,
    value_shape,
    dout_shape,
    data_type,
    *,
    generator=None,
    device=None,
    requires_grad=(True, True, True),
):
    """Create attention Q, K, V, and backward dout inputs consistently.

    Q/K/V use ``[-5, 5]`` by default, while dout uses ``[-1, 1]``. The caller
    supplies all input shapes, allowing reuse for BSND, packed TND, and paged
    KV. Each paged-KV test still constructs its own block table.
    """
    query = make_random_tensor(
        query_shape,
        data_type,
        generator=generator,
        device=device,
        requires_grad=requires_grad[0],
    )
    key = make_random_tensor(
        key_shape,
        data_type,
        generator=generator,
        device=device,
        requires_grad=requires_grad[1],
    )
    value = make_random_tensor(
        value_shape,
        data_type,
        generator=generator,
        device=device,
        requires_grad=requires_grad[2],
    )
    dout = make_random_tensor(
        dout_shape,
        data_type,
        low=-1.0,
        high=1.0,
        generator=generator,
        device=device,
    )
    return query, key, value, dout


def make_cu_seqlens(seqlens):
    """Convert per-batch sequence lengths to cumulative int32 offsets.

    Returns a CPU tensor of length ``len(seqlens) + 1`` whose first element is
    zero and whose last element is the sum of all sequence lengths. It can be
    passed directly as ``cu_seqlens`` for variable-length attention.
    """
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    for i, seqlen in enumerate(seqlens, start=1):
        cu[i] = cu[i - 1] + int(seqlen)
    return cu


def make_varlen_seqlens(batch_size, max_seqlen_q, max_seqlen_k, seed=CASE_SEED):
    """Generate reproducible variable Q/KV sequence lengths.

    Following Tri Dao's tests, valid lengths are sampled near the maximum from
    ``[max_seqlen - 20, max_seqlen]``. Returns separate Python lists for Q and
    KV lengths. A dedicated generator keeps these lengths independent of the
    order in which other random tensors are created.
    """
    generator = torch.Generator().manual_seed(seed)
    min_q = max(1, max_seqlen_q - 20)
    min_k = max(1, max_seqlen_k - 20)
    seqlens_q = torch.randint(min_q, max_seqlen_q + 1, (batch_size,), generator=generator).tolist()
    seqlens_k = []
    for q_seqlen in seqlens_q:
        # Keep the Tri Dao near-max distribution while preserving the
        # q_seqlen <= kv_seqlen contract used by the NPU varlen causal path.
        k_low = max(min_k, q_seqlen)
        if k_low > max_seqlen_k:
            k_low = max_seqlen_k
        seqlens_k.append(int(torch.randint(k_low, max_seqlen_k + 1, (1,), generator=generator).item()))
    return seqlens_q, seqlens_k


def make_packed_random_tensor(
    seqlens,
    max_seqlen,
    num_heads,
    head_size,
    data_type,
    *,
    generator=None,
    device=None,
    requires_grad=False,
):
    """Generate random TND inputs using Tri Dao's padded-to-unpadded flow."""
    padded = make_random_tensor(
        (len(seqlens), max_seqlen, num_heads, head_size),
        data_type,
        generator=generator,
    )
    valid = torch.arange(max_seqlen) < torch.tensor(seqlens)[:, None]
    packed = padded[valid]
    if device is not None:
        packed = packed.npu() if device == "npu" else packed.to(device)
    return packed.detach().requires_grad_(requires_grad)


def pad_packed_tensor(packed, seqlens, max_seqlen):
    """Restore a packed TND tensor to a padded 4D tensor using ``seqlens``."""
    valid = torch.arange(max_seqlen, device=packed.device) < torch.as_tensor(
        seqlens, device=packed.device
    )[:, None]
    padded = torch.zeros(
        (len(seqlens), max_seqlen, *packed.shape[1:]),
        dtype=packed.dtype,
        device=packed.device,
    )
    padded[valid] = packed
    return padded


def make_block_table(batch_size, kv_seqlen, block_size):
    """Create a paged-KV index table with contiguous physical blocks per batch.

    Returns an int32 CPU tensor shaped
    ``[batch_size, ceil(kv_seqlen / block_size)]``. Each row represents one
    batch, with contiguous physical block indices that do not overlap batches.
    """
    blocks_per_sequence = (kv_seqlen + block_size - 1) // block_size
    return torch.arange(
        batch_size * blocks_per_sequence,
        dtype=torch.int32,
    ).reshape(batch_size, blocks_per_sequence)


def make_paged_kv_cache(batch_size, kv_seqlen, block_size, kv_heads, head_size, data_type,
                        *, device="npu", requires_grad=False, generator=None):
    """Allocate paged K/V caches matching ``make_block_table``'s block count.

    ``make_block_table`` assigns ``ceil(kv_seqlen/block_size)`` physical blocks
    to each batch, so the cache must contain
    ``batch_size * ceil(kv_seqlen/block_size)`` blocks. Otherwise, long-KV
    cases such as Tri Dao's ``(1, 131072)`` can make the block table reference
    nonexistent blocks, causing a kernel-side DDR overrun (AICore exception
    0x800000) and cascading failures in later cases.
    """
    num_blocks = batch_size * ((kv_seqlen + block_size - 1) // block_size)
    key_cache = make_random_tensor((num_blocks, block_size, kv_heads, head_size), data_type,
                                   generator=generator, device=device, requires_grad=requires_grad)
    value_cache = make_random_tensor((num_blocks, block_size, kv_heads, head_size), data_type,
                                     generator=generator, device=device, requires_grad=requires_grad)
    return key_cache, value_cache


def gather_paged_kv(key_cache, value_cache, block_table_row, kv_seqlen, block_size):
    """Gather valid-length K/V values from paged indices for one batch.

    The first two dimensions of ``key_cache`` and ``value_cache`` are the
    physical block and in-block offset. ``block_table_row`` maps logical blocks
    to physical block indices. The returned K/V tensors both have
    ``kv_seqlen`` as their first dimension, and vectorized indexing preserves
    the input cache's autograd path.
    """
    block_table_row = block_table_row.to(device=key_cache.device, dtype=torch.long)
    positions = torch.arange(kv_seqlen, device=key_cache.device)
    block_indices = block_table_row[positions // block_size]
    block_offsets = positions % block_size
    return (
        key_cache[block_indices, block_offsets],
        value_cache[block_indices, block_offsets],
    )


def gather_paged_kv_batch(key_cache, value_cache, block_tables, kv_seqlen, block_size):
    """Batched paged-KV gather returning ``[B, kv_seqlen, H, D]``."""
    block_tables = block_tables.to(device=key_cache.device, dtype=torch.long)
    positions = torch.arange(kv_seqlen, device=key_cache.device)
    block_indices = block_tables[:, positions // block_size]
    block_offsets = positions % block_size
    return (
        key_cache[block_indices, block_offsets[None, :]],
        value_cache[block_indices, block_offsets[None, :]],
    )


def make_local_attention_mask(q_seqlen, kv_seqlen, window_size_left, window_size_right):
    """Build a right-aligned boolean mask for local attention.

    Returns a CPU tensor shaped ``[q_seqlen, kv_seqlen]``. ``True`` marks Q/KV
    positions outside the left/right window that the reference must mask.
    """
    left_boundary = kv_seqlen - q_seqlen - window_size_left
    right_boundary = kv_seqlen - q_seqlen + window_size_right
    row = torch.arange(q_seqlen)[:, None]
    col = torch.arange(kv_seqlen)[None, :]
    return ((-row + col) < left_boundary) | ((-row + col) > right_boundary)


def make_golden_attention_mask(q_seqlen, kv_seqlen, is_causal, window_size_left, window_size_right):
    """Normalize window parameters and build the reference attention mask.

    ``-1`` means unlimited context in that direction; windows extending beyond
    the KV length are also treated as unlimited. Causal mode normalizes the
    right window to zero. Returns
    ``(mask, is_causal_golden, is_local_golden)``. ``mask`` is ``None`` when no
    mask is needed; otherwise ``True`` marks masked positions.
    """
    wl = window_size_left
    wr = window_size_right
    if kv_seqlen > 0 and wl >= kv_seqlen:
        wl = -1
    if kv_seqlen > 0 and wr >= kv_seqlen:
        wr = -1
    if is_causal:
        wr = 0
    is_causal_golden = wl < 0 and wr == 0
    is_local_golden = (wl >= 0 or wr > 0) and not is_causal_golden
    if is_local_golden:
        if wl < 0:
            wl = kv_seqlen
        if wr < 0:
            wr = kv_seqlen
    if is_causal_golden:
        diagonal = kv_seqlen - q_seqlen + 1
        mask = torch.triu(torch.ones(q_seqlen, kv_seqlen), diagonal=diagonal).to(torch.bool)
    elif is_local_golden:
        mask = make_local_attention_mask(q_seqlen, kv_seqlen, wl, wr)
    else:
        mask = None
    return mask, is_causal_golden, is_local_golden


def make_padded_varlen_mask(
    q_seqlens,
    kv_seqlens,
    max_q_seqlen,
    max_kv_seqlen,
    is_causal,
    window_size_left,
    window_size_right,
):
    """Build batch-level valid positions and the attention mask for padded TND golden data."""
    q_seqlens = torch.as_tensor(q_seqlens)
    kv_seqlens = torch.as_tensor(kv_seqlens)
    q_valid = torch.arange(max_q_seqlen) < q_seqlens[:, None]
    k_valid = torch.arange(max_kv_seqlen) < kv_seqlens[:, None]
    mask = (~q_valid[:, :, None]) | (~k_valid[:, None, :])
    row = torch.arange(max_q_seqlen)[None, :, None]
    col = torch.arange(max_kv_seqlen)[None, None, :]
    if max_kv_seqlen > 0 and window_size_left >= max_kv_seqlen:
        window_size_left = -1
    if max_kv_seqlen > 0 and window_size_right >= max_kv_seqlen:
        window_size_right = -1
    if is_causal:
        window_size_right = 0
    is_causal_golden = window_size_left < 0 and window_size_right == 0
    is_local_golden = (window_size_left >= 0 or window_size_right > 0) and not is_causal_golden
    offsets = (kv_seqlens - q_seqlens)[:, None, None]
    if is_causal_golden:
        mask = mask | (col - row >= offsets + 1)
    elif is_local_golden:
        left = max_kv_seqlen if window_size_left < 0 else window_size_left
        right = max_kv_seqlen if window_size_right < 0 else window_size_right
        diff = col - row
        mask = mask | (diff < offsets - left) | (diff > offsets + right)
    return q_valid, k_valid, mask

def check_kvcache_inplace(key_cache_orig, value_cache_orig, key_cache, value_cache,
                          k_new, v_new, cache_seqlens, block_tables, block_size):
    """Validate the in-place append: cache[i][old:old+s_new] == k_new[i], the rest untouched.
    Non-paged: row-internal coordinates; paged: locate each token via the block table."""
    torch.npu.synchronize()
    kc, vc = key_cache.detach().cpu(), value_cache.detach().cpu()
    kc0, vc0 = key_cache_orig.detach().cpu(), value_cache_orig.detach().cpu()
    kn, vn = k_new.detach().cpu(), v_new.detach().cpu()
    sl = cache_seqlens.detach().cpu()
    bt = block_tables.detach().cpu() if block_tables is not None else None
    s_new = kn.shape[1]
    page_size = kc.shape[1] if bt is not None else 0
    n_fail = 0
    for i in range(len(sl)):
        old = int(sl[i])
        if bt is not None:
            table = bt[i]
            ok_region = all(
                torch.equal(kc[int(table[(old + j) // page_size]), (old + j) % page_size], kn[i, j]) and
                torch.equal(vc[int(table[(old + j) // page_size]), (old + j) % page_size], vn[i, j])
                for j in range(s_new))
            ok_keep = all(
                torch.equal(kc[int(table[j // page_size]), j % page_size], kc0[int(table[j // page_size]), j % page_size]) and
                torch.equal(vc[int(table[j // page_size]), j % page_size], vc0[int(table[j // page_size]), j % page_size])
                for j in range(old))
        else:
            ok_region = (torch.equal(kc[i, old:old + s_new], kn[i]) and
                         torch.equal(vc[i, old:old + s_new], vn[i]))
            ok_keep = (torch.equal(kc[i, :old], kc0[i, :old]) and
                       torch.equal(vc[i, :old], vc0[i, :old]) and
                       torch.equal(kc[i, old + s_new:], kc0[i, old + s_new:]) and
                       torch.equal(vc[i, old + s_new:], vc0[i, old + s_new:]))
        if not (ok_region and ok_keep):
            n_fail += 1
    assert n_fail == 0, f"kvcache in-place append mismatch: {n_fail}/{len(sl)} batches"