"""Deterministic merge of unnormalized FP32 Split-KV outputs and max/sum state."""

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


@tla.jit
def _broadcast_stat(stats, split, column):
    return tla.tile_view(stats, tla.make_shape(1, 1), tla.make_coord(split, column)).load(
        params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
    )


@tla.kernel
def flex_attention_split_combine_kernel(
    partial_out: tla.Tensor,
    partial_stats: tla.Tensor,
    out: tla.Tensor,
    lse,
    tokens: tla.Constexpr[int],
    heads: tla.Constexpr[int],
    dim: tla.Constexpr[int],
    splits: tla.Constexpr[int],
    is_fp16: tla.Constexpr[bool],
):
    rows = tokens * heads
    pitch = (dim + 63) // 64 * 64
    values_ptr = tla.allocate(splits * pitch, tla.Float32, tla.AddressSpace.ub, 256)
    stats_ptr = tla.allocate(splits * 8, tla.Float32, tla.AddressSpace.ub, 256)
    weight_capacity = (splits + 63) // 64 * 64
    weights_ptr = tla.allocate(weight_capacity, tla.Float32, tla.AddressSpace.ub, 256)
    output_dtype = tla.Float16 if is_fp16 else tla.BFloat16
    result_ptr = tla.allocate(pitch, output_dtype, tla.AddressSpace.ub, 256)
    if tla.const_expr(lse is not None):
        lse_ptr = tla.allocate(8, tla.Float32, tla.AddressSpace.ub, 256)
    load_ready = tla.flag("split_load_ready", tla.arch.MTE2, tla.arch.VECTOR)
    values_consumed = tla.flag("split_values_consumed", tla.arch.VECTOR, tla.arch.MTE2)
    compute_ready = tla.flag("split_compute_ready", tla.arch.VECTOR, tla.arch.MTE3)
    store_done = tla.flag("split_store_done", tla.arch.MTE3, tla.arch.VECTOR)
    with tla.vector():
        tla.set_flag(store_done)
        tla.set_flag(values_consumed)
        for row in tla.range(tla.arch.block_idx(), rows, tla.arch.block_num()):
            gm_values = tla.make_tensor(
                partial_out.ptr + row * dim,
                tla.make_layout(tla.make_shape(splits, dim), tla.make_stride(rows * dim, 1)),
            )
            values = tla.make_tensor(
                values_ptr, tla.make_layout(tla.make_shape(splits, dim), tla.make_stride(pitch, 1))
            )
            stats = tla.make_tensor(
                stats_ptr, tla.make_layout(tla.make_shape(splits, 2), tla.make_stride(8, 1))
            )
            gm_stats = tla.make_tensor(
                partial_stats.ptr + row * 2,
                tla.make_layout(tla.make_shape(splits, 2), tla.make_stride(rows * 2, 1)),
            )
            # Each row's readers finish before MTE2 reuses these two input buffers.
            tla.wait_flag(values_consumed)
            tla.copy(values, gm_values)
            tla.copy(stats, gm_stats)
            tla.set_flag(load_ready)
            tla.wait_flag(load_ready)
            tla.wait_flag(store_done)
            weights = tla.make_tensor(
                weights_ptr, tla.make_layout(tla.make_shape(weight_capacity), tla.make_stride(1))
            )
            result = tla.make_tensor(
                result_ptr, tla.make_layout(tla.make_shape(pitch), tla.make_stride(1))
            )
            with tla.vec.func(mode="simd"):
                full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                one, _ = tla.update_mask(1, dtype=tla.Float32)
                packed_mask, _ = tla.update_mask(64, dtype=tla.Float16)
                zero = tla.full(0.0, tla.Float32)
                negative_inf = tla.full(float("-inf"), tla.Float32)
                maximum = negative_inf
                for max_part in tla.range(splits):
                    max_count = _broadcast_stat(stats, max_part, 1)
                    maximum = tla.max(
                        maximum,
                        tla.where(
                            tla.cmp(max_count, zero, "gt", mask=full),
                            _broadcast_stat(stats, max_part, 0),
                            negative_inf,
                        ),
                        mask=full,
                    )
                valid = tla.cmp(maximum, negative_inf, "gt", mask=full)
                safe_maximum = tla.where(valid, maximum, zero)
                denominator = zero
                for weight_part in tla.range(splits):
                    weight_count = _broadcast_stat(stats, weight_part, 1)
                    part_weight = tla.where(
                        tla.cmp(weight_count, zero, "gt", mask=full),
                        tla.exp(_broadcast_stat(stats, weight_part, 0) - safe_maximum, mask=full),
                        zero,
                    )
                    denominator = denominator + part_weight * weight_count
                    tla.tile_view(weights, tla.make_shape(1), tla.make_coord(weight_part)).store(
                        part_weight, params=tla.params.UnalignStoreParams(), mask=one
                    )
                tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                divisor = tla.where(valid, denominator, tla.full(1.0, tla.Float32))
                for chunk in tla.range_constexpr((dim + 63) // 64):
                    numerator = zero
                    for value_part in tla.range(splits):
                        weight = tla.tile_view(
                            weights, tla.make_shape(1), tla.make_coord(value_part)
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        fragment = tla.tile_view(
                            values, tla.make_shape(1, 64), tla.make_coord(value_part, chunk)
                        ).load()
                        numerator = numerator + tla.where(
                            tla.cmp(weight, zero, "gt", mask=full), fragment * weight, zero
                        )
                    value = tla.where(valid, numerator / divisor, zero)
                    cast = value.to(
                        output_dtype,
                        tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.SAT,
                            round_mode=tla.params.RoundMode.CAST_ROUND,
                        ),
                        mask=full,
                    )
                    packed, _ = tla.deinterleave(cast, tla.full(0, output_dtype))
                    tla.tile_view(result, tla.make_shape(64), tla.make_coord(chunk)).store(
                        packed, mask=packed_mask
                    )
                if tla.const_expr(lse is not None):
                    logsum = tla.where(valid, maximum + tla.log(divisor, mask=full), negative_inf)
                    tla.make_tensor(
                        lse_ptr, tla.make_layout(tla.make_shape(1), tla.make_stride(1))
                    ).store(logsum, params=tla.params.UnalignStoreParams(), mask=one)
            tla.set_flag(values_consumed)
            tla.set_flag(compute_ready)
            tla.wait_flag(compute_ready)
            out_view = tla.make_tensor(
                out.ptr + row * dim, tla.make_layout(tla.make_shape(dim), tla.make_stride(1))
            )
            tla.copy(out_view, tla.tile_view(result, tla.make_shape(dim), tla.make_coord(0)))
            if tla.const_expr(lse is not None):
                head = row % heads
                token = row // heads
                lse_view = tla.make_tensor(
                    lse.ptr + head * tokens + token,
                    tla.make_layout(tla.make_shape(1), tla.make_stride(1)),
                )
                tla.copy(
                    lse_view,
                    tla.make_tensor(
                        lse_ptr, tla.make_layout(tla.make_shape(1), tla.make_stride(1))
                    ),
                )
            tla.set_flag(store_done)
        tla.wait_flag(store_done)
        tla.wait_flag(values_consumed)
        tla.pipe_barrier(tla.pipes.ALL)


def combine_split_results(partials, stats, out, lse, cores):
    import torch

    from .interface import _compilation_key, _compile_kernel

    splits, tokens, heads, dim = partials.shape
    tensors = (partials, stats, out, lse)
    constants = (tokens, heads, dim, splits, out.dtype == torch.float16)
    key = _compilation_key(tensors, None, None, constants, out.device)

    def compile_args():
        return (
            *(
                from_dlpack(t.reshape(-1), layout_tag=tla.arch.RowMajor) if t is not None else None
                for t in tensors
            ),
            *constants,
        )

    compiled = _compile_kernel(flex_attention_split_combine_kernel, key, compile_args)
    compiled(*tensors, block_num=min(tokens * heads, cores * 2))
