# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Ascend 950 BSND attention with score/mask callbacks and optional block sparsity.

Each task processes one 128-row Q tile for one batch and Q head. Cube computes
QK and PV; vector stages apply callbacks, online softmax and output rescaling.
QK is issued two pipeline iterations before its PV. L1 P and UB alpha each use
three slots, rotated by active-block ordinal. Physical KV indices select data
and callback coordinates.
"""

import catlass.tla as tla

from .modifiers import (
    process_score_tile_simd,
    process_score_tile_simt,
)
from .seqlen_info import create_seqlen_info

from .softmax import (
    output_empty_row,
    output_pack_row,
    output_update_tile,
    online_softmax_tile,
)

DTYPE_S = tla.Float32
DTYPE_OTMP = tla.Float32
DTYPE_ACC = tla.Float32
L0_TILE_M = 128
L0_TILE_N = 128
L0_TILE_K = 128
PRE_LAUNCH = 2
L0_STAGES = 2
K_L1_BUF = 2
V_L1_BUF = 2


@tla.jit
def _sparse_block(sparse_tensors, ordinal, partial_count, partial_base, full_base):
    """Return the KV block index and kind (0 partial, 1 full) for an active ordinal.

    Partial entries precede full entries. QK and delayed PV use the same mapping
    so each P tile is multiplied by the V tile from its corresponding KV block.
    """
    physical_index = tla.as_numeric(0)
    block_kind = tla.as_numeric(1)
    if tla.const_expr(sparse_tensors[1] is not None):
        if tla.const_expr(sparse_tensors[3] is not None):
            if ordinal < partial_count:
                physical_index = sparse_tensors[1][partial_base + ordinal]
                block_kind = 0
            else:
                physical_index = sparse_tensors[3][full_base + ordinal - partial_count]
        else:
            physical_index = sparse_tensors[1][partial_base + ordinal]
            block_kind = 0
    elif tla.const_expr(sparse_tensors[3] is not None):
        physical_index = sparse_tensors[3][full_base + ordinal]
    return physical_index, block_kind


@tla.kernel
def flex_attention_kernel(
    query: tla.Tensor,
    key: tla.Tensor,
    value: tla.Tensor,
    attention_out: tla.Tensor,
    lse,
    sparse_tensors,
    aux_tensors,
    aux_scalars,
    batch: tla.Constexpr[int],
    q_heads: tla.Constexpr[int],
    kv_heads: tla.Constexpr[int],
    q_len: tla.Constexpr[int | None],
    kv_len: tla.Constexpr[int | None],
    scale: tla.Constexpr[float],
    is_fp16: tla.Constexpr[bool],
    mask_mod: tla.Constexpr,
    score_mod: tla.Constexpr,
    window_left: tla.Constexpr = None,
    window_right: tla.Constexpr = None,
    mask_uses_simd: tla.Constexpr[bool] = False,
    score_uses_simd: tla.Constexpr[bool] = False,
    head_dim: tla.Constexpr[int] = 128,
    sparse_strides: tla.Constexpr = None,
) -> None:
    if tla.const_expr(q_len is None):
        # Contiguous BSND roots use (B*S, H*D); B/H/D stay compile-time facts.
        q_len = query.shape[0] // batch
        kv_len = key.shape[0] // batch
    mixed_modifiers = (
        mask_mod is not None and score_mod is not None and mask_uses_simd != score_uses_simd
    )
    use_simd_modifiers = mask_uses_simd if mask_mod is not None else score_uses_simd
    if tla.const_expr(is_fp16):
        DTYPE_Q = tla.Float16
        DTYPE_K = tla.Float16
        DTYPE_V = tla.Float16
        DTYPE_P = tla.Float16
    else:
        DTYPE_Q = tla.BFloat16
        DTYPE_K = tla.BFloat16
        DTYPE_V = tla.BFloat16
        DTYPE_P = tla.BFloat16
    c0 = 0
    c1 = 1
    qBaseTile_ = 128
    kvBaseTile_ = 128
    embed_ = head_dim
    curQSeqlen = q_len
    curKvSeqlen = kv_len
    qNOffset = head_dim
    qSOffset = q_heads * head_dim
    kNOffset = head_dim
    kSOffset = kv_heads * head_dim
    vSOffset = kv_heads * head_dim
    mm1L0ATotalStages_ = 1
    mm1L0BTotalStages_ = 1
    mm2L0ATotalStages_ = 1
    mm2L0BTotalStages_ = 1
    q_tiles = (q_len + 127) // 128
    kv_tiles = (kv_len + 127) // 128
    total_tasks = batch * q_heads * q_tiles
    seqlen_info = create_seqlen_info(q_len, kv_len)
    # --- L1->L0 完成 -> GM->L1 可开始 ---
    q_l0a_ready_l1 = tla.flag("l0a_ready_l1", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_0 = tla.flag("k_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_1 = tla.flag("k_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_0 = tla.flag("v_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_1 = tla.flag("v_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)

    # --- CUBE MMAD 完成 -> L1->L0 可开始 ---
    mmad_ready_l0a_0 = tla.flag("mmad_ready_l0a_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0a_1 = tla.flag("mmad_ready_l0a_1", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_0 = tla.flag("mmad_ready_l0b_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_1 = tla.flag("mmad_ready_l0b_1", tla.arch.CUBE, tla.arch.MTE1)

    # --- FIX (L0C->UB) 完成 -> CUBE 可开始 ---
    fix_ready_mmad_0 = tla.flag("fix_ready_mmad_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_1 = tla.flag("fix_ready_mmad_1", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_2 = tla.flag("fix_ready_mmad_2", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_3 = tla.flag("fix_ready_mmad_3", tla.arch.FIX, tla.arch.CUBE)
    # --- GM->L1 加载完成 -> L1->L0 可开始 ---
    q_l1_ready_l0 = tla.flag("q_l1_ready_l0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_0 = tla.flag("k_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_1 = tla.flag("k_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_0 = tla.flag("v_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_1 = tla.flag("v_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    # --- L1->L0 完成 -> CUBE 可开始 ---
    l0a_ready_mmad_0 = tla.flag("l0a_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0a_ready_mmad_1 = tla.flag("l0a_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_0 = tla.flag("l0b_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_1 = tla.flag("l0b_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    # --- CUBE MMAD 完成 -> FIX (L0C -> UB) 可开始 ---
    mmad_ready_fix_0 = tla.flag("mmad_ready_fix_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_1 = tla.flag("mmad_ready_fix_1", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_2 = tla.flag("mmad_ready_fix_2", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_3 = tla.flag("mmad_ready_fix_3", tla.arch.CUBE, tla.arch.FIX)

    # MTE3 -> VECTOR: P has reached L1, so its UB slot can be reused.
    mte3_ready_softmax_0 = tla.flag("mte3_ready_softmax_0", tla.arch.MTE3, tla.arch.VECTOR)
    mte3_ready_softmax_1 = tla.flag("mte3_ready_softmax_1", tla.arch.MTE3, tla.arch.VECTOR)
    # MTE3 -> VECTOR: output UB can be reused after GM writeback.
    mte3_ready_rescale = tla.flag("mte3_ready_rescale", tla.arch.MTE3, tla.arch.VECTOR)

    p_ub_ready_l1_0 = tla.flag("p_ub_ready_l1_0", tla.arch.VECTOR, tla.arch.MTE3)
    p_ub_ready_l1_1 = tla.flag("p_ub_ready_l1_1", tla.arch.VECTOR, tla.arch.MTE3)

    mm1_ready_sm_0 = tla.cross_flag("mm1_ready_sm_0")
    mm1_ready_sm_1 = tla.cross_flag("mm1_ready_sm_1")

    mm2_ready_re_0 = tla.cross_flag("mm2_ready_re_0")
    mm2_ready_re_1 = tla.cross_flag("mm2_ready_re_1")

    sm_ready_mm2_0 = tla.cross_flag("sm_ready_mm2_0")
    sm_ready_mm2_1 = tla.cross_flag("sm_ready_mm2_1")
    sm_ready_mm2_2 = tla.cross_flag("sm_ready_mm2_2")

    with tla.cube():
        tla.set_flag(q_l0a_ready_l1)
        tla.set_flag(k_l0b_ready_l1_0)
        tla.set_flag(k_l0b_ready_l1_1)
        tla.set_flag(v_l0b_ready_l1_0)
        tla.set_flag(v_l0b_ready_l1_1)
        tla.set_flag(mmad_ready_l0a_0)
        tla.set_flag(mmad_ready_l0a_1)
        tla.set_flag(mmad_ready_l0b_0)
        tla.set_flag(mmad_ready_l0b_1)
        tla.set_flag(fix_ready_mmad_0)
        tla.set_flag(fix_ready_mmad_1)
        tla.set_flag(fix_ready_mmad_2)
        tla.set_flag(fix_ready_mmad_3)

        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)
    with tla.vector():
        tla.set_flag(mte3_ready_softmax_0)
        tla.set_flag(mte3_ready_softmax_1)
        tla.set_flag(mte3_ready_rescale)

        tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)

    # 片上内存分配
    l1Q_ptrs = [
        tla.allocate(qBaseTile_ * 128, DTYPE_Q, tla.AddressSpace.l1, 512),
    ]
    l1K_ptrs = [
        tla.allocate(128 * kvBaseTile_, DTYPE_K, tla.AddressSpace.l1, 512),
        tla.allocate(128 * kvBaseTile_, DTYPE_K, tla.AddressSpace.l1, 512),
    ]
    l1P_ptrs = [
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
    ]
    l1V_ptrs = [
        tla.allocate(kvBaseTile_ * 128, DTYPE_V, tla.AddressSpace.l1, 512),
        tla.allocate(kvBaseTile_ * 128, DTYPE_V, tla.AddressSpace.l1, 512),
    ]

    l0a_ptrs = [
        tla.allocate(L0_TILE_M * L0_TILE_K, DTYPE_Q, tla.AddressSpace.l0a, 512),
        tla.allocate(L0_TILE_M * L0_TILE_K, DTYPE_Q, tla.AddressSpace.l0a, 512),
    ]
    l0b_ptrs = [
        tla.allocate(L0_TILE_K * L0_TILE_N, DTYPE_K, tla.AddressSpace.l0b, 512),
        tla.allocate(L0_TILE_K * L0_TILE_N, DTYPE_K, tla.AddressSpace.l0b, 512),
    ]
    l0c_ptrs = [
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
    ]

    ubS_ptrs = [
        tla.allocate(qBaseTile_ // 2 * kvBaseTile_, DTYPE_S, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2 * kvBaseTile_, DTYPE_S, tla.AddressSpace.ub, 256),
    ]
    ubP_ptrs = [
        tla.allocate((qBaseTile_ // 2 + 1) * kvBaseTile_, DTYPE_P, tla.AddressSpace.ub, 256),
        tla.allocate((qBaseTile_ // 2 + 1) * kvBaseTile_, DTYPE_P, tla.AddressSpace.ub, 256),
    ]
    ubOTmp_ptrs = [
        tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256),
    ]

    # O 累加器 + 行统计标量
    ubO_ptr = tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256)
    ubO16_ptr = tla.recast_ptr(ubO_ptr, dtype=DTYPE_Q)
    expMax_ptrs = [
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
    ]
    lastMax_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)
    lastSum_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)

    if tla.const_expr(lse is not None):
        lse_ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    else:
        lse_ub_ptr = None
    row_max_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    tile_sum_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    # Interleave heads/batches for each Q tile; metadata keeps its logical order.
    for work in tla.range(tla.arch.block_idx(), total_tasks, tla.arch.block_num()):
        q_tile = work // (batch * q_heads)
        batch_head = work % (batch * q_heads)
        qSTileIdx = q_tile
        qHeadIdx = batch_head % q_heads
        curBatch = batch_head // q_heads
        kvHeadIdx = qHeadIdx // (q_heads // kv_heads)
        qBOffset = curBatch * q_len * qSOffset
        kBOffset = curBatch * kv_len * kSOffset
        gmOffsetV = curBatch * kv_len * vSOffset + kvHeadIdx * head_dim
        rowNum = min(128, q_len - qSTileIdx * 128)
        first_kv = 0
        partial_count = 0
        partial_base = 0
        full_base = 0
        if tla.const_expr(sparse_tensors is None):
            end_kv = kv_tiles
            if tla.const_expr(window_left is not None):
                first_kv = min(
                    kv_tiles,
                    max(0, (qSTileIdx * 128 + kv_len - q_len - window_left) // 128),
                )
            if tla.const_expr(window_right is not None):
                end_kv = min(
                    kv_tiles,
                    max(
                        0,
                        (qSTileIdx * 128 + rowNum + kv_len - q_len + window_right + 127) // 128,
                    ),
                )
            kvSLoopNum = max(0, end_kv - first_kv)
        else:
            full_count = 0
            if tla.const_expr(sparse_tensors[1] is not None):
                partial_count = sparse_tensors[0][
                    curBatch * sparse_strides[0][0]
                    + qHeadIdx * sparse_strides[0][1]
                    + qSTileIdx * sparse_strides[0][2]
                ]
                partial_base = (
                    curBatch * sparse_strides[1][0]
                    + qHeadIdx * sparse_strides[1][1]
                    + qSTileIdx * sparse_strides[1][2]
                )
            if tla.const_expr(sparse_tensors[3] is not None):
                full_count = sparse_tensors[2][
                    curBatch * sparse_strides[2][0]
                    + qHeadIdx * sparse_strides[2][1]
                    + qSTileIdx * sparse_strides[2][2]
                ]
                full_base = (
                    curBatch * sparse_strides[3][0]
                    + qHeadIdx * sparse_strides[3][1]
                    + qSTileIdx * sparse_strides[3][2]
                )
            kvSLoopNum = partial_count + full_count
        if kvSLoopNum > 0:
            with tla.cube():
                gm_q = tla.make_tensor(
                    query.ptr + qBOffset + qHeadIdx * head_dim,
                    tla.make_layout(tla.make_shape(q_len, head_dim), tla.make_stride(qSOffset, 1)),
                )
                gm_q_tile = tla.tile_view(
                    gm_q, tla.make_shape(128, head_dim), tla.make_coord(qSTileIdx, 0)
                )
                l1_q = tla.make_tensor_like(l1Q_ptrs[0], gm_q_tile, tla.arch.zN)
                tla.wait_flag(q_l0a_ready_l1)
                tla.copy(l1_q, gm_q_tile)
                tla.set_flag(q_l1_ready_l0)
                tla.wait_flag(q_l1_ready_l0)

            for step in tla.range(0, kvSLoopNum + PRE_LAUNCH):
                if step < kvSLoopNum:
                    validIdx = step
                    gatheredKvSTileIdx = first_kv + validIdx
                    block_kind = 0
                    if tla.const_expr(mask_mod is None):
                        block_kind = 1
                    if tla.const_expr(sparse_tensors is not None):
                        gatheredKvSTileIdx, block_kind = _sparse_block(
                            sparse_tensors,
                            validIdx,
                            partial_count,
                            partial_base,
                            full_base,
                        )
                    kvSTileSizeAct = min(128, kv_len - gatheredKvSTileIdx * 128)
                    ubSBufId = validIdx % 2
                    ubS_ptr = ubS_ptrs[0] if ubSBufId == 0 else ubS_ptrs[1]
                    ubSTensorTla = tla.make_tensor(
                        ubS_ptr,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeAct),
                            tla.make_stride(128, 1),
                        ),
                    )
                    # QK Mmad
                    with tla.cube():
                        gm_q = tla.make_tensor(
                            query.ptr + qBOffset + qHeadIdx * qNOffset,
                            tla.make_layout(
                                tla.make_shape(curQSeqlen, embed_),
                                tla.make_stride(qSOffset, 1),
                            ),
                        )

                        gmQTensorTla = tla.tile_view(
                            gm_q,
                            tla.make_shape(qBaseTile_, embed_),
                            tla.make_coord(qSTileIdx, c0),
                        )
                        l1_q = tla.make_tensor_like(l1Q_ptrs[0], gmQTensorTla, tla.arch.zN)

                        gm_k = tla.make_tensor(
                            key.ptr + (kBOffset + kvHeadIdx * kNOffset),
                            tla.make_layout(
                                tla.make_shape(embed_, curKvSeqlen),
                                tla.make_stride(1, kSOffset),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )

                        prefixSumL0AStages = (
                            (validIdx * mm1L0ATotalStages_)
                            if validIdx <= PRE_LAUNCH
                            else (
                                validIdx * mm1L0ATotalStages_
                                + (validIdx - PRE_LAUNCH) * mm2L0ATotalStages_
                            )
                        )
                        prefixSumL0BStages = (
                            (validIdx * mm1L0BTotalStages_)
                            if validIdx <= PRE_LAUNCH
                            else (
                                validIdx * mm1L0BTotalStages_
                                + (validIdx - PRE_LAUNCH) * mm2L0BTotalStages_
                            )
                        )
                        # -----------------QK-----------------
                        nLoopCounterL1 = validIdx

                        # copy gm_k to L1
                        l1BBufId = nLoopCounterL1 % K_L1_BUF
                        l1K_ptr = l1K_ptrs[0] if l1BBufId == c0 else l1K_ptrs[1]
                        gm_k_tile = tla.tile_view(
                            gm_k,
                            tla.make_shape(embed_, kvBaseTile_),
                            tla.make_coord(c0, gatheredKvSTileIdx),
                        )
                        l1_k_tile = tla.make_tensor_like(l1K_ptr, gm_k_tile, layoutTag=tla.arch.nZ)
                        if l1BBufId == c0:
                            tla.wait_flag(k_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.wait_flag(k_l0b_ready_l1_1)
                        tla.copy(l1_k_tile, gm_k_tile)
                        if l1BBufId == c0:
                            tla.set_flag(k_l1_ready_l0_0)  # MTE2_MTE1
                        else:
                            tla.set_flag(k_l1_ready_l0_1)

                        # copy L1 to l0
                        l0CBufId = (nLoopCounterL1) % L0_STAGES
                        l0c_ptr = l0c_ptrs[0] if l0CBufId == c0 else l0c_ptrs[1]
                        ub_s_tile = tla.tile_view(
                            ubSTensorTla,
                            tla.make_shape(qBaseTile_, kvBaseTile_),
                            tla.make_coord(c0, c0),
                        )
                        l0c_s = tla.make_tensor_like(
                            l0c_ptr, ub_s_tile, layoutTag=tla.arch.L0Clayout
                        )

                        l0ALoopCounter = prefixSumL0AStages
                        l0BLoopCounter = prefixSumL0BStages
                        l0ABufId = l0ALoopCounter % L0_STAGES
                        l0BBufId = l0BLoopCounter % L0_STAGES

                        l0a_ptr = l0a_ptrs[0] if l0ABufId == c0 else l0a_ptrs[1]
                        l0a_q_tensor = tla.make_tensor_like(l0a_ptr, l1_q, tla.arch.zN)

                        if l0ABufId == 0:
                            tla.wait_flag(mmad_ready_l0a_0)  # CUBE_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0a_1)
                        tla.copy(l0a_q_tensor, l1_q)
                        if l0ABufId == 0:
                            tla.set_flag(l0a_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.set_flag(l0a_ready_mmad_1)

                        l0b_ptr = l0b_ptrs[0] if l0BBufId == c0 else l0b_ptrs[1]
                        l0b_k_tensor = tla.make_tensor_like(l0b_ptr, l1_k_tile)
                        if l0BBufId == 0:
                            tla.wait_flag(mmad_ready_l0b_0)  # CUBE_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0b_1)
                        if l1BBufId == c0:
                            tla.wait_flag(k_l1_ready_l0_0)  # MTE2_MTE1
                        else:
                            tla.wait_flag(k_l1_ready_l0_1)
                        tla.copy(l0b_k_tensor, l1_k_tile)

                        if l0BBufId == 0:
                            tla.set_flag(l0b_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.set_flag(l0b_ready_mmad_1)
                        if l1BBufId == 0:
                            tla.set_flag(k_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.set_flag(k_l0b_ready_l1_1)

                        if l0ABufId == 0:
                            tla.wait_flag(l0a_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.wait_flag(l0a_ready_mmad_1)
                        if l0BBufId == 0:
                            tla.wait_flag(l0b_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.wait_flag(l0b_ready_mmad_1)
                        if l0CBufId == 0:
                            tla.wait_flag(fix_ready_mmad_0)  # FIX_CUBE
                        else:
                            tla.wait_flag(fix_ready_mmad_1)

                        tla.mmad(l0c_s, l0a_q_tensor, l0b_k_tensor, init_c=True)

                        if l0ABufId == 0:
                            tla.set_flag(mmad_ready_l0a_0)  # CUBE_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0a_1)
                        if l0BBufId == 0:
                            tla.set_flag(mmad_ready_l0b_0)  # CUBE_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0b_1)

                        # ---- fixPipe：L0C(fp32) -> UB(fp32 S) ----
                        if ubSBufId == 0:
                            tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.FIX)
                        else:
                            tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.FIX)
                        if l0CBufId == 0:
                            tla.set_flag(mmad_ready_fix_0)  # CUBE-FIX
                            tla.wait_flag(mmad_ready_fix_0)  # CUBE-FIX
                        else:
                            tla.set_flag(mmad_ready_fix_1)
                            tla.wait_flag(mmad_ready_fix_1)

                        tla.copy(
                            ubSTensorTla,
                            l0c_s,
                            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
                        )

                        if l0CBufId == 0:
                            tla.set_flag(fix_ready_mmad_0)  # FIX_CUBE
                        else:
                            tla.set_flag(fix_ready_mmad_1)
                        if ubSBufId == 0:
                            tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.FIX)
                        else:
                            tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.FIX)

                        if validIdx == kvSLoopNum - 1:
                            tla.set_flag(q_l0a_ready_l1)  # MTE1_MTE2

                    l1PBufId = validIdx % 3
                    l1p_ptr = (
                        l1P_ptrs[0]
                        if l1PBufId == 0
                        else (l1P_ptrs[1] if l1PBufId == 1 else l1P_ptrs[2])
                    )
                    l1PTensorTla = tla.make_tensor_like(
                        l1p_ptr,
                        tla.tile_view(ubSTensorTla, tla.make_shape(128, 128), tla.make_coord(0, 0)),
                        tla.arch.zN,
                    )
                    with tla.vector():
                        subIdxEff = tla.arch.sub_block_idx()
                        mHalf = (rowNum + 1) // 2
                        m = mHalf if subIdxEff == 0 else rowNum - mHalf
                        if ubSBufId == 0:
                            tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                            tla.wait_flag(mte3_ready_softmax_0)
                        else:
                            tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                            tla.wait_flag(mte3_ready_softmax_1)
                        if m > 0:
                            ubP_ptr = ubP_ptrs[0] if ubSBufId == 0 else ubP_ptrs[1]
                            expMax_ptr = (
                                expMax_ptrs[0]
                                if l1PBufId == 0
                                else (expMax_ptrs[1] if l1PBufId == 1 else expMax_ptrs[2])
                            )
                            ub_s = tla.make_tensor(
                                ubS_ptr,
                                tla.make_layout(tla.make_shape(m, 128), tla.make_stride(128, 1)),
                            )
                            score_flat = tla.make_tensor(
                                ubS_ptr,
                                tla.make_layout(tla.make_shape(64 * 128), tla.make_stride(1)),
                            )
                            ub_p = tla.make_tensor(
                                ubP_ptr,
                                tla.make_layout(tla.make_shape(65, 128), tla.make_stride(128, 1)),
                            )
                            ub_p_zN_full = tla.make_tensor_like(ubP_ptr, ub_p, tla.arch.zNUnAlign)
                            state_layout = tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                            running_max = tla.make_tensor(lastMax_ptr, state_layout)
                            running_sum = tla.make_tensor(lastSum_ptr, state_layout)
                            alpha = tla.make_tensor(expMax_ptr, state_layout)
                            row_max = tla.make_tensor(row_max_ptr, state_layout)
                            tile_sum = tla.make_tensor(tile_sum_ptr, state_layout)
                            # FIX -> modifier -> softmax share this UB allocation.
                            tla.pipe_barrier(tla.pipes.ALL)
                            if tla.const_expr(mixed_modifiers):
                                # Each writer completes before the next mode reads
                                # the same UB tile. Custom masks override windows.
                                coordinates = (
                                    m,
                                    qSTileIdx * 128 + subIdxEff * mHalf,
                                    gatheredKvSTileIdx * 128,
                                    curBatch,
                                    qHeadIdx,
                                )
                                scalar_inputs = (
                                    score_flat,
                                    *coordinates,
                                    0,
                                    aux_tensors,
                                    aux_scalars,
                                    q_len,
                                    kv_len,
                                    scale,
                                    seqlen_info,
                                )
                                vector_inputs = (
                                    ub_s,
                                    *coordinates,
                                    kv_len,
                                    scale,
                                    seqlen_info,
                                    aux_tensors,
                                    aux_scalars,
                                    None,
                                    None,
                                )
                                if tla.const_expr(score_uses_simd):
                                    process_score_tile_simd(
                                        *vector_inputs,
                                        mask_mod=None,
                                        score_mod=score_mod,
                                        apply_scale=True,
                                        apply_mask=False,
                                    )
                                else:
                                    process_score_tile_simt(
                                        *scalar_inputs,
                                        mask_mod=None,
                                        score_mod=score_mod,
                                        apply_scale=True,
                                    )
                                tla.pipe_barrier(tla.pipes.ALL)
                                if block_kind == 0:
                                    if tla.const_expr(mask_uses_simd):
                                        process_score_tile_simd(
                                            *vector_inputs,
                                            mask_mod=mask_mod,
                                            score_mod=None,
                                            apply_scale=False,
                                            apply_mask=True,
                                        )
                                    else:
                                        process_score_tile_simt(
                                            *scalar_inputs,
                                            mask_mod=mask_mod,
                                            score_mod=None,
                                            apply_scale=False,
                                        )
                                    tla.pipe_barrier(tla.pipes.ALL)
                                scores_are_scaled = True
                            else:
                                use_simt = True
                                if tla.const_expr(use_simd_modifiers):
                                    use_simt = False
                                elif tla.const_expr(score_mod is None):
                                    if tla.const_expr(mask_mod is None):
                                        use_simt = False
                                    else:
                                        use_simt = block_kind == 0
                                if use_simt:
                                    process_score_tile_simt(
                                        score_flat,
                                        m,
                                        qSTileIdx * 128 + subIdxEff * mHalf,
                                        gatheredKvSTileIdx * 128,
                                        curBatch,
                                        qHeadIdx,
                                        block_kind,
                                        aux_tensors,
                                        aux_scalars,
                                        q_len,
                                        kv_len,
                                        scale,
                                        seqlen_info,
                                        mask_mod,
                                        score_mod,
                                    )
                                    tla.pipe_barrier(tla.pipes.ALL)
                                if tla.const_expr(use_simd_modifiers):
                                    if block_kind == 0:
                                        process_score_tile_simd(
                                            ub_s,
                                            m,
                                            qSTileIdx * 128 + subIdxEff * mHalf,
                                            gatheredKvSTileIdx * 128,
                                            curBatch,
                                            qHeadIdx,
                                            kv_len,
                                            scale,
                                            seqlen_info,
                                            aux_tensors,
                                            aux_scalars,
                                            window_left,
                                            window_right,
                                            mask_mod,
                                            score_mod,
                                            True,
                                            True,
                                        )
                                    else:
                                        process_score_tile_simd(
                                            ub_s,
                                            m,
                                            qSTileIdx * 128 + subIdxEff * mHalf,
                                            gatheredKvSTileIdx * 128,
                                            curBatch,
                                            qHeadIdx,
                                            kv_len,
                                            scale,
                                            seqlen_info,
                                            aux_tensors,
                                            aux_scalars,
                                            window_left,
                                            window_right,
                                            mask_mod,
                                            score_mod,
                                            True,
                                            False,
                                        )
                                    tla.pipe_barrier(tla.pipes.ALL)
                                elif tla.const_expr(
                                    window_left is not None or window_right is not None
                                ):
                                    # A scalar score is already scaled by the SIMT stage.
                                    process_score_tile_simd(
                                        ub_s,
                                        m,
                                        qSTileIdx * 128 + subIdxEff * mHalf,
                                        gatheredKvSTileIdx * 128,
                                        curBatch,
                                        qHeadIdx,
                                        kv_len,
                                        scale,
                                        seqlen_info,
                                        aux_tensors,
                                        aux_scalars,
                                        window_left,
                                        window_right,
                                        None,
                                        None,
                                        score_mod is None,
                                        False,
                                    )
                                    tla.pipe_barrier(tla.pipes.ALL)
                                scores_are_scaled = use_simt
                                if tla.const_expr(
                                    use_simd_modifiers
                                    or window_left is not None
                                    or window_right is not None
                                ):
                                    scores_are_scaled = True
                            if scores_are_scaled:
                                online_softmax_tile(
                                    ub_s,
                                    ub_p_zN_full,
                                    running_max,
                                    running_sum,
                                    alpha,
                                    validIdx == 0,
                                    scale,
                                    kvSTileSizeAct,
                                    False,
                                    m,
                                    DTYPE_P,
                                    row_max,
                                    tile_sum,
                                )
                            else:
                                online_softmax_tile(
                                    ub_s,
                                    ub_p_zN_full,
                                    running_max,
                                    running_sum,
                                    alpha,
                                    validIdx == 0,
                                    scale,
                                    kvSTileSizeAct,
                                    True,
                                    m,
                                    DTYPE_P,
                                    row_max,
                                    tile_sum,
                                )
                            if ubSBufId == 0:
                                tla.set_flag(p_ub_ready_l1_0)
                                tla.wait_flag(p_ub_ready_l1_0)
                            else:
                                tla.set_flag(p_ub_ready_l1_1)
                                tla.wait_flag(p_ub_ready_l1_1)
                        if ubSBufId == 0:
                            tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                        else:
                            tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                        if l1PBufId == 0:
                            tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
                        elif l1PBufId == 1:
                            tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
                        else:
                            tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
                        if m > 0:
                            ubP_copy_ptr = ubP_ptrs[0] if ubSBufId == 0 else ubP_ptrs[1]
                            p_copy_source = tla.make_tensor(
                                ubP_copy_ptr,
                                tla.make_layout(tla.make_shape(65, 128), tla.make_stride(128, 1)),
                            )
                            p_copy_source_zn = tla.make_tensor_like(
                                ubP_copy_ptr, p_copy_source, tla.arch.zNUnAlign
                            )
                            p_copy_tile = tla.tile_view(
                                p_copy_source_zn,
                                tla.make_shape(m, kvSTileSizeAct),
                                tla.make_coord(0, 0),
                            )
                            l1P_tile = tla.tile_view(
                                l1PTensorTla,
                                tla.make_shape(mHalf, kvSTileSizeAct),
                                tla.make_coord(subIdxEff, 0),
                            )
                            tla.copy(l1P_tile, p_copy_tile)
                        if ubSBufId == 0:
                            tla.set_flag(mte3_ready_softmax_0)
                        else:
                            tla.set_flag(mte3_ready_softmax_1)
                        if l1PBufId == 0:
                            tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                        elif l1PBufId == 1:
                            tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                        else:
                            tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)

                if step >= PRE_LAUNCH:
                    validIdxDe = step - PRE_LAUNCH
                    gatheredKvSTileIdxDe = first_kv + validIdxDe
                    if tla.const_expr(sparse_tensors is not None):
                        gatheredKvSTileIdxDe, _ = _sparse_block(
                            sparse_tensors,
                            validIdxDe,
                            partial_count,
                            partial_base,
                            full_base,
                        )
                    kvSTileSizeActDe = min(128, kv_len - gatheredKvSTileIdxDe * 128)
                    ubOTmpBufId = validIdxDe % 2
                    ubOTmp_ptr = ubOTmp_ptrs[0] if ubOTmpBufId == 0 else ubOTmp_ptrs[1]
                    ubOTmpTensorTla = tla.make_tensor(
                        ubOTmp_ptr,
                        tla.make_layout(tla.make_shape(rowNum, head_dim), tla.make_stride(128, 1)),
                    )
                    l1PBufIdDe = validIdxDe % 3
                    l1P_ptrDe = (
                        l1P_ptrs[0]
                        if l1PBufIdDe == 0
                        else (l1P_ptrs[1] if l1PBufIdDe == 1 else l1P_ptrs[2])
                    )
                    p_shape = tla.make_tensor(
                        ubOTmp_ptr,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeActDe),
                            tla.make_stride(128, 1),
                        ),
                    )
                    l1PTensorTlaDe = tla.make_tensor_like(l1P_ptrDe, p_shape, tla.arch.zN)

                    # PV Mmad
                    with tla.cube():
                        kvShapeRowDe = curKvSeqlen
                        kvShapeColDe = tla.as_numeric(embed_)
                        gm_v = tla.make_tensor(
                            value.ptr + gmOffsetV,
                            tla.make_layout(
                                tla.make_shape(kvShapeRowDe, kvShapeColDe),
                                tla.make_stride(vSOffset, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        # QK and PV share L0 operands: count issued QK and earlier PV loads.
                        prefixSumL0AStagesDe = min(step + 1, kvSLoopNum) + validIdxDe
                        prefixSumL0BStagesDe = prefixSumL0AStagesDe
                        # Select V's L1 slot by active ordinal, not physical KV block index.
                        l1BvBufId = validIdxDe % V_L1_BUF
                        l1V_ptr = l1V_ptrs[0] if l1BvBufId == c0 else l1V_ptrs[1]
                        gm_v_tile = tla.tile_view(
                            gm_v,
                            tla.make_shape(kvBaseTile_, embed_),
                            tla.make_coord(gatheredKvSTileIdxDe, c0),
                        )
                        l1_v_tile = tla.make_tensor_like(l1V_ptr, gm_v_tile, layoutTag=tla.arch.zN)
                        if l1BvBufId == c0:
                            tla.wait_flag(v_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.wait_flag(v_l0b_ready_l1_1)
                        tla.copy(l1_v_tile, gm_v_tile)
                        if l1BvBufId == c0:
                            tla.set_flag(v_l1_ready_l0_0)  # MTE2_MTE1
                            tla.wait_flag(v_l1_ready_l0_0)  # MTE2_MTE1
                        else:
                            tla.set_flag(v_l1_ready_l0_1)
                            tla.wait_flag(v_l1_ready_l0_1)
                        if l1PBufIdDe == c0:
                            tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE1)
                        elif l1PBufIdDe == c1:
                            tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE1)
                        else:
                            tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE1)
                        # copy L1 to l0
                        nLoopCounter = validIdxDe
                        l0CBufIdDe = nLoopCounter % L0_STAGES  # Alternate the two PV result slots.
                        l0c_ptrDe = l0c_ptrs[2] if l0CBufIdDe == c0 else l0c_ptrs[3]
                        ub_o_tile = tla.tile_view(
                            ubOTmpTensorTla,
                            tla.make_shape(qBaseTile_, embed_),
                            tla.make_coord(c0, c0),
                        )
                        l0c_o = tla.make_tensor_like(
                            l0c_ptrDe, ub_o_tile, layoutTag=tla.arch.L0Clayout
                        )

                        # Rotate operand slots using the combined QK/PV load count.
                        l0ALoopCounterDe = prefixSumL0AStagesDe
                        l0BLoopCounterDe = prefixSumL0BStagesDe
                        l0ABufIdDe = l0ALoopCounterDe % L0_STAGES
                        l0BBufIdDe = l0BLoopCounterDe % L0_STAGES
                        # V: L1 -> L0B
                        l0b_ptrDe = l0b_ptrs[0] if l0BBufIdDe == c0 else l0b_ptrs[1]
                        l0_b2 = tla.make_tensor_like(l0b_ptrDe, l1_v_tile, tla.arch.nZ)
                        if l0BBufIdDe == c0:
                            tla.wait_flag(mmad_ready_l0b_0)  # M_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0b_1)
                        tla.copy(l0_b2, l1_v_tile)  # copyL1ToL0B
                        if l0BBufIdDe == c0:
                            tla.set_flag(l0b_ready_mmad_0)  # MTE1_M
                        else:
                            tla.set_flag(l0b_ready_mmad_1)
                        if l1BvBufId == c0:
                            tla.set_flag(v_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.set_flag(v_l0b_ready_l1_1)
                        l1_p_l0 = tla.tile_view(
                            l1PTensorTlaDe,
                            tla.make_shape(128, 128),
                            tla.make_coord(c0, c0),
                        )
                        l0a_ptrDe = l0a_ptrs[0] if l0ABufIdDe == c0 else l0a_ptrs[1]
                        l0_a2 = tla.make_tensor_like(l0a_ptrDe, l1PTensorTlaDe, tla.arch.zN)
                        if l0ABufIdDe == c0:
                            tla.wait_flag(mmad_ready_l0a_0)  # M_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0a_1)
                        tla.copy(l0_a2, l1_p_l0)  # copyL1ToL0A
                        if l0ABufIdDe == c0:
                            tla.set_flag(l0a_ready_mmad_0)  # MTE1_M
                        else:
                            tla.set_flag(l0a_ready_mmad_1)
                        if l1PBufIdDe == c0:
                            tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
                        elif l1PBufIdDe == c1:
                            tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
                        else:
                            tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)

                        if l0ABufIdDe == c0:
                            tla.wait_flag(l0a_ready_mmad_0)  # MTE1_M
                        else:
                            tla.wait_flag(l0a_ready_mmad_1)
                        if l0BBufIdDe == c0:
                            tla.wait_flag(l0b_ready_mmad_0)  # MTE1_M
                        else:
                            tla.wait_flag(l0b_ready_mmad_1)
                        if l0CBufIdDe == c0:
                            # FIX must finish reading this L0C slot before PV overwrites it.
                            tla.wait_flag(fix_ready_mmad_2)
                        else:
                            tla.wait_flag(fix_ready_mmad_3)
                        tla.mmad(l0c_o, l0_a2, l0_b2, init_c=True)
                        if l0ABufIdDe == c0:
                            tla.set_flag(mmad_ready_l0a_0)  # M_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0a_1)
                        if l0BBufIdDe == c0:
                            tla.set_flag(mmad_ready_l0b_0)  # M_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0b_1)
                        if ubOTmpBufId == c0:
                            tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.FIX)
                        else:
                            tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.FIX)
                        if l0CBufIdDe == c0:
                            # Publish the PV result before FIX copies it from L0C to UB.
                            tla.set_flag(mmad_ready_fix_2)
                            tla.wait_flag(mmad_ready_fix_2)  # M_FIX
                        else:
                            tla.set_flag(mmad_ready_fix_3)
                            tla.wait_flag(mmad_ready_fix_3)
                        tla.copy(
                            ubOTmpTensorTla,
                            l0c_o,
                            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
                        )
                        if l0CBufIdDe == c0:
                            tla.set_flag(fix_ready_mmad_2)  # FIX_M
                        else:
                            tla.set_flag(fix_ready_mmad_3)
                        if ubOTmpBufId == c0:
                            tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.FIX)
                        else:
                            tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.FIX)

                    with tla.vector():
                        re_subIdxEff = tla.arch.sub_block_idx()
                        re_mHalf = (rowNum + 1) // 2
                        re_m = re_mHalf if re_subIdxEff == 0 else rowNum - re_mHalf
                        if ubOTmpBufId == 0:
                            tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.VECTOR)
                        else:
                            tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.VECTOR)
                        tla.wait_flag(mte3_ready_rescale)
                        if re_m > 0:
                            expMax_ptrDe = (
                                expMax_ptrs[0]
                                if l1PBufIdDe == 0
                                else (expMax_ptrs[1] if l1PBufIdDe == 1 else expMax_ptrs[2])
                            )
                            re_out_layout = tla.make_layout(
                                tla.make_shape(re_m, 128), tla.make_stride(128, 1)
                            )
                            re_state_layout = tla.make_layout(
                                tla.make_shape(64), tla.make_stride(1)
                            )
                            re_pv_ub = tla.make_tensor(ubOTmp_ptr, re_out_layout)
                            re_output_ub = tla.make_tensor(ubO_ptr, re_out_layout)
                            re_alpha = tla.make_tensor(expMax_ptrDe, re_state_layout)
                            re_running_max = tla.make_tensor(lastMax_ptr, re_state_layout)
                            re_running_sum = tla.make_tensor(lastSum_ptr, re_state_layout)
                            re_lse_ub = None
                            if tla.const_expr(lse is not None):
                                re_lse_ub = tla.make_tensor(lse_ub_ptr, re_state_layout)
                            output_update_tile(
                                re_pv_ub,
                                re_output_ub,
                                re_alpha,
                                re_running_max,
                                re_running_sum,
                                re_lse_ub,
                                re_m,
                                validIdxDe,
                                kvSLoopNum,
                                head_dim,
                            )
                        if ubOTmpBufId == 0:
                            tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
                        else:
                            tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)
                        tla.set_flag(mte3_ready_rescale)
        else:
            with tla.vector():
                empty_subIdxEff = tla.arch.sub_block_idx()
                empty_mHalf = (rowNum + 1) // 2
                empty_m = empty_mHalf if empty_subIdxEff == 0 else rowNum - empty_mHalf
                tla.wait_flag(mte3_ready_rescale)
                if empty_m > 0:
                    empty_output_ub = tla.make_tensor(
                        ubO_ptr,
                        tla.make_layout(tla.make_shape(empty_m, 128), tla.make_stride(128, 1)),
                    )
                    empty_lse_ub = None
                    if tla.const_expr(lse is not None):
                        empty_lse_ub = tla.make_tensor(
                            lse_ub_ptr,
                            tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
                        )
                    with tla.vec.func(mode="simd"):
                        empty_full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        empty_one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                        for empty_row in tla.range(empty_m):
                            output_empty_row(
                                empty_output_ub,
                                empty_lse_ub,
                                empty_row,
                                empty_full_mask,
                                empty_one_mask,
                            )
                tla.set_flag(mte3_ready_rescale)

        # Every task, including an empty sparse row, writes its complete O/LSE.
        with tla.vector():
            final_subIdxEff = tla.arch.sub_block_idx()
            final_mHalf = (rowNum + 1) // 2
            final_m = final_mHalf if final_subIdxEff == 0 else rowNum - final_mHalf
            tla.wait_flag(mte3_ready_rescale)
            if final_m > 0:
                final_out_layout = tla.make_layout(
                    tla.make_shape(final_m, 128), tla.make_stride(128, 1)
                )
                final_output_ub = tla.make_tensor(ubO_ptr, final_out_layout)
                final_packed_output = tla.make_tensor(ubO16_ptr, final_out_layout)
                # O16 aliases the FP32 accumulator: finish normalization before
                # packing rows in increasing order into the same storage.
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    final_full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    final_packed_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                    for final_row in tla.range(final_m):
                        output_pack_row(
                            final_output_ub,
                            final_packed_output,
                            final_row,
                            DTYPE_Q,
                            final_full_mask,
                            final_packed_mask,
                        )
                tla.set_flag(p_ub_ready_l1_0)
                tla.wait_flag(p_ub_ready_l1_0)
                final_gm_out = tla.make_tensor(
                    attention_out.ptr
                    + qBOffset
                    + qHeadIdx * head_dim
                    + (qSTileIdx * 128 + final_subIdxEff * final_mHalf) * qSOffset,
                    tla.make_layout(
                        tla.make_shape(final_m, head_dim), tla.make_stride(qSOffset, 1)
                    ),
                )
                final_packed_valid = tla.tile_view(
                    final_packed_output,
                    tla.make_shape(final_m, head_dim),
                    tla.make_coord(0, 0),
                )
                tla.copy(final_gm_out, final_packed_valid)
                if tla.const_expr(lse is not None):
                    final_lse_layout = tla.make_layout(tla.make_shape(final_m), tla.make_stride(1))
                    final_gm_lse = tla.make_tensor(
                        lse.ptr
                        + (curBatch * q_heads + qHeadIdx) * q_len
                        + qSTileIdx * 128
                        + final_subIdxEff * final_mHalf,
                        final_lse_layout,
                    )
                    tla.copy(final_gm_lse, tla.make_tensor(lse_ub_ptr, final_lse_layout))
            tla.set_flag(mte3_ready_rescale)
    # Drain outstanding transfers and buffer-release flags before kernel exit.
    with tla.cube():
        tla.wait_flag(q_l0a_ready_l1)
        tla.wait_flag(k_l0b_ready_l1_0)
        tla.wait_flag(k_l0b_ready_l1_1)
        tla.wait_flag(v_l0b_ready_l1_0)
        tla.wait_flag(v_l0b_ready_l1_1)
        tla.wait_flag(mmad_ready_l0a_0)
        tla.wait_flag(mmad_ready_l0a_1)
        tla.wait_flag(mmad_ready_l0b_0)
        tla.wait_flag(mmad_ready_l0b_1)
        tla.wait_flag(fix_ready_mmad_0)
        tla.wait_flag(fix_ready_mmad_1)
        tla.wait_flag(fix_ready_mmad_2)
        tla.wait_flag(fix_ready_mmad_3)
        tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.FIX)
        tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.FIX)
        tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.FIX)
        tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.wait_flag(mte3_ready_softmax_0)
        tla.wait_flag(mte3_ready_softmax_1)
        tla.wait_flag(mte3_ready_rescale)
        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)
