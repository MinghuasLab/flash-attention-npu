"""MXFP8 Flex attention on Ascend 950 (architecture 3510).
The fast fixed-exponent path and FP32 repair share scheduling, slots and channels.
The exact path uses FP32 online statistics and group32 RNE probability packing.
"""

from functools import lru_cache

import catlass.tla as tla

from .callbacks import prepare_scores_simt
from .flash_fwd_arch35 import _cache_mask, _cache_mask_pair, _score_max_pass

__all__ = ["flash_fwd_mxfp8_kernel"]
# E8M0 descriptors and RNE FP8 packing; vector CAST_ROUND is not ties-to-even.
_MX_CPP_HEADER = """#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;using namespace AscendC::Reg;
"""


@tla.jit
def q_scale_gm(ptr, rows, pitch, groups: tla.Constexpr = 4):
    # MX descriptors have a structural singleton axis with stride zero.
    # CATLASS 2.2 rejects a literal zero in make_stride, so retain its SSA
    # expression until the descriptor has passed frontend validation.
    return tla.make_tensor(
        tla.recast_ptr(ptr, dtype=tla.Float8E8M0),
        tla.make_layout(
            tla.make_shape((2, groups // 2), (1, rows)),
            tla.make_stride((1, 2), (rows - rows, pitch)),
            origin_shape=tla.make_shape(groups, rows),
            layoutTag=tla.arch.ColMajorMxScaleB,
        ),
    )


@tla.jit
def k_scale_gm(ptr, rows, pitch, groups: tla.Constexpr = 4):
    return tla.make_tensor(
        tla.recast_ptr(ptr, dtype=tla.Float8E8M0),
        tla.make_layout(
            tla.make_shape((1, rows), (2, groups // 2)),
            tla.make_stride((rows - rows, pitch), (1, 2)),
            origin_shape=tla.make_shape(rows, groups),
            layoutTag=tla.arch.RowMajorMxScaleA,
        ),
    )


@tla.jit
def v_scale_gm(ptr, groups, pair_pitch, width: tla.Constexpr = 128):
    return tla.make_tensor(
        tla.recast_ptr(ptr, dtype=tla.Float8E8M0),
        tla.make_layout(
            tla.make_shape((1, width), (2, groups // 2)),
            tla.make_stride((groups - groups, 2), (1, pair_pitch)),
            origin_shape=tla.make_shape(width, groups),
            layoutTag=tla.arch.ColMajorMxScaleA,
        ),
    )


@tla.jit
def probability_to_l1(prob_l1, scale_l1, prob_ub, scale_ub, sub):
    mx_prob64_to_l1(prob_l1 + sub * 8192, prob_ub)
    sref = tla.make_tensor(
        tla.recast_ptr(scale_ub, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
    )
    su = tla.make_tensor_like(sref.ptr, sref, tla.arch.zNUnAlign)
    sd = tla.make_tensor(
        tla.recast_ptr(scale_l1, dtype=tla.Float16),
        tla.make_layout_with_tag((16, 16), tla.Float16, tla.arch.zN),
    )
    tla.copy(tla.tile_view(sd, tla.make_shape(8, 16), tla.make_coord(sub, 0)), su)


@tla.jit
def _mx_group_scale(maximum, previous, group: tla.Constexpr, inverses, ph0, ph1):
    bits = maximum.bitcast(tla.Int32)
    power = tla.shift_right(bits, 23) - 135
    mantissa = tla.bitwise_and(bits, tla.full(0x7FFFFF, tla.Int32))
    power = power + tla.where(
        tla.cmp(mantissa, 0x600000, mode="gt"), tla.full(1, tla.Int32), tla.full(0, tla.Int32)
    )
    power = tla.max(power, tla.full(-127, tla.Int32))
    power = tla.min(power, tla.full(127, tla.Int32))
    power = tla.where(tla.cmp(maximum, 0.0, mode="eq"), tla.full(0, tla.Int32), power)
    inv = tla.shift_left(tla.full(127, tla.Int32) - power, 23).bitcast(tla.Float32)
    tla.tile_view(inverses, tla.make_shape(1, 64), tla.make_coord(group, 0)).store(inv)
    encoded = power + 127
    if tla.const_expr(group % 2 == 1):
        joined = tla.bitwise_or(previous, tla.shift_left(encoded, 8))
        words = joined.to(
            tla.Int16,
            tla.params.CastParams(
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_TRUNC,
                reg_slot=tla.params.RegSlot.ZERO,
            ),
        )
        packed, unused = tla.deinterleave(words, words)
        mask = tla.create_mask(pattern=tla.mask.VL64, dtype=tla.Int16)
        if tla.const_expr(group == 1):
            ph0.store(packed, tla.params.BlockStoreParams(block_stride=2), mask=mask)
        else:
            ph1.store(packed, tla.params.BlockStoreParams(block_stride=2), mask=mask)
    return encoded


@tla.jit
def online_quantize_probabilities(
    scores,
    mstate,
    lstate,
    astate,
    ph0,
    ph1,
    bad,
    inverses,
    blockm,
    raw_scale=1.0,
    raw_scores: tla.Constexpr = False,
    sanitize_nonfinite: tla.Constexpr = True,
):
    """Compute unquantized softmax statistics and MX scales in one tile pass."""
    zero = tla.full(0.0, tla.Float32)
    oldm, oldl = mstate.load(), lstate.load()
    newm = tla.max(oldm, blockm)
    safem = tla.where(tla.cmp(newm, float("-inf"), mode="eq"), zero, newm)
    alpha = tla.where(tla.cmp(oldl, 0.0, mode="gt"), tla.exp(oldm - safem), zero)
    sums = (zero,) * 8
    previous = tla.full(0, tla.Int32)
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    for group in tla.range_constexpr(4):
        maxima = (zero,) * 8
        for row_group in tla.range(4):
            for lane in tla.range_constexpr(8):
                row = group * 32 + row_group * 8 + lane
                item = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(row, 0))
                score = item.load()
                if tla.const_expr(raw_scores):
                    score = score * raw_scale
                p = tla.exp(score - safem)
                safe_p = p
                if tla.const_expr(sanitize_nonfinite):
                    safe_p = tla.where(tla.cmp(p, float("inf"), mode="lt"), p, zero)
                item.store(safe_p)
                sums = sums[:lane] + (sums[lane] + p,) + sums[lane + 1 :]
                maxima = maxima[:lane] + (tla.max(maxima[lane], safe_p),) + maxima[lane + 1 :]
        maximum = tla.max(
            tla.max(tla.max(maxima[0], maxima[1]), tla.max(maxima[2], maxima[3])),
            tla.max(tla.max(maxima[4], maxima[5]), tla.max(maxima[6], maxima[7])),
        )
        previous = _mx_group_scale(maximum, previous, group, inverses, ph0, ph1)
    blockl = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + (
        (sums[4] + sums[5]) + (sums[6] + sums[7])
    )
    bad.store(
        tla.where(tla.cmp(blockl, float("inf"), mode="lt"), bad.load(), tla.full(1.0, tla.Float32))
    )
    mstate.store(newm)
    lstate.store(oldl * alpha + blockl)
    astate.store(alpha)
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def scaled_group_max_readonly(scores, scale, groupmax, positive: tla.Constexpr = True):
    """Identity full-tile path: scale four group extrema after reduction."""
    neginf = tla.full(float("-inf"), tla.Float32)
    initial = neginf if tla.const_expr(positive) else tla.full(float("inf"), tla.Float32)
    blockm = neginf
    for group in tla.range_constexpr(4):
        maxima = (initial,) * 8
        for subgroup in tla.range(4):
            for lane in tla.range_constexpr(8):
                item = tla.tile_view(
                    scores,
                    tla.make_shape(1, 64),
                    tla.make_coord(group * 32 + subgroup * 8 + lane, 0),
                )
                z = item.load()
                value = (
                    tla.max(maxima[lane], z)
                    if tla.const_expr(positive)
                    else tla.min(maxima[lane], z)
                )
                maxima = maxima[:lane] + (value,) + maxima[lane + 1 :]
        maximum = neginf
        if tla.const_expr(positive):
            maximum = tla.max(
                tla.max(tla.max(maxima[0], maxima[1]), tla.max(maxima[2], maxima[3])),
                tla.max(tla.max(maxima[4], maxima[5]), tla.max(maxima[6], maxima[7])),
            )
        else:
            maximum = tla.min(
                tla.min(tla.min(maxima[0], maxima[1]), tla.min(maxima[2], maxima[3])),
                tla.min(tla.min(maxima[4], maxima[5]), tla.min(maxima[6], maxima[7])),
            )
        maximum = maximum * scale
        tla.tile_view(groupmax, tla.make_shape(1, 64), tla.make_coord(group, 0)).store(maximum)
        blockm = tla.max(blockm, maximum)
    return blockm


@tla.jit
def prepare_group_scales(mstate, lstate, astate, safemstate, ph0, ph1, inverses, blockm, groupmax):
    """Prepare exact E8M0 scales and online m/alpha; keep old l for the VF."""
    zero = tla.full(0.0, tla.Float32)
    oldm, oldl = mstate.load(), lstate.load()
    newm = tla.max(oldm, blockm)
    safem = tla.where(tla.cmp(newm, float("-inf"), mode="eq"), zero, newm)
    alpha = tla.where(tla.cmp(oldl, 0.0, mode="gt"), tla.exp(oldm - safem), zero)
    previous = tla.full(0, tla.Int32)
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    for group in tla.range_constexpr(4):
        score_max = tla.tile_view(groupmax, tla.make_shape(1, 64), tla.make_coord(group, 0)).load()
        maximum = tla.exp(score_max - safem)
        maximum = tla.where(tla.cmp(maximum, float("inf"), mode="lt"), maximum, zero)
        previous = _mx_group_scale(maximum, previous, group, inverses, ph0, ph1)
    mstate.store(newm)
    astate.store(alpha)
    safemstate.store(safem)
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


# RNE packing stays explicitly unrolled to preserve register scheduling.
_PACK_SOURCE = r"""#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;using namespace AscendC::Reg;
constexpr CastTrait rne0 = {RegLayout::ZERO, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne1 = {RegLayout::ONE, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne2 = {RegLayout::TWO, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne3 = {RegLayout::THREE, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
__simd_vf__ inline void flex_exp_pack_update(__ubuf__ float* src, __ubuf__ uint8_t* dst,
                                __ubuf__ float* inverses, __ubuf__ float* safemstate,
                                __ubuf__ float* lstate, __ubuf__ float* astate, __ubuf__ float* badstate, float scale) {
  MaskReg mf = CreateMask<float, MaskPattern::ALL>();
  MaskReg mb = CreateMask<uint8_t, MaskPattern::ALL>();
  RegTensor<int8_t> seq;
  RegTensor<uint8_t> loidx, hiidx, upper;
  Arange(seq, int8_t(0));
  ShiftLefts(loidx, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(3), mb);
  ShiftRights(loidx, loidx, int16_t(1), mb);
  ShiftLefts(upper, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(1), mb);
  ShiftRights(upper, upper, int16_t(6), mb);
  Add(loidx, loidx, upper, mb);
  ShiftRights(upper, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(7), mb);
  ShiftLefts(upper, upper, int16_t(7), mb);
  Or(loidx, loidx, upper, mb);
  RegTensor<float> sm, inv, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
  Duplicate(sum0, 0.0f, mf); Duplicate(sum1, 0.0f, mf);
  Duplicate(sum2, 0.0f, mf); Duplicate(sum3, 0.0f, mf);
  Duplicate(sum4, 0.0f, mf); Duplicate(sum5, 0.0f, mf);
  Duplicate(sum6, 0.0f, mf); Duplicate(sum7, 0.0f, mf);
  for (uint16_t row = 0; row < 128; row += 8) {
    LoadAlign(inv, inverses + (row / 32) * 64);
    {
      RegTensor<float> x0, x1, x2, x3;
      RegTensor<fp8_e4m3fn_t> p0, p1, p2, p3;
      RegTensor<uint8_t> packed, lo;
      LoadAlign(x0, src + (row + 0) * 64);
      Mul(x0, x0, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne0>(p0, x0, mf);
      LoadAlign(x1, src + (row + 1) * 64);
      Mul(x1, x1, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne1>(p1, x1, mf);
      LoadAlign(x2, src + (row + 2) * 64);
      Mul(x2, x2, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne2>(p2, x2, mf);
      LoadAlign(x3, src + (row + 3) * 64);
      Mul(x3, x3, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne3>(p3, x3, mf);
      Or(packed, reinterpret_cast<RegTensor<uint8_t>&>(p0), reinterpret_cast<RegTensor<uint8_t>&>(p1), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p2), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p3), mb);
      Gather(lo, packed, loidx);
      StoreAlign(dst + (row + 0) * 64, lo, mb);
    }
    {
      RegTensor<float> x0, x1, x2, x3;
      RegTensor<fp8_e4m3fn_t> p0, p1, p2, p3;
      RegTensor<uint8_t> packed, lo;
      LoadAlign(x0, src + (row + 4) * 64);
      Mul(x0, x0, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne0>(p0, x0, mf);
      LoadAlign(x1, src + (row + 5) * 64);
      Mul(x1, x1, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne1>(p1, x1, mf);
      LoadAlign(x2, src + (row + 6) * 64);
      Mul(x2, x2, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne2>(p2, x2, mf);
      LoadAlign(x3, src + (row + 7) * 64);
      Mul(x3, x3, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne3>(p3, x3, mf);
      Or(packed, reinterpret_cast<RegTensor<uint8_t>&>(p0), reinterpret_cast<RegTensor<uint8_t>&>(p1), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p2), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p3), mb);
      Gather(lo, packed, loidx);
      StoreAlign(dst + (row + 4) * 64, lo, mb);
    }
  }
}
extern "C" [aicore] __attribute__((always_inline))
void flex_mx_pack_scaled_rne(uint64_t src,uint64_t dst,uint64_t inverses) {
  flex_exp_pack_update((__ubuf__ float*)src,(__ubuf__ uint8_t*)dst,
      (__ubuf__ float*)inverses,nullptr,nullptr,nullptr,nullptr,1.f);
}
"""


@tla.extern(name="flex_mx_pack_scaled_rne", source=_PACK_SOURCE)
def pack_scaled_rne(
    src: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    dst: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    inverses: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
) -> None: ...


@tla.extern(
    name="mx_zero_ub",
    source=_MX_CPP_HEADER
    + r"""__simd_vf__ inline void zub(__ubuf__ uint8_t* p, uint32_t bytes) {
  RegTensor<uint8_t> z; Duplicate(z,uint8_t(0));
  MaskReg all=CreateMask<uint8_t,MaskPattern::ALL>();
  for(uint16_t i=0;i<bytes/256;++i) StoreAlign(p+i*256,z,all);
  uint32_t tail=bytes%256; MaskReg m=UpdateMask<uint8_t>(tail);
  StoreAlign(p+(bytes/256)*256,z,m);
}
extern "C" [aicore] __attribute__((always_inline)) void mx_zero_ub(uint64_t ptr,int32_t bytes) {
  zub(reinterpret_cast<__ubuf__ uint8_t*>(ptr),uint32_t(bytes));
}
""",
)
def mx_zero_ub(ptr: tla.Pointer[tla.UInt8, tla.AddressSpace.ub], bytes: tla.Int32) -> None: ...
@tla.jit
def mx_zero_scale_l1(dst, src, bytes: tla.Constexpr[int]):
    # One contiguous DMA; FP16 is only a 16-bit storage view of E8M0 bytes.
    layout = tla.make_layout_with_tag((bytes // 32, 16), tla.Float16, tla.arch.zN)
    target = tla.make_tensor(tla.recast_ptr(dst, dtype=tla.Float16), layout)
    source = tla.make_tensor(tla.recast_ptr(src, dtype=tla.Float16), layout)
    tla.copy(target, source)


@tla.jit
def mx_prob64_to_l1(dst, src):
    # Two 32-repeat DMAs, each copying four 32-byte blocks. The UB pitch
    # is eight blocks and the L1 pitch is four; no numerical conversion.
    source_layout = tla.make_layout(
        tla.make_shape((4, 1), (16, 32)),
        tla.make_stride((16, 64), (1, 128)),
        origin_shape=tla.make_shape(4, 512),
        layoutTag=tla.arch.zNUnAlign,
    )
    target_layout = tla.make_layout(
        tla.make_shape((16, 1), (16, 32)),
        tla.make_stride((16, 256), (1, 64)),
        origin_shape=tla.make_shape(4, 512),
        layoutTag=tla.arch.zN,
    )
    d = tla.recast_ptr(dst, dtype=tla.Float16)
    s = tla.recast_ptr(src, dtype=tla.Float16)
    tla.copy(tla.make_tensor(d, target_layout), tla.make_tensor(s, source_layout))
    tla.copy(tla.make_tensor(d + 2048, target_layout), tla.make_tensor(s + 64, source_layout))


@tla.extern(
    name="mx_score_unit",
    source=_MX_CPP_HEADER
    + r"""extern "C" [aicore] __attribute__((always_inline)) void mx_score_unit(uint64_t dst, uint64_t src, float scale) {
  AscendC::LocalTensor<half> d(AscendC::TPosition::VECCALC, uint32_t(dst), 8192);
  AscendC::LocalTensor<float> s(AscendC::TPosition::CO1, uint32_t(src), 16384);
  AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> p;
  p.nSize=64; p.mSize=128; p.srcStride=128; p.dstStride=64;
  p.quantPre=QuantMode_t::QF322F16_PRE; p.reluEn=false; p.unitFlag=3; p.dualDstCtl=0; p.subBlockId=false;
  union {float f; uint32_t u;} bits;
  bits.f=scale; p.deqScalar=bits.u;
  constexpr static AscendC::FixpipeConfig cfg={AscendC::CO2Layout::ROW_MAJOR, true};
  AscendC::Fixpipe<half, float, cfg>(d,s,p);
  p.subBlockId=true;
  AscendC::Fixpipe<half, float, cfg>(d,s[8192],p);
}
""",
)
def mx_score_unit(
    dst: tla.Pointer[tla.Float16, tla.AddressSpace.ub],
    src: tla.Pointer[tla.Float32, tla.AddressSpace.l0c],
    scale: tla.Float32,
) -> None: ...


@tla.extern(
    name="mx_pad",
    source=_MX_CPP_HEADER
    + r"""__simd_vf__ inline void mx_pad_vf(__ubuf__ uint8_t* v, __ubuf__ uint8_t* s) {
  using namespace AscendC::Reg;
  RegTensor<uint8_t> value;
  MaskReg all=CreateMask<uint8_t, MaskPattern::ALL>();
  Duplicate(value,uint8_t(0x38));
  for(uint16_t i=0;i<8;++i) StoreAlign(v+i*256,value,all);
  Duplicate(value,uint8_t(127));
  uint32_t n=64; MaskReg m=UpdateMask<uint8_t>(n);
  StoreAlign(s,value,m);
}
extern "C" [aicore] __attribute__((always_inline)) void mx_pad(uint64_t v, uint64_t s) {
  mx_pad_vf(reinterpret_cast<__ubuf__ uint8_t*>(v),reinterpret_cast<__ubuf__ uint8_t*>(s));
}
""",
)
def mx_pad(
    v: tla.Pointer[tla.UInt8, tla.AddressSpace.ub], s: tla.Pointer[tla.UInt8, tla.AddressSpace.ub]
) -> None: ...


@tla.jit
def _mx_vector_tile(ptr, lanes: tla.Constexpr[int]):
    return tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(lanes), tla.make_stride(1), layoutTag=tla.arch.VectorLayout),
    )


# Fast tiles use exp(FP16(score) + 1.9462890625) with Pscale=1/8.
# Device saturation, score-range and denominator guards request FP32 repair.
# Periodic masks additionally reject tiny total mass; interval/auxiliary masks
# check mass per selected block. Empty rows remain zero with LSE=-inf.
_CPP_SHORT_BODY = (
    _MX_CPP_HEADER
    + r"""constexpr CastTrait mx_short_cast0={RegLayout::ZERO,SatMode::SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait mx_short_cast1={RegLayout::ONE,SatMode::SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait mx_short_cast2={RegLayout::TWO,SatMode::SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait mx_short_cast3={RegLayout::THREE,SatMode::SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
__simd_callee__ inline void mx_short_pack(RegTensor<float>& x0,RegTensor<float>& x1,
 RegTensor<float>& x2,RegTensor<float>& x3,RegTensor<uint8_t>& packed,MaskReg& mf) {
 RegTensor<fp8_e4m3fn_t> p0,p1,p2,p3;
 Cast<fp8_e4m3fn_t,float,mx_short_cast0>(p0,x0,mf);
 Cast<fp8_e4m3fn_t,float,mx_short_cast1>(p1,x1,mf);
 Cast<fp8_e4m3fn_t,float,mx_short_cast2>(p2,x2,mf);
 Cast<fp8_e4m3fn_t,float,mx_short_cast3>(p3,x3,mf);
 MaskReg mb=CreateMask<uint8_t,MaskPattern::ALL>();
 Or(packed,reinterpret_cast<RegTensor<uint8_t>&>(p0),reinterpret_cast<RegTensor<uint8_t>&>(p1),mb);
 Or(packed,packed,reinterpret_cast<RegTensor<uint8_t>&>(p2),mb);
 Or(packed,packed,reinterpret_cast<RegTensor<uint8_t>&>(p3),mb);
}
// Both stored masks and fused callbacks provide the same vector policy.
struct MxStoredMask {
 __simd_callee__ static inline void init(RegTensor<int32_t>& query,int32_t qbase,MaskReg& all) {}
 __simd_callee__ static inline MaskReg load(RegTensor<int32_t>& query,__ubuf__ uint32_t* visibility,
     int32_t row,int32_t key,int32_t qlen,int32_t klen,MaskReg& all) {
  MaskReg result;DataCopy<uint32_t>(result,visibility+row*8);return result;
 }
};
template<bool FULL,bool MASKED=false,bool ADAPTIVE=false,bool RANGE=false,
 bool SUM_DENOMINATOR=true,bool CHECK_MINIMUM=true,class Mask=MxStoredMask>
__simd_vf__ inline void mx_short_quantize_vf(__ubuf__ half* s,__ubuf__ uint8_t* p,__ubuf__ float* maximum,__ubuf__ float* denominator,int32_t tile,int32_t cols,__ubuf__ uint32_t* visibility=nullptr,__ubuf__ float* offset=nullptr,int32_t first=0,int32_t last=128,int32_t qbase=0,int32_t kbase=0,int32_t qlen=0,int32_t klen=0) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),mh=CreateMask<half,MaskPattern::ALL>(),mb=CreateMask<uint8_t,MaskPattern::ALL>();
 RegTensor<int32_t> query;Mask::init(query,qbase,mf);
 RegTensor<int8_t> seq;RegTensor<uint8_t> ix,upper;Arange(seq,int8_t(0));
 ShiftLefts(ix,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(3),mb);ShiftRights(ix,ix,int16_t(1),mb);
 ShiftLefts(upper,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(1),mb);ShiftRights(upper,upper,int16_t(6),mb);Add(ix,ix,upper,mb);
 ShiftRights(upper,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(7),mb);ShiftLefts(upper,upper,int16_t(7),mb);Or(ix,ix,upper,mb);
 RegTensor<float> den,tmp,den1,den2,den3;Duplicate(den,0.f);Duplicate(den1,0.f);Duplicate(den2,0.f);Duplicate(den3,0.f);
 RegTensor<half> shift;Duplicate(shift,half(-1.9462890625f));
 if constexpr (ADAPTIVE) {
  RegTensor<half> m0,m1,m2,m3,negative;RegTensor<float> maximum,zero;
  Duplicate(negative,half(-__builtin_inff()));Duplicate(zero,0.f);
  m0=negative;m1=negative;m2=negative;m3=negative;
  for(uint16_t row=RANGE?first:0;row<(RANGE?last:128);row+=4) {
   RegTensor<half> x0,x1,x2,x3;
   LoadAlign<half,LoadDist::DIST_UNPACK_B16>(x0,s+(row+0)*64);
   LoadAlign<half,LoadDist::DIST_UNPACK_B16>(x1,s+(row+1)*64);
   LoadAlign<half,LoadDist::DIST_UNPACK_B16>(x2,s+(row+2)*64);
   LoadAlign<half,LoadDist::DIST_UNPACK_B16>(x3,s+(row+3)*64);
   if constexpr(MASKED) {
    MaskReg v0,v1,v2,v3;
    v0=Mask::load(query,visibility,row+0,kbase+row+0,qlen,klen,mf);v1=Mask::load(query,visibility,row+1,kbase+row+1,qlen,klen,mf);
    v2=Mask::load(query,visibility,row+2,kbase+row+2,qlen,klen,mf);v3=Mask::load(query,visibility,row+3,kbase+row+3,qlen,klen,mf);
    Select(x0,x0,negative,v0);Select(x1,x1,negative,v1);Select(x2,x2,negative,v2);Select(x3,x3,negative,v3);
   }
   if constexpr(!FULL) {
    if(row>=cols) x0=negative;if(row+1>=cols) x1=negative;
    if(row+2>=cols) x2=negative;if(row+3>=cols) x3=negative;
   }
   Max(m0,m0,x0,mh);Max(m1,m1,x1,mh);Max(m2,m2,x2,mh);Max(m3,m3,x3,mh);
  }
  Max(m0,m0,m1,mh);Max(m2,m2,m3,mh);Max(shift,m0,m2,mh);Cast<float,half,mx_short_cast0>(maximum,shift,mf);RegTensor<float> lse_offset;Adds(lse_offset,maximum,-3.4657359027997265f,mf);StoreAlign(offset,lse_offset,mf);
  MaskReg empty;Compares<float,CMPMODE::EQ>(empty,maximum,-__builtin_inff(),mf);
  Select(maximum,zero,maximum,empty);Cast<half,float,mx_short_cast0>(shift,maximum,mf);
 }
 RegTensor<half> lowest,low01,low23;Duplicate(lowest,half(__builtin_inff()));
 RegTensor<uint8_t> largest;if(tile==0) Duplicate(largest,uint8_t(0));else LoadAlign(largest,(__ubuf__ uint8_t*)maximum);
 if constexpr(RANGE) {
  RegTensor<uint8_t> zeros;Duplicate(zeros,uint8_t(0));
  for(uint16_t pos=0;pos<first*64;pos+=256) StoreAlign(p+pos,zeros,mb);
  for(uint16_t pos=last*64;pos<8192;pos+=256) StoreAlign(p+pos,zeros,mb);
 }
 #pragma unroll 4
 for(uint16_t row=RANGE?first:0;row<(RANGE?last:128);row+=4) {
  RegTensor<half> s0,s1,s2,s3;RegTensor<int16_t> a,b;
  LoadAlign<half,LoadDist::DIST_UNPACK_B16>(s0,s+row*64);LoadAlign<half,LoadDist::DIST_UNPACK_B16>(s1,s+(row+1)*64);
  LoadAlign<half,LoadDist::DIST_UNPACK_B16>(s2,s+(row+2)*64);LoadAlign<half,LoadDist::DIST_UNPACK_B16>(s3,s+(row+3)*64);
  RegTensor<float> x0,x1,x2,x3;
  MaskReg v0=mh,v1=mh,v2=mh,v3=mh;
  if constexpr (MASKED) {
    v0=Mask::load(query,visibility,row+0,kbase+row+0,qlen,klen,mf);v1=Mask::load(query,visibility,row+1,kbase+row+1,qlen,klen,mf);
    v2=Mask::load(query,visibility,row+2,kbase+row+2,qlen,klen,mf);v3=Mask::load(query,visibility,row+3,kbase+row+3,qlen,klen,mf);
  }
  FusedExpSub<float,half,RegLayout::ZERO,MaskMergeMode::ZEROING>(x0,s0,shift,v0);
  FusedExpSub<float,half,RegLayout::ZERO,MaskMergeMode::ZEROING>(x1,s1,shift,v1);
  FusedExpSub<float,half,RegLayout::ZERO,MaskMergeMode::ZEROING>(x2,s2,shift,v2);
  FusedExpSub<float,half,RegLayout::ZERO,MaskMergeMode::ZEROING>(x3,s3,shift,v3);
  if constexpr(!ADAPTIVE && CHECK_MINIMUM && (MASKED || !SUM_DENOMINATOR)) {
   Min(low01,s0,s1,mh);Min(low23,s2,s3,mh);
   Min(low01,low01,low23,mh);Min(lowest,lowest,low01,mh);
  }
  if constexpr (!FULL) {
    if(row>=cols) Duplicate(x0,0.f); if(row+1>=cols) Duplicate(x1,0.f);
    if(row+2>=cols) Duplicate(x2,0.f); if(row+3>=cols) Duplicate(x3,0.f);
  }
  if constexpr(SUM_DENOMINATOR) {Add(den,den,x0,mf);Add(den1,den1,x1,mf);Add(den2,den2,x2,mf);Add(den3,den3,x3,mf);}
  RegTensor<uint8_t> packed;
  if constexpr(ADAPTIVE) {
   Muls(x0,x0,256.f,mf);Muls(x1,x1,256.f,mf);Muls(x2,x2,256.f,mf);Muls(x3,x3,256.f,mf);
  }
  mx_short_pack(x0,x1,x2,x3,packed,mf);
  Max(largest,largest,packed,mb);
  Gather(packed,packed,ix);StoreAlign(p+row*64,packed,mb);
 }
 RegTensor<uint32_t> unsafe;Duplicate(unsafe,uint32_t(0x7f7f7f7f));
 if constexpr(SUM_DENOMINATOR) {
  Add(den,den,den1,mf);Add(den2,den2,den3,mf);Add(den,den,den2,mf);
  if constexpr(ADAPTIVE) Muls(den,den,32.f,mf);else Muls(den,den,0.125f,mf);
  if constexpr(!MASKED && !ADAPTIVE) {
   // At most 1/8192 absolute rounding error per unmasked FP8 value.
   // Certify each tile before accumulating its denominator, including tails.
   MaskReg safe;Compares<float,CMPMODE::GE>(safe,den,float(RANGE?last-first:(FULL?128:cols))*0.125f,mf);
   Select(reinterpret_cast<RegTensor<uint32_t>&>(largest),reinterpret_cast<RegTensor<uint32_t>&>(largest),unsafe,safe);
  }
  if(tile!=0) {LoadAlign(tmp,denominator);Add(den,den,tmp,mf);}
  StoreAlign(denominator,den,mf);
 }
 if constexpr(!ADAPTIVE && CHECK_MINIMUM && (MASKED || !SUM_DENOMINATOR)) {
  // Arbitrary masks need no visible-lane count: certify that their possible
  // FP8 probabilities are normal. Testing masked scores is conservative.
  RegTensor<half> threshold;Adds(threshold,shift,half(-4.f),mh);
  RegTensor<float> low32,threshold32;
  Cast<float,half,mx_short_cast0>(low32,lowest,mf);
  Cast<float,half,mx_short_cast0>(threshold32,threshold,mf);
  MaskReg subnormal;Compare<float,CMPMODE::LT>(subnormal,low32,threshold32,mf);
  Select(reinterpret_cast<RegTensor<uint32_t>&>(largest),unsafe,reinterpret_cast<RegTensor<uint32_t>&>(largest),subnormal);
 }
 StoreAlign((__ubuf__ uint8_t*)maximum,largest,mb);
 if constexpr(!ADAPTIVE) {
  if(offset!=nullptr) {RegTensor<float> c;Duplicate(c,0.13315247917983575f);StoreAlign(offset,c,mf);}
 }
}
"""
)


_CPP_SHORT_QUANTIZE = (
    _CPP_SHORT_BODY
    + r"""
template<bool ADAPTIVE,bool SUM_DENOMINATOR>
[aicore] __attribute__((always_inline)) void mx_short_dispatch(
 uint64_t s,uint64_t p,uint64_t m,uint64_t d,int32_t tile,int32_t cols,uint64_t visibility,__ubuf__ float* offset,int32_t masked) {
#define CALL(F,M) mx_short_quantize_vf<F,M,ADAPTIVE,false,SUM_DENOMINATOR>((__ubuf__ half*)s,(__ubuf__ uint8_t*)p,(__ubuf__ float*)m,(__ubuf__ float*)d,tile,cols,(__ubuf__ uint32_t*)visibility,offset)
 if(masked) {if(cols==128) CALL(true,true);else CALL(false,true);}
 else {if(cols==128) CALL(true,false);else CALL(false,false);}
#undef CALL
}
template<bool SUM_DENOMINATOR>
[aicore] __attribute__((always_inline)) void mx_short_run(
 uint64_t s,uint64_t p,uint64_t m,uint64_t d,int32_t tile,int32_t cols,uint64_t visibility,uint64_t offset,int32_t masked,int32_t adaptive,int32_t first,int32_t last) {
 __ubuf__ float* offsetp=adaptive<0?nullptr:(__ubuf__ float*)offset;
 if(first>=0) mx_short_quantize_vf<true,false,false,true,SUM_DENOMINATOR>((__ubuf__ half*)s,(__ubuf__ uint8_t*)p,(__ubuf__ float*)m,(__ubuf__ float*)d,0,128,nullptr,offsetp,first,last);
 else if(adaptive>0) mx_short_dispatch<true,SUM_DENOMINATOR>(s,p,m,d,tile,cols,visibility,offsetp,masked);
 else mx_short_dispatch<false,SUM_DENOMINATOR>(s,p,m,d,tile,cols,visibility,offsetp,masked);
}
extern "C" [aicore] __attribute__((always_inline)) void mx_short_quantize_dispatch(
 uint64_t s,uint64_t p,uint64_t m,uint64_t d,int32_t tile,int32_t cols,uint64_t visibility,uint64_t offset,int32_t masked,int32_t adaptive,int32_t first,int32_t last) {
 mx_short_run<true>(s,p,m,d,tile,cols,visibility,offset,masked,adaptive,first,last);
}
extern "C" [aicore] __attribute__((always_inline)) void mx_short_quantize_post_dispatch(
 uint64_t s,uint64_t p,uint64_t m,uint64_t d,int32_t tile,int32_t cols,uint64_t visibility,uint64_t offset,int32_t masked,int32_t adaptive,int32_t first,int32_t last) {
 mx_short_run<false>(s,p,m,d,tile,cols,visibility,offset,masked,adaptive,first,last);
}
"""
)

_MX_SHORT_QUANTIZERS = {}


def _mx_short_quantizer(post_quant):
    if post_quant in _MX_SHORT_QUANTIZERS:
        return _MX_SHORT_QUANTIZERS[post_quant]
    name = "mx_short_quantize_post_dispatch" if post_quant else "mx_short_quantize_dispatch"

    @tla.extern(name=name, source=_CPP_SHORT_QUANTIZE)
    def pack(
        s: tla.Pointer[tla.Float16, tla.AddressSpace.ub],
        p: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
        m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
        d: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
        tile: tla.Int32,
        cols: tla.Int32,
        visibility: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
        offset: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
        masked: tla.Int32,
        adaptive: tla.Int32,
        first: tla.Int32,
        last: tla.Int32,
    ) -> None: ...

    _MX_SHORT_QUANTIZERS[post_quant] = pack
    return pack


@tla.jit
def _mx_short_quantize(
    s,
    p,
    m,
    d,
    tile,
    cols,
    visibility,
    offset,
    masked,
    adaptive,
    first,
    last,
    post_quant: tla.Constexpr = False,
):
    pack = _mx_short_quantizer(post_quant)
    pack(s, p, m, d, tile, cols, visibility, offset, masked, adaptive, first, last)


@tla.extern(
    name="mx_key_aux_quantize",
    source=_CPP_SHORT_BODY
    + r"""
struct MxKeyAuxMask {
 __simd_callee__ static inline void init(RegTensor<int32_t>& q,int32_t base,MaskReg& all) {}
 __simd_callee__ static inline MaskReg load(RegTensor<int32_t>& q,__ubuf__ uint32_t* aux,
     int32_t row,int32_t key,int32_t qlen,int32_t klen,MaskReg& all) {
  RegTensor<int32_t> position,threshold;Duplicate(position,key);
  LoadAlign<int32_t,LoadDist::DIST_BRC_B32>(threshold,(__ubuf__ int32_t*)aux+row);
  MaskReg visible,tail;Compare<int32_t,CMPMODE::GE>(visible,position,threshold,all);
  Compares<int32_t,CMPMODE::LT>(tail,position,klen,all);And(visible,visible,tail,all);
  return visible;
 }
};
extern "C" [aicore] __attribute__((always_inline)) void mx_key_aux_quantize(
 uint64_t s,uint64_t p,uint64_t m,uint64_t d,int32_t tile,int32_t cols,uint64_t aux,
 uint64_t offset,int32_t single,int32_t kbase,int32_t klen) {
#define CALL(F,A) mx_short_quantize_vf<F,true,A,false,true,true,MxKeyAuxMask>((__ubuf__ half*)s,(__ubuf__ uint8_t*)p,(__ubuf__ float*)m,(__ubuf__ float*)d,tile,cols,(__ubuf__ uint32_t*)aux,single?(__ubuf__ float*)offset:nullptr,0,128,0,kbase,0,klen)
 if(single) {if(cols==128) CALL(true,true);else CALL(false,true);}
 else {if(cols==128) CALL(true,false);else CALL(false,false);}
#undef CALL
}
""",
)
def _mx_quantize_key_aux(
    s: tla.Pointer[tla.Float16, tla.AddressSpace.ub],
    p: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    d: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    tile: tla.Int32,
    cols: tla.Int32,
    aux: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    offset: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    single: tla.Int32,
    kbase: tla.Int32,
    klen: tla.Int32,
) -> None: ...


@tla.extern(
    name="mx_mask_probe",
    source=_MX_CPP_HEADER
    + r"""__simd_vf__ inline void mx_mask_probe_vf(__ubuf__ uint32_t* visibility,__ubuf__ uint32_t* result) {
 MaskReg all=CreateMask<uint32_t,MaskPattern::ALL>();
 RegTensor<uint32_t> a,b,c,d;
 LoadAlign(a,visibility);LoadAlign(b,visibility+64);LoadAlign(c,visibility+128);LoadAlign(d,visibility+192);
 Min(a,a,b,all);Min(c,c,d,all);Min(a,a,c,all);ReduceMin(a,a,all);StoreAlign(result,a,all);
}
extern "C" [aicore] __attribute__((always_inline)) void mx_mask_probe(uint64_t visibility,uint64_t result) {mx_mask_probe_vf((__ubuf__ uint32_t*)visibility,(__ubuf__ uint32_t*)result);}
""",
)
def mask_has_32_visible(
    visibility: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    result: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
) -> None: ...
@tla.extern(
    name="mx_short_statistics",
    source=_MX_CPP_HEADER
    + r"""template<typename OUT,bool ADAPTIVE=false>
__simd_vf__ inline void mx_short_statistics_vf(__ubuf__ float* acc,__ubuf__ float* maximum,
    __ubuf__ OUT* out,__ubuf__ float* lse,int32_t dim,__ubuf__ int32_t* guard,__ubuf__ int32_t* aggregate,int32_t count,__ubuf__ float* denominator,__ubuf__ float* offset,int32_t rows,int32_t bounded) {
  uint32_t active=uint32_t(rows),active_bytes=uint32_t(rows*4);
  MaskReg af=UpdateMask<float>(active),all=CreateMask<float,MaskPattern::ALL>();
  MaskReg ah=CreateMask<uint16_t,MaskPattern::ALL>();
  RegTensor<float> den,inv,one,m,l;
  LoadAlign(den,denominator);
  MaskReg ab=UpdateMask<uint8_t>(active_bytes),bad,good;
  RegTensor<uint8_t> bytes,badbytes,bytezero,byteone;
  LoadAlign(bytes,(__ubuf__ uint8_t*)maximum);
  Duplicate(bytezero,uint8_t(0));Duplicate(byteone,uint8_t(1));
  Compares<uint8_t,CMPMODE::GT>(bad,bytes,uint8_t(125),ab);
  Select(badbytes,byteone,bytezero,bad);
  RegTensor<uint32_t> lowden,zero,ones,result;
  Duplicate(zero,uint32_t(0));Duplicate(ones,uint32_t(1));
  if(bounded) {
   if constexpr(ADAPTIVE) Compares<float,CMPMODE::GE>(good,den,32.f,af);
   else Compares<float,CMPMODE::GE>(good,den,bounded==2?16.f:float(count)*16.f,af);
  } else Compares<float,CMPMODE::GT>(good,den,0.f,af);
  Select(lowden,zero,ones,good);
  Or(result,lowden,reinterpret_cast<RegTensor<uint32_t>&>(badbytes),af);
  ReduceMax(result,result,af);if(count==0 || rows==0) Duplicate(result,uint32_t(0));
  StoreAlign((__ubuf__ uint32_t*)guard,result,all);
  RegTensor<uint32_t> oldflag;LoadAlign(oldflag,(__ubuf__ uint32_t*)aggregate);
  Or(result,result,oldflag,all);StoreAlign((__ubuf__ uint32_t*)aggregate,result,all);
  constexpr static LnSpecificMode logarithm={MaskMergeMode::ZEROING,LnAlgo::PRECISION_1ULP_FTZ_TRUE};
  constexpr static DivSpecificMode divide={MaskMergeMode::ZEROING,false,DivAlgo::PRECISION_1ULP_FTZ_TRUE};
  Ln<float,&logarithm>(l,den,af);if constexpr(ADAPTIVE) {LoadAlign(m,offset);Add(l,l,m,af);}else Adds(l,l,0.13315247917983575f,af);StoreAlign(lse,l,af);
  Maxs(den,den,1e-30f,af);Duplicate(one,1.f);Div<float,&divide>(inv,one,den,af);
  RegTensor<uint32_t> inverse0,inverse1,zeros;
  Duplicate(zeros,uint32_t(0));
  Interleave(inverse0,inverse1,reinterpret_cast<RegTensor<uint32_t>&>(inv),zeros);
  StoreAlign((__ubuf__ uint32_t*)out,inverse0,all);
  StoreAlign((__ubuf__ uint32_t*)out+64,inverse1,all);
}
extern "C" [aicore] __attribute__((always_inline)) void mx_short_statistics(uint64_t a,uint64_t m,uint64_t y,uint64_t e,uint64_t g,uint64_t aggregate,int32_t count,uint64_t denominator,uint64_t offset,int32_t rows,int32_t bounded) {
 if(count==1) mx_short_statistics_vf<bfloat16_t,true>((__ubuf__ float*)a,(__ubuf__ float*)m,(__ubuf__ bfloat16_t*)y,(__ubuf__ float*)e,128,(__ubuf__ int32_t*)g,(__ubuf__ int32_t*)aggregate,count,(__ubuf__ float*)denominator,(__ubuf__ float*)offset,rows,bounded);
 else mx_short_statistics_vf<bfloat16_t>((__ubuf__ float*)a,(__ubuf__ float*)m,(__ubuf__ bfloat16_t*)y,(__ubuf__ float*)e,128,(__ubuf__ int32_t*)g,(__ubuf__ int32_t*)aggregate,count,(__ubuf__ float*)denominator,(__ubuf__ float*)offset,rows,bounded);
}
""",
)
def short_statistics(
    a: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    y: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    e: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    g: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    agg: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    count: tla.Int32,
    d: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    offset: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    rows: tla.Int32,
    bounded: tla.Int32,
) -> None: ...


@tla.jit
def initialize_short_pipeline(aggregate, scales):
    target = tla.make_tensor(aggregate, tla.make_layout(tla.make_shape(64), tla.make_stride(1)))
    ps = tla.make_tensor(
        tla.recast_ptr(scales, dtype=tla.Int8),
        tla.make_layout(tla.make_shape(512), tla.make_stride(1)),
    )
    with tla.vec.func():
        target.store(tla.full(0, tla.Int32))
        for seed in tla.range_constexpr(2):
            tla.tile_view(ps, tla.make_shape(256), tla.make_coord(seed)).store(
                tla.full(124, tla.Int8)
            )


@tla.extern(
    name="flex_mx_fill_scales",
    source=_MX_CPP_HEADER
    + r"""extern "C" [aicore] __attribute__((always_inline)) void flex_mx_fill_scales(uint64_t dst,int32_t bytes,int32_t value) {
 AscendC::LocalTensor<uint16_t> d(AscendC::TPosition::A1,uint32_t(dst),bytes/2);
 AscendC::InitConstValueParams<uint16_t> p;
 p.repeatTimes=1;p.blockNum=bytes/32;p.dstGap=0;p.initValue=uint16_t(value);
 AscendC::Fill(d,p);
}
""",
)
def _mx_fill_scales(
    dst: tla.Pointer[tla.UInt8, tla.AddressSpace.l1], bytes: tla.Int32, value: tla.Int32
) -> None: ...
# L1 Fill and ND/MX DMA share MTE2 but may overlap within the pipe.
# Complete the zero fill before copying valid tail elements into that region.
@tla.extern(
    name="flex_mx_fill_ready",
    source=_MX_CPP_HEADER
    + r"""extern "C" [aicore] __attribute__((always_inline)) void flex_mx_fill_ready() { AscendC::PipeBarrier<PIPE_MTE2>(); }
""",
)
def _mx_fill_ready() -> None: ...
@tla.jit
def initialize_short_state(aggregate):
    target = tla.make_tensor(aggregate, tla.make_layout(tla.make_shape(64), tla.make_stride(1)))
    with tla.vec.func():
        target.store(tla.full(0, tla.Int32))
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def _mx_scale64_offset(cu_k, sequence, start=0, current=0):
    offset = tla.Int32(current)
    for previous in tla.range(start, sequence):
        offset += (cu_k[previous + 1] - cu_k[previous] + 63) // 64
    return offset


@tla.jit
def _mx_advance_task(
    linear,
    cu_q,
    cu_k,
    heads,
    blocks,
    sequence,
    prefix,
    scale_prefix,
    q_packed: tla.Constexpr,
    k_packed: tla.Constexpr,
    q_major: tla.Constexpr,
    single_query_length=None,
):
    seq = tla.Int32(sequence)
    query_prefix = tla.Int32(prefix)
    vprefix = tla.Int32(scale_prefix)
    # Every producer/consumer cursor advances monotonically across sequences.
    # Walk each boundary once per core instead of rescanning the whole batch
    # for every query tile. Zero-length segments are consumed by the loop.
    task = linear
    if tla.const_expr(single_query_length is not None):
        # A single packed segment is [0, T] by the public metadata contract.
        # Retain the sparse capacity stride when max_seqlen is an upper bound.
        count = (single_query_length + 127) // 128 if tla.const_expr(q_packed) else blocks
        if tla.const_expr(q_major):
            task = (linear % heads) * blocks + linear // heads
        else:
            task = (linear // count) * blocks + linear % count
    elif tla.const_expr(q_packed):
        ordinal = linear // heads
        count = (cu_q[seq + 1] - cu_q[seq] + 127) // 128
        while ordinal >= query_prefix + count:
            query_prefix += count
            if tla.const_expr(k_packed):
                vprefix += (cu_k[seq + 1] - cu_k[seq] + 63) // 64
            seq += 1
            count = (cu_q[seq + 1] - cu_q[seq] + 127) // 128
        if tla.const_expr(q_major):
            task = (seq * heads + linear % heads) * blocks + ordinal - query_prefix
        else:
            local = linear - query_prefix * heads
            task = (seq * heads + local // count) * blocks + local % count
    else:
        if tla.const_expr(q_major):
            task = (
                (linear // (heads * blocks)) * (heads * blocks)
                + (linear % heads) * blocks
                + (linear // heads) % blocks
            )
        next_sequence = task // (heads * blocks)
        if tla.const_expr(k_packed):
            vprefix = _mx_scale64_offset(cu_k, next_sequence, seq, vprefix)
        seq = next_sequence
    return task, seq, query_prefix, vprefix


# Fast and exact phases share task ownership, drained channels and storage.
@tla.extern(
    name="mx_output_normalized",
    source=_MX_CPP_HEADER
    + r"""extern "C" [aicore] __attribute__((always_inline)) void mx_output_normalized(uint64_t dst,uint64_t src,uint64_t inverse,int32_t rows,int32_t pitch,int32_t streaming,int32_t unit_sync,int32_t stride) {
 AscendC::GlobalTensor<bfloat16_t> d;d.SetGlobalBuffer((__gm__ bfloat16_t*)dst);if(streaming) d.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
 AscendC::LocalTensor<float> s(AscendC::TPosition::CO1,uint32_t(src),128*stride);
 AscendC::LocalTensor<uint64_t> q(AscendC::TPosition::A1,uint32_t(inverse),128);
 constexpr static AscendC::FixpipeConfig cfg={AscendC::CO2Layout::COLUMN_MAJOR,false};
 AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> p;
 p.nSize=rows;p.mSize=128;p.srcStride=stride;p.dstStride=pitch;p.params.srcNzC0Stride=1;
 p.quantPre=QuantMode_t::VQF322BF16_PRE;p.reluEn=false;p.unitFlag=unit_sync;
 AscendC::Fixpipe<bfloat16_t,float,cfg>(d,s,q,p);
}
""",
)
def mx_output_normalized(
    dst: tla.Pointer[tla.BFloat16, tla.AddressSpace.gm],
    src: tla.Pointer[tla.Float32, tla.AddressSpace.l0c],
    inverse: tla.Pointer[tla.UInt8, tla.AddressSpace.l1],
    rows: tla.Int32,
    pitch: tla.Int32,
    streaming: tla.Int32,
    unit_sync: tla.Int32,
    stride: tla.Int32,
) -> None: ...
@tla.jit
def _mx_cache_packed_pair(
    mask_mod: tla.Constexpr,
    b,
    head,
    qstart,
    sub,
    k0,
    k1,
    mask0,
    mask1,
    aux_tensors,
    aux_scalars,
    info,
):
    words0 = tla.make_tensor(
        tla.recast_ptr(mask0.ptr, dtype=tla.Int16),
        tla.make_layout(tla.make_shape(128, 16), tla.make_stride(16, 1)),
    )
    words1 = tla.make_tensor(
        tla.recast_ptr(mask1.ptr, dtype=tla.Int16),
        tla.make_layout(tla.make_shape(128, 16), tla.make_stride(16, 1)),
    )
    with tla.vec.func(mode="simt", thread_block_dim=128):
        tid, _, _ = tla.arch.thread_idx()
        group = tid & 15
        chunk = (tid >> 4) & 3
        second = (tid & 64) != 0
        start = k0
        if second:
            start = k1
        qbase = (qstart & 0x7FFFFF80) | ((sub & 1) << 6)
        kbase = (start & 0x7FFFFF80) | (chunk << 5)
        planes = ()
        for lane in tla.range_constexpr(4):
            query = qbase | (group << 2) | lane
            plane = tla.Int32(0)
            for key_lane in tla.range_constexpr(32):
                if mask_mod(
                    tla.Float32(0.0),
                    b,
                    head,
                    query,
                    kbase | key_lane,
                    aux_tensors,
                    aux_scalars,
                    info,
                    True,
                ):
                    plane = plane | (tla.Int32(1) << key_lane)
            planes = planes + (plane,)
        for pair in tla.range_constexpr(16):
            word = tla.Int32(0)
            for lane in tla.range_constexpr(4):
                bits = (planes[lane] >> (pair * 2)) & 3
                bits = (bits | (bits << 15)) & 0x00010001
                word = word | (bits << (lane * 4))
            if second:
                words1[(chunk << 5) | (pair * 2), group] = word.to(tla.Int16)
                words1[(chunk << 5) | (pair * 2 + 1), group] = (word >> 16).to(tla.Int16)
            else:
                words0[(chunk << 5) | (pair * 2), group] = word.to(tla.Int16)
                words0[(chunk << 5) | (pair * 2 + 1), group] = (word >> 16).to(tla.Int16)


class _MxComparable:
    def __bool__(self):
        raise NotImplementedError("data-dependent control flow requires SIMT")

    def __eq__(self, other):
        return self.compare(other, "EQ")

    def __ne__(self, other):
        return self.compare(other, "NE")

    def __lt__(self, other):
        return self.compare(other, "LT")

    def __le__(self, other):
        return self.compare(other, "LE")

    def __gt__(self, other):
        return self.compare(other, "GT")

    def __ge__(self, other):
        return self.compare(other, "GE")


class _MxMaskValue(_MxComparable):
    def __init__(self, value):
        self.value = value

    def __and__(self, other):
        return _MxMaskValue(tla.bitwise_and(self.value, other.value))

    def __or__(self, other):
        return _MxMaskValue(tla.bitwise_or(self.value, other.value))


def _mx_vreg(value):
    return value.value if isinstance(value, _MxIndexValue) else value


class _MxIndexValue(_MxComparable):
    def __init__(self, value, origin=None, offset=None, scalar=None):
        self.value, self.origin, self.offset, self.scalar = value, origin, offset, scalar

    def __add__(self, other):
        return _MxIndexValue(tla.add(self.value, _mx_vreg(other)), self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return _MxIndexValue(tla.sub(self.value, _mx_vreg(other)))

    def __rsub__(self, other):
        return _MxIndexValue(tla.sub(tla.add(tla.full(0, tla.Int32), _mx_vreg(other)), self.value))

    def __mod__(self, other):
        if not isinstance(other, int) or other <= 0 or other & (other - 1):
            raise NotImplementedError("vector modulo requires a power-of-two divisor")
        return _MxIndexValue(tla.bitwise_and(self.value, tla.full(other - 1, tla.Int32)))

    def __floordiv__(self, other):
        if not isinstance(other, int) or other <= 0 or other & (other - 1):
            raise NotImplementedError("vector division requires a power-of-two divisor")
        return _MxIndexValue(tla.shift_right(self.value, other.bit_length() - 1))

    def compare(self, other, mode):
        return _MxMaskValue(tla.cmp(self.value, _mx_vreg(other), mode=mode.lower()))


class _MxLabel64(_MxComparable):
    def __init__(self, low, high):
        self.low, self.high = low, high

    def __eq__(self, other):
        return _MxMaskValue(
            tla.bitwise_and(
                tla.cmp(self.low, other.low, mode="eq"), tla.cmp(self.high, other.high, mode="eq")
            )
        )

    def __ne__(self, other):
        return _MxMaskValue(
            tla.bitwise_or(
                tla.cmp(self.low, other.low, mode="ne"), tla.cmp(self.high, other.high, mode="ne")
            )
        )


def _mx_same_index(left, right):
    if left is right:
        return True
    if hasattr(left, "ir_value") and hasattr(right, "ir_value"):
        return left.ir_value() == right.ir_value()
    return False


def _mx_cached_aux(
    source, cache, strides, prefix, qstart, wide, key_caches=(), key_starts=(), global_offsets=()
):
    return _MxCachedAux(
        source,
        cache,
        strides,
        wide,
        prefix,
        qstart,
        key_caches=key_caches,
        key_starts=key_starts,
        global_offsets=global_offsets,
    )


def _mx_global_index(index, origin, offset):
    return (
        isinstance(index, (_MxIndexValue, _MxShiftedIndex))
        and index.origin is origin
        and _mx_same_index(index.offset, offset)
    )


def _mx_cached_index(index, origin, offsets, axis):
    return (_mx_global_index(index, origin, offsets[axis]) if offsets else False) or (
        (not offsets or not offsets[2]) and _mx_same_index(index, origin)
    )


class _MxShiftedIndex:
    def __init__(self, origin, offset):
        self.origin, self.offset = origin, offset


class _MxCachedAux:
    def __init__(
        self,
        source,
        cache,
        strides,
        wide,
        prefix,
        qstart,
        context=None,
        query_value=None,
        key_caches=(),
        key_starts=(),
        global_offsets=(),
    ):
        self.global_offsets = global_offsets
        self.source, self.cache, self.strides, self.wide = source, cache, strides, wide
        self.context, self.query_value, self.prefix, self.qstart = (
            context,
            query_value,
            prefix,
            qstart,
        )
        self.key_caches, self.key_starts = key_caches, key_starts

    def query_index(self, sub):
        return sub * 64 if self.key_caches else self.prefix + self.qstart + sub * 64

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __getitem__(self, index):
        return self.source[index]

    def load_cached(self, indices):
        if self.context is None:
            return None
        b, h, q, k, key_window = self.context
        prefix = (b, h) if len(self.strides) == 3 else (b,) if len(self.strides) == 2 else ()
        if len(indices) != len(self.strides) or not all(
            _mx_same_index(i, j) for i, j in zip(indices[:-1], prefix)
        ):
            raise NotImplementedError("vector mask requires direct auxiliary token indices")
        qmatch = _mx_cached_index(indices[-1], q, self.global_offsets, 0)
        kmatch = _mx_cached_index(indices[-1], k, self.global_offsets, 1)
        if qmatch:
            return self.query_value
        if kmatch:
            scalar = k.scalar if isinstance(k, _MxIndexValue) else k
            index = (
                scalar - self.key_starts[key_window - 1]
                if self.key_caches
                else self.prefix + scalar
            )
            cache = self.key_caches[key_window - 1] if self.key_caches else self.cache
            stride = 2 if self.wide else 1
            tile = tla.get_tile(cache, tla.make_coord(0, index * stride), tla.make_shape(1, 64))
            low = tile.load(tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
            if self.wide:
                tile = tla.get_tile(cache, tla.make_coord(0, index * 2 + 1), tla.make_shape(1, 64))
                high = tile.load(
                    tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
                )
                return _MxLabel64(low, high)
            return _MxIndexValue(low)
        raise NotImplementedError("vector mask requires direct auxiliary token indices")


def _mx_load_query_aux(aux, sub):
    values = ()
    for t in aux:
        if not isinstance(t, _MxCachedAux):
            raise NotImplementedError("vector mask requires staged token auxiliary data")
        if t.wide:
            first = tla.get_tile(
                t.cache, tla.make_coord(0, t.query_index(sub) * 2), tla.make_shape(1, 64)
            ).load()
            second = tla.get_tile(
                t.cache, tla.make_coord(0, t.query_index(sub) * 2 + 64), tla.make_shape(1, 64)
            ).load()
            lane = tla.arange(0, dtype=tla.Int32)
            even = tla.mul(tla.bitwise_and(lane, tla.full(31, tla.Int32)), 2)
            odd = tla.add(even, 1)
            first_half = tla.cmp(lane, 32, mode="lt")
            low = tla.where(first_half, tla.gather(first, even), tla.gather(second, even))
            high = tla.where(first_half, tla.gather(first, odd), tla.gather(second, odd))
            value = _MxLabel64(low, high)
        else:
            tile = tla.get_tile(
                t.cache, tla.make_coord(0, t.query_index(sub)), tla.make_shape(1, 64)
            )
            value = _MxIndexValue(tile.load())
        values = values + (value,)
    return values


@lru_cache(maxsize=128)
def _mx_trace_callback(fn):
    """Evaluate metadata branches in Python for optional vector-mask tracing."""
    import inspect
    from types import FunctionType

    memo = {}

    def cell(value):
        return (lambda: value).__closure__[0]

    def unwrap(value):
        if getattr(value, "_tla_jit", False):
            source = inspect.unwrap(value)
        elif hasattr(value, "fn") and inspect.isfunction(value.fn):
            source = value.fn
        else:
            return value
        if id(value) in memo:
            result = memo[id(value)]
            if result is None:
                raise NotImplementedError("recursive callbacks require SIMT")
            return result
        memo[id(value)] = None
        scope = source.__globals__.copy()
        for name in source.__code__.co_names:
            if name in scope:
                scope[name] = unwrap(scope[name])
        closure = (
            tuple(cell(unwrap(c.cell_contents)) for c in source.__closure__)
            if source.__closure__
            else None
        )
        result = FunctionType(source.__code__, scope, source.__name__, source.__defaults__, closure)
        result.__kwdefaults__ = source.__kwdefaults__
        memo[id(value)] = result
        return result

    return unwrap(fn)


class _MxMaskDependency:
    """Conservative query dependence, used only for compile-time routing."""

    def __init__(self, query=False):
        self.query = query

    def combine(self, other):
        return _MxMaskDependency(
            self.query or (isinstance(other, _MxMaskDependency) and other.query)
        )

    __add__ = __radd__ = __sub__ = __rsub__ = combine
    __mul__ = __rmul__ = __floordiv__ = __rfloordiv__ = combine
    __mod__ = __rmod__ = __and__ = __rand__ = __or__ = __ror__ = combine
    __xor__ = __rxor__ = __lshift__ = __rshift__ = combine
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = combine

    def __neg__(self):
        return self

    __invert__ = __neg__

    def to(self, *args, **kwargs):
        return self

    def __bool__(self):
        raise NotImplementedError("data-dependent branches need conservative routing")


class _MxCacheRoleIndex(_MxMaskDependency):
    def __add__(self, offset):
        return _MxShiftedIndex(self, offset)

    __radd__ = __add__


class _MxCacheRoleAux:
    def __init__(self, source, rank, b, h, q, k, info, packed):
        self.source, self.rank, self.b, self.h = source, rank, b, h
        self.q, self.k, self.info, self.packed = q, k, info, packed
        self.used = [False, False]

    def __getattr__(self, name):
        return getattr(self.source, name)

    def load_cached(self, indices):
        prefix = (self.b, self.h) if self.rank == 3 else (self.b,) if self.rank == 2 else ()
        if len(indices) != self.rank or not all(
            _mx_same_index(i, j) for i, j in zip(indices[:-1], prefix)
        ):
            raise NotImplementedError("unknown auxiliary prefix requires both caches")
        position = indices[-1]
        qmatch = _mx_cached_index(
            position, self.q, (*self.info[2:], self.packed) if self.rank == 1 else (), 0
        )
        kmatch = _mx_cached_index(
            position, self.k, (*self.info[2:], self.packed) if self.rank == 1 else (), 1
        )
        if not qmatch and not kmatch:
            raise NotImplementedError("unknown auxiliary indexing requires both caches")
        self.used[0] |= qmatch
        self.used[1] |= kmatch
        return _MxMaskDependency()

    def __getitem__(self, index):
        return self.load_cached(index if isinstance(index, tuple) else (index,))


def _mx_aux_cache_roles(mask_mod, aux, scalars, metadata, packed):
    """Omit only auxiliary DMA roles proven unused by the whole callback."""
    fallback = tuple((True, True) for _ in aux)
    if len(metadata) != len(aux):
        return fallback
    b, h = _MxMaskDependency(), _MxMaskDependency()
    q, k = _MxCacheRoleIndex(), _MxCacheRoleIndex()
    info = tuple(_MxMaskDependency() for _ in range(4))
    views = tuple(
        _MxCacheRoleAux(t, len(meta[1]), b, h, q, k, info, packed) for t, meta in zip(aux, metadata)
    )
    try:
        _mx_trace_callback(mask_mod)(0.0, b, h, q, k, views, scalars, info, True)
    except Exception:
        return fallback
    return tuple(tuple(v.used) for v in views)


class _MxDependencyAux:
    def __init__(self, source):
        self.dtype = getattr(source, "dtype", None)
        if self.dtype is None:
            self.dtype = source.ptr.dtype

    def load_cached(self, indices):
        return _MxMaskDependency(any(isinstance(i, _MxMaskDependency) and i.query for i in indices))

    def __getitem__(self, indices):
        return self.load_cached(indices if isinstance(indices, tuple) else (indices,))


def _mx_mask_depends_on_query(mask_mod, aux, scalars):
    """Keep query-independent KV masks on the existing guarded fast pipeline."""
    q, scalar = _MxMaskDependency(True), _MxMaskDependency()
    try:
        result = _mx_trace_callback(mask_mod)(
            0.0,
            scalar,
            scalar,
            q,
            scalar,
            tuple(_MxDependencyAux(t) for t in aux),
            scalars,
            (scalar,) * 4,
            True,
        )
    except Exception:
        return True
    return result.query if isinstance(result, _MxMaskDependency) else True


def _mx_apply_vector_mask(mask_mod, b, h, q, k, aux, values, scalars, info, key_window):
    if not aux or any(t.global_offsets for t in aux):
        k = _MxIndexValue(tla.add(tla.full(0, tla.Int32), k), scalar=k)
    views = tuple(
        _MxCachedAux(
            t.source,
            t.cache,
            t.strides,
            t.wide,
            t.prefix,
            t.qstart,
            (b, h, q, k, key_window),
            value,
            t.key_caches,
            t.key_starts,
            t.global_offsets,
        )
        for t, value in zip(aux, values)
    )
    result = _mx_trace_callback(mask_mod)(tla.Float32(0.0), b, h, q, k, views, scalars, info, True)
    return result.value


def _mx_query_vector(qstart, sub):
    return _MxIndexValue(tla.arange(qstart + sub * 64, dtype=tla.Int32))


@tla.jit
def _mx_cache_aux_pair(
    mask_mod: tla.Constexpr,
    b,
    h,
    qstart,
    sub,
    k0,
    k1,
    mask0,
    mask1,
    aux,
    scalars,
    info,
    checked: tla.Constexpr,
    paired: tla.Constexpr,
):
    with tla.vec.func(mode="simd"):
        query = _mx_query_vector(qstart, sub)
        values = _mx_load_query_aux(aux, sub)
        for row in tla.range(128):
            key = k0 + row
            visible = _mx_apply_vector_mask(
                mask_mod, b, h, query, key, aux, values, scalars, info, 1
            )
            if tla.const_expr(not checked):
                visible = tla.bitwise_and(visible, tla.cmp(query.value, info[0], mode="lt"))
                visible = tla.bitwise_and(
                    visible, tla.cmp(tla.add(tla.full(0, tla.Int32), key), info[1], mode="lt")
                )
            tla.get_tile(mask0, tla.make_coord(row, 0), tla.make_shape(1, 8)).store(
                visible, tla.params.MaskStoreParams()
            )
            if tla.const_expr(paired):
                key1 = k1 + row
                visible1 = _mx_apply_vector_mask(
                    mask_mod, b, h, query, key1, aux, values, scalars, info, 2
                )
                if tla.const_expr(not checked):
                    visible1 = tla.bitwise_and(visible1, tla.cmp(query.value, info[0], mode="lt"))
                    visible1 = tla.bitwise_and(
                        visible1, tla.cmp(tla.add(tla.full(0, tla.Int32), key1), info[1], mode="lt")
                    )
                tla.get_tile(mask1, tla.make_coord(row, 0), tla.make_shape(1, 8)).store(
                    visible1, tla.params.MaskStoreParams()
                )


class _MxLabelPosition:
    def __add__(self, offset):
        return _MxShiftedIndex(self, offset)

    __radd__ = __add__

    def __bool__(self):
        raise NotImplementedError("label specialization requires an equality expression")


class _MxLabelReference:
    def __init__(self, role, ptr, wide):
        self.role, self.ptr, self.wide = role, ptr, wide

    def __eq__(self, other):
        if (
            not isinstance(other, _MxLabelReference)
            or self.wide != other.wide
            or self.role == other.role
        ):
            raise NotImplementedError("label specialization requires Q/K equality")
        q, k = (self, other) if self.role == "q" else (other, self)
        return _MxLabelEquality(q.ptr, k.ptr, self.wide)


class _MxLabelEquality:
    def __init__(self, q, k, wide):
        self.q, self.k, self.wide = q, k, wide

    def __bool__(self):
        raise NotImplementedError("label specialization requires an equality expression")


class _MxProbeAux:
    def __init__(self, tensor, b, h, q, k, sub, kstart):
        self.tensor, self.b, self.h, self.q, self.k = tensor, b, h, q, k
        self.sub, self.kstart = sub, kstart

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def load_cached(self, indices):
        t = self.tensor
        prefix = (
            (self.b, self.h) if len(t.strides) == 3 else (self.b,) if len(t.strides) == 2 else ()
        )
        if len(indices) != len(t.strides) or not all(
            _mx_same_index(i, j) for i, j in zip(indices[:-1], prefix)
        ):
            raise NotImplementedError("label specialization requires direct token indexing")
        words = 2 if t.wide else 1
        qmatch = _mx_cached_index(indices[-1], self.q, t.global_offsets, 0)
        kmatch = _mx_cached_index(indices[-1], self.k, t.global_offsets, 1)
        if qmatch:
            return _MxLabelReference("q", t.cache.ptr + t.query_index(self.sub) * words, t.wide)
        if kmatch:
            return _MxLabelReference(
                "k",
                t.key_caches[0].ptr
                if t.key_caches
                else t.cache.ptr + (t.prefix + self.kstart) * words,
                t.wide,
            )
        raise NotImplementedError("label specialization requires direct token indexing")

    def __getitem__(self, index):
        raise NotImplementedError("label specialization requires logical auxiliary views")


def _mx_trace_label_equality(mask_mod, b, h, sub, kstart, aux, scalars, info):
    if not aux or not all(isinstance(t, _MxCachedAux) for t in aux):
        return None
    q, k = _MxLabelPosition(), _MxLabelPosition()
    views = tuple(_MxProbeAux(t, b, h, q, k, sub, kstart) for t in aux)
    try:
        result = _mx_trace_callback(mask_mod)(
            tla.Float32(0.0), b, h, q, k, views, scalars, info, True
        )
    except Exception:
        return None
    return result if isinstance(result, _MxLabelEquality) else None


_CPP_LABEL_MASK = (
    _MX_CPP_HEADER
    + r"""template<bool WIDE>
__simd_callee__ inline void mx_load_labels(RegTensor<uint32_t>& lo,RegTensor<uint32_t>& hi,__ubuf__ uint32_t* ptr) {
 if constexpr(WIDE) {
  RegTensor<float> a,b;LoadAlign<float,LoadDist::DIST_DINTLV_B32>(a,b,(__ubuf__ float*)ptr);
  lo=reinterpret_cast<RegTensor<uint32_t>&>(a);hi=reinterpret_cast<RegTensor<uint32_t>&>(b);
 } else {LoadAlign(lo,ptr);Duplicate(hi,uint32_t(0));}
}
template<bool WIDE>
__simd_vf__ inline void mx_label_probe_vf(__ubuf__ uint32_t* q,__ubuf__ uint32_t* k,__ubuf__ uint32_t* result) {
 MaskReg all=CreateMask<uint32_t,MaskPattern::ALL>(),m0,m1,match,uniform;
 RegTensor<uint32_t> qlo,qhi,firstlo,firsthi,zero,one,a,b,count0,count1,good,encoded;
 mx_load_labels<WIDE>(qlo,qhi,q);LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(firstlo,q);
 if constexpr(WIDE) LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(firsthi,q+1);else Duplicate(firsthi,uint32_t(0));
 Duplicate(zero,uint32_t(0));Duplicate(one,uint32_t(1));
 Compare<uint32_t,CMPMODE::EQ>(m0,qlo,firstlo,all);Compare<uint32_t,CMPMODE::EQ>(m1,qhi,firsthi,all);And(match,m0,m1,all);
 Select(good,one,zero,match);ReduceMin(good,good,all);
 mx_load_labels<WIDE>(a,b,k);Compare<uint32_t,CMPMODE::EQ>(m0,a,firstlo,all);Compare<uint32_t,CMPMODE::EQ>(m1,b,firsthi,all);And(match,m0,m1,all);Select(count0,one,zero,match);ReduceSum(count0,count0,all);
 mx_load_labels<WIDE>(a,b,k+64*(WIDE?2:1));Compare<uint32_t,CMPMODE::EQ>(m0,a,firstlo,all);Compare<uint32_t,CMPMODE::EQ>(m1,b,firsthi,all);And(match,m0,m1,all);Select(count1,one,zero,match);ReduceSum(count1,count1,all);
 Add(encoded,count0,count1,all);ShiftLefts(encoded,encoded,int16_t(1),all);Adds(encoded,encoded,uint32_t(1),all);
 RegTensor<uint32_t> halves,bit;MaskReg valid0,valid1,empty,full;
 Compares<uint32_t,CMPMODE::EQ>(empty,count0,uint32_t(0),all);Compares<uint32_t,CMPMODE::EQ>(full,count0,uint32_t(64),all);Or(valid0,empty,full,all);Select(halves,one,zero,full);
 Compares<uint32_t,CMPMODE::EQ>(empty,count1,uint32_t(0),all);Compares<uint32_t,CMPMODE::EQ>(full,count1,uint32_t(64),all);Or(valid1,empty,full,all);Select(bit,one,zero,full);
 ShiftLefts(bit,bit,int16_t(1),all);Or(halves,halves,bit,all);And(valid0,valid0,valid1,all);Select(halves,halves,zero,valid0);ShiftLefts(halves,halves,int16_t(10),all);Or(encoded,encoded,halves,all);
 Compares<uint32_t,CMPMODE::EQ>(uniform,good,uint32_t(1),all);Select(encoded,encoded,zero,uniform);StoreAlign(result,encoded,all);
}
template<bool WIDE>
__simd_vf__ inline void mx_label_uniform_vf(__ubuf__ uint32_t* q,__ubuf__ uint32_t* k,__ubuf__ uint32_t* output) {
 MaskReg all=CreateMask<uint32_t,MaskPattern::ALL>(),m0,m1,match;
 RegTensor<uint32_t> firstlo,firsthi,zero,ones,a,b,pattern,index,offset,packed;
 LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(firstlo,q);
 if constexpr(WIDE) LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(firsthi,q+1);else Duplicate(firsthi,uint32_t(0));
 Duplicate(zero,uint32_t(0));Duplicate(ones,uint32_t(0x11111111));RegTensor<int32_t> sequence;Arange(sequence,int32_t(0));index=reinterpret_cast<RegTensor<uint32_t>&>(sequence);ShiftRights(index,index,int16_t(3),all);
 for(uint16_t part=0;part<2;part++) {
  mx_load_labels<WIDE>(a,b,k+part*64*(WIDE?2:1));Compare<uint32_t,CMPMODE::EQ>(m0,a,firstlo,all);Compare<uint32_t,CMPMODE::EQ>(m1,b,firsthi,all);And(match,m0,m1,all);Select(pattern,ones,zero,match);
  #pragma unroll 4
  for(uint16_t chunk=0;chunk<8;chunk++) {Adds(offset,index,uint32_t(chunk*8),all);Gather(packed,pattern,offset);StoreAlign(output+part*512+chunk*64,packed,all);}
 }
}
extern "C" [aicore] __attribute__((always_inline)) void mx_label_probe(uint64_t q,uint64_t k,uint64_t result,int32_t wide) {
 if(wide) mx_label_probe_vf<true>((__ubuf__ uint32_t*)q,(__ubuf__ uint32_t*)k,(__ubuf__ uint32_t*)result);
 else mx_label_probe_vf<false>((__ubuf__ uint32_t*)q,(__ubuf__ uint32_t*)k,(__ubuf__ uint32_t*)result);
}
extern "C" [aicore] __attribute__((always_inline)) void mx_label_uniform(uint64_t q,uint64_t k,uint64_t result,int32_t wide) {
 if(wide) mx_label_uniform_vf<true>((__ubuf__ uint32_t*)q,(__ubuf__ uint32_t*)k,(__ubuf__ uint32_t*)result);
 else mx_label_uniform_vf<false>((__ubuf__ uint32_t*)q,(__ubuf__ uint32_t*)k,(__ubuf__ uint32_t*)result);
}
"""
)


@tla.extern(name="mx_label_probe", source=_CPP_LABEL_MASK)
def mx_label_probe(
    q: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    k: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    result: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    wide: tla.Int32,
) -> None: ...
@tla.extern(name="mx_label_uniform", source=_CPP_LABEL_MASK)
def mx_label_uniform(
    q: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    k: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    result: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    wide: tla.Int32,
) -> None: ...


class _MxUnknownIndex:
    def unsupported(self, *args, **kwargs):
        raise NotImplementedError("this scalar requires the generic callback")

    __bool__ = __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = unsupported
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = unsupported
    __mod__ = __rmod__ = __floordiv__ = __rfloordiv__ = unsupported
    __and__ = __or__ = __xor__ = __index__ = __int__ = unsupported
    __getattr__ = unsupported


class _MxKeyOffset:
    pass


class _MxKeyPosition(_MxUnknownIndex):
    def __init__(self, global_key=False):
        self.global_key = global_key

    def __add__(self, offset):
        if not isinstance(offset, _MxKeyOffset):
            raise NotImplementedError
        return _MxKeyPosition(True)

    __radd__ = __add__

    def __ge__(self, other):
        if self.global_key or not isinstance(other, _MxKeyAuxValue):
            raise NotImplementedError
        return ("key_aux_ge", other.global_key)


class _MxKeyAuxValue(_MxUnknownIndex):
    def __init__(self, global_key):
        self.global_key = global_key

    def __le__(self, other):
        return other.__ge__(self)


class _MxKeyAuxProbe:
    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]
        if not isinstance(index, _MxKeyPosition):
            raise NotImplementedError
        return _MxKeyAuxValue(index.global_key)


def _mx_trace_key_aux(mask_mod):
    import inspect

    unknown = _MxUnknownIndex()
    try:
        value = inspect.unwrap(mask_mod)(
            unknown,
            unknown,
            unknown,
            _MxKeyPosition(),
            (unknown, unknown, unknown, _MxKeyOffset()),
            (_MxKeyAuxProbe(),),
            (),
        )
    except Exception:
        return ()
    return (value[1],) if type(value) is tuple and value[0] == "key_aux_ge" else ()


class _MxAffineMask(_MxComparable):
    def __init__(self, tree):
        self.tree = tree

    def __and__(self, other):
        return _MxAffineMask(("and", self.tree, other.tree))

    def __or__(self, other):
        return _MxAffineMask(("or", self.tree, other.tree))


class _MxAffineIndex(_MxComparable):
    def __init__(self, q=0, k=0, c=0, qlen=0, klen=0):
        self.coefficients = (q, k, c, qlen, klen)

    @staticmethod
    def lift(other):
        if isinstance(other, _MxAffineIndex):
            return other
        if type(other) is int and -(2**30) < other < 2**30:
            return _MxAffineIndex(c=other)
        raise NotImplementedError("affine masks require integer constants")

    def __add__(self, other):
        return _MxAffineIndex(
            *(a + b for a, b in zip(self.coefficients, self.lift(other).coefficients))
        )

    __radd__ = __add__

    def __sub__(self, other):
        return _MxAffineIndex(
            *(a - b for a, b in zip(self.coefficients, self.lift(other).coefficients))
        )

    def __rsub__(self, other):
        return self.lift(other).__sub__(self)

    def __mod__(self, divisor):
        return _MxPeriodicIndex(self, divisor, "mod")

    def __floordiv__(self, divisor):
        return _MxPeriodicIndex(self, divisor, "div")

    def compare(self, other, mode):
        return _MxAffineMask((mode, (self - other).coefficients))


class _MxPeriodicIndex(_MxComparable):
    """Positive power-of-two integer operations; other cases retain SIMT."""

    def __init__(self, value, divisor, operation):
        if type(divisor) is not int or divisor <= 0 or divisor >= 2**31 or divisor & (divisor - 1):
            raise NotImplementedError("periodic masks require a positive power of two")
        self.tree = (operation, value.coefficients, divisor)

    @staticmethod
    def lift(other):
        if isinstance(other, _MxPeriodicIndex):
            return other.tree
        return ("affine", _MxAffineIndex.lift(other).coefficients, 1)

    def compare(self, other, mode):
        return _MxAffineMask(("periodic", mode, self.tree, self.lift(other)))


def _mx_mask_bounds(tree):
    """A metadata-only bound on the local KV neighborhood, or unknown."""
    lower, upper = [], []

    def visit(node):
        mode, *args = node
        if mode == "and":
            visit(args[0])
            visit(args[1])
            return
        if mode == "or":
            return
        if mode == "periodic":
            comparison, left, right = args
            if comparison == "EQ":
                if right[0] == "div" and left[0] == "affine":
                    left, right = right, left
                if left[0] == "div" and right[0] == "affine":
                    a, k, c, ql, kl = left[1]
                    rq, rk, rc, rql, rkl = right[1]
                    if (a, k) == (1, -1) and (rq, rk, rql, rkl) == (0, 0, 0, 0):
                        lower.append((rc * left[2] - c, -ql, -kl))
                        upper.append(((rc + 1) * left[2] - 1 - c, -ql, -kl))
            if comparison == "EQ" and left[0] == right[0] == "div" and left[2] == right[2]:
                axes = {left[1], right[1]}
                if axes == {(1, 0, 0, 0, 0), (0, 1, 0, 0, 0)}:
                    lower.append((1 - left[2], 0, 0))
                    upper.append((left[2] - 1, 0, 0))
            return
        a, k, c, ql, kl = args[0]
        if (a, k) == (-1, 1):
            a, k, c, ql, kl = 1, -1, -c, -ql, -kl
            mode = {"LT": "GT", "LE": "GE", "GT": "LT", "GE": "LE", "EQ": "EQ", "NE": "NE"}[mode]
        if (a, k) != (1, -1):
            return
        if mode in ("GE", "GT", "EQ"):
            lower.append((-c + int(mode == "GT"), -ql, -kl))
        if mode in ("LE", "LT", "EQ"):
            upper.append((-c - int(mode == "LT"), -ql, -kl))

    visit(tree)
    return lower, upper


def _mx_mask_span(tree):
    lower, upper = _mx_mask_bounds(tree)
    return min(
        (max(0, hi[0] - lo[0] + 1) for lo in lower for hi in upper if lo[1:] == hi[1:]),
        default=None,
    )


def _mx_trace_affine_tree(mask_mod, scalar_count, packed=False):
    import inspect

    try:
        info = (
            _MxAffineIndex(qlen=1),
            _MxAffineIndex(klen=1),
            _MxUnknownIndex() if packed else 0,
            _MxUnknownIndex() if packed else 0,
        )
        result = inspect.unwrap(mask_mod)(
            _MxUnknownIndex(),
            _MxUnknownIndex(),
            _MxAffineIndex(q=1),
            _MxAffineIndex(k=1),
            info,
            (),
            tuple(_MxUnknownIndex() for _ in range(scalar_count)),
        )
    except Exception:
        return None
    return result.tree if isinstance(result, _MxAffineMask) else None


def _mx_trace_affine_mask(mask_mod, nq, nk, scalar_count, post_quant=False, packed=False):
    tree = _mx_trace_affine_tree(mask_mod, scalar_count, packed)
    return _mx_affine_quantizer(tree, post_quant) if tree is not None else None


@lru_cache(maxsize=128)
def _mx_is_periodic_prefix(mask_mod, scalar_count, packed=False):
    """Prove a block-local causal predicate with the attention tile period."""
    tree = _mx_trace_affine_tree(mask_mod, scalar_count, packed)
    query = ("mod", (1, 0, 0, 0, 0), 128)
    key = ("mod", (0, 1, 0, 0, 0), 128)
    return tree in (("periodic", "GE", query, key), ("periodic", "LE", key, query))


def _mx_dense_mask_bounds(mask_mod, scalar_count, packed=False):
    """Reuse proven affine constraints; never inspect auxiliary tensor values."""
    tree = _mx_trace_affine_tree(mask_mod, scalar_count, packed)
    if tree is None:
        return ()
    lower, upper = _mx_mask_bounds(tree)
    return (tuple(lower), tuple(upper)) if lower or upper else ()


def _mx_mask_is_all_visible(mask_mod):
    """Canonicalize only a predicate proven constant with unknown indices."""
    import inspect

    unknown = _MxUnknownIndex()
    try:
        result = inspect.unwrap(mask_mod)(
            unknown, unknown, unknown, unknown, (unknown,) * 4, (), ()
        )
    except Exception:
        return False
    return result is True or getattr(result, "value", None) is True


class _MxMaskLowering:
    """Lower the symbolic callback to vector instructions, independently of quantization.

    Registers and calls form a small instruction list. Only this callback-dependent
    list is emitted as C++; the quantizer and dispatch are ordinary static functions.
    No kernel source is searched, patched or used as a substitution template.
    """

    def __init__(self):
        self.instructions = []
        self.expressions = {}
        self.operands = {}
        self.predicates = {}
        self.register_count = 0

    def register(self, dtype):
        name = "mask_r" + str(self.register_count)
        self.register_count += 1
        self.instructions.append(("declare", dtype, name))
        return name

    def call(self, operation, *arguments, types=()):
        self.instructions.append(("call", operation, types, arguments))

    @staticmethod
    def scalar(coefficients):
        # Parenthesize signed coefficients and every operation, preserving the
        # original int32 expression and its overflow behavior.
        c, ql, kl = coefficients
        return "(" + str(c) + "+(" + str(ql) + ")*qlen+(" + str(kl) + ")*klen)"

    def expression(self, coefficients):
        a, b, c, ql, kl = coefficients
        if (a, b) not in self.expressions:
            value = self.register("RegTensor<int32_t>")
            key = "int32_t(" + str(b) + "*key)"
            if a == 0:
                self.call("Duplicate", value, key)
            else:
                if a != 1:
                    self.call("Muls", value, "query", "int32_t(" + str(a) + ")", "all")
                self.call("Adds", value, "query" if a == 1 else value, key, "all")
            self.expressions[a, b] = value
        return self.expressions[a, b], self.scalar((c, ql, kl))

    def operand(self, node):
        if node not in self.operands:
            kind, coefficients, divisor = node
            base, shift = self.expression(coefficients)
            value = self.register("RegTensor<int32_t>")
            self.call("Adds", value, base, "int32_t(-(-" + shift + "))", "all")
            if kind == "mod":
                mask = self.register("RegTensor<int32_t>")
                self.call("Duplicate", mask, "int32_t(" + str(divisor - 1) + ")")
                self.call("And", value, value, mask, "all")
            elif kind == "div":
                self.call(
                    "ShiftRights",
                    value,
                    value,
                    "int16_t(" + str(divisor.bit_length() - 1) + ")",
                    "all",
                )
            self.operands[node] = value
        return self.operands[node]

    def predicate(self, node):
        if node not in self.predicates:
            mode, *args = node
            if mode in ("and", "or"):
                left, right = (self.predicate(child) for child in args)
                operation, types = mode.title(), ()
            elif mode == "periodic":
                comparison, lhs, rhs = args
                left, right = self.operand(lhs), self.operand(rhs)
                operation, types = "Compare", ("int32_t", "CMPMODE::" + comparison)
            else:
                left, shift = self.expression(args[0])
                right = "int32_t(-" + shift + ")"
                operation, types = "Compares", ("int32_t", "CMPMODE::" + mode)
            value = self.register("MaskReg")
            self.call(operation, value, left, right, "all", types=types)
            self.predicates[node] = value
        return self.predicates[node]

    def source(self, result, *, lower, upper, sum_denominator, skip_minimum, continuous):
        # Serialize complete policy declarations and instructions. The shared
        # quantizer calls this policy through its C++ type parameter.
        tokens = ["struct MxAffineMask {\n"]
        for name, value in (
            ("lower", lower),
            ("upper", upper),
            ("sum_denominator", int(sum_denominator)),
            ("skip_minimum", int(skip_minimum)),
            ("continuous", int(continuous)),
        ):
            tokens.extend((" static constexpr auto ", name, "=", str(value), ";\n"))
        tokens.append(r"""
 __simd_callee__ static inline void init(RegTensor<int32_t>& query,int32_t qbase,MaskReg& all) {
  Arange(query,int32_t(0));Adds(query,query,qbase,all);
 }
 __simd_callee__ static inline MaskReg load(RegTensor<int32_t>& query,__ubuf__ uint32_t* visibility,
     int32_t row,int32_t key,int32_t qlen,int32_t klen,MaskReg& all) {
  // The caller forms (kbase + row) + lane, retaining the aligned-key strength reduction.
""")
        for instruction in self.instructions:
            if instruction[0] == "declare":
                _, dtype, name = instruction
                tokens.extend(("  ", dtype, " ", name, ";\n"))
            else:
                _, operation, types, arguments = instruction
                tokens.extend(("  ", operation))
                if types:
                    tokens.extend(("<", ",".join(types), ">"))
                tokens.extend(("(", ",".join(arguments), ");\n"))
        tokens.extend(("  return ", result, ";\n }\n};\n"))
        return "".join(tokens)


_CPP_AFFINE_DISPATCH = r"""
[aicore] __attribute__((always_inline)) int32_t mx_affine_visible(int32_t q,int32_t k,int32_t lower,int32_t upper) {
 int32_t a=q-upper,b=q-lower+1,c=q+63-upper,d=q+64-lower;
 a=a<k?k:a;b=b>k+128?k+128:b;c=c<k?k:c;d=d>k+128?k+128:d;
 return b-a<d-c?b-a:d-c;
}
template<bool FULL>
[aicore] __attribute__((always_inline)) void mx_affine_dispatch(
 uint64_t s,uint64_t p,uint64_t m,uint64_t d,int32_t tile,int32_t qbase,int32_t kbase,
 uint64_t offset,int32_t single,int32_t qlen,int32_t klen,int32_t first,int32_t last,int32_t cols) {
 using Mask=MxAffineMask;
#define CALL(ADAPTIVE,TILE,OFFSET) mx_short_quantize_vf<FULL,true,ADAPTIVE,true,Mask::sum_denominator,!Mask::skip_minimum,Mask>((__ubuf__ half*)s,(__ubuf__ uint8_t*)p,(__ubuf__ float*)m,(__ubuf__ float*)d,TILE,cols,nullptr,OFFSET,first,last,qbase,kbase,qlen,klen)
 if(single) {
  int32_t minimum=0;
  if constexpr(Mask::continuous) minimum=mx_affine_visible(qbase,kbase,Mask::lower,Mask::upper);
  if(minimum>=32) CALL(false,0,(__ubuf__ float*)offset);
  else CALL(true,0,(__ubuf__ float*)offset);
 } else CALL(false,tile,nullptr);
#undef CALL
}
extern "C" [aicore] __attribute__((always_inline)) void mx_affine_quantize_dispatch(
 uint64_t s,uint64_t p,uint64_t m,uint64_t d,int32_t tile,int32_t qbase,int32_t kbase,
 uint64_t offset,int32_t single,int32_t qlen,int32_t klen) {
 using Mask=MxAffineMask;
 int32_t first=0,last=128;
 if constexpr(Mask::upper!=(1<<30)) first=qbase-kbase-Mask::upper;
 if constexpr(Mask::lower!=-(1<<30)) last=qbase-kbase+64-Mask::lower;
 first=first<0?0:(first>128?128:first);last=last<0?0:(last>128?128:last);
 first=(first/4)*4;last=((last+3)/4)*4;
 if(kbase+128<=klen) mx_affine_dispatch<true>(s,p,m,d,tile,qbase,kbase,offset,single,qlen,klen,first,last,128);
 else mx_affine_dispatch<false>(s,p,m,d,tile,qbase,kbase,offset,single,qlen,klen,first,last,klen-kbase);
}
"""


@lru_cache(maxsize=128)
def _mx_affine_quantizer(tree, post_quant=False):
    def continuous_interval(node):
        mode, *args = node
        if mode == "and":
            return continuous_interval(args[0]) and continuous_interval(args[1])
        return (
            mode in ("GE", "GT", "LE", "LT", "EQ")
            and args[0][:2] in ((1, -1), (-1, 1))
            and args[0][3:] == (0, 0)
        )

    continuous = continuous_interval(tree)
    bounded = (1 if continuous else 2) if not post_quant else 0
    lows, highs = _mx_mask_bounds(tree)
    lower = max((x[0] for x in lows if x[1:] == (0, 0)), default=-(2**30))
    upper = min((x[0] for x in highs if x[1:] == (0, 0)), default=2**30)
    lowering = _MxMaskLowering()
    result = lowering.predicate(tree)
    policy = lowering.source(
        result,
        lower=lower,
        upper=upper,
        sum_denominator=not post_quant,
        skip_minimum=bounded == 1,
        continuous=continuous,
    )

    @tla.extern(
        name="mx_affine_quantize_dispatch", source=_CPP_SHORT_BODY + policy + _CPP_AFFINE_DISPATCH
    )
    def quantize(
        s: tla.Pointer[tla.Float16, tla.AddressSpace.ub],
        p: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
        m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
        d: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
        tile: tla.Int32,
        qbase: tla.Int32,
        kbase: tla.Int32,
        offset: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
        single: tla.Int32,
        qlen: tla.Int32,
        klen: tla.Int32,
    ) -> None: ...
    def periodic(node):
        return node[0] == "periodic" or (
            node[0] in ("and", "or") and any(periodic(child) for child in node[1:])
        )

    return quantize, bounded, _mx_mask_span(tree), periodic(tree)


@tla.jit
def _mx_quantize_masked_inline(
    s,
    p,
    m,
    d,
    tile,
    qbase,
    kbase,
    qlen,
    klen,
    visibility,
    inline_quantize: tla.Constexpr,
    post_quant: tla.Constexpr,
    offset=None,
    single: tla.Constexpr = False,
):
    if tla.const_expr(inline_quantize is not None):
        inline_quantize(
            s,
            p,
            m,
            d,
            tile,
            qbase,
            kbase,
            offset if offset is not None else m,
            tla.Int32(single),
            qlen,
            klen,
        )
    else:
        _mx_short_quantize(
            s,
            p,
            m,
            d,
            tile,
            tla.Int32(128),
            visibility,
            m,
            tla.Int32(1),
            tla.Int32(-1),
            tla.Int32(-1),
            tla.Int32(128),
            post_quant,
        )


@tla.jit
def _mx_balanced_task(ordinal, core, cores, total, paired: tla.Constexpr):
    """Keep complete head pairs together; distribute the final wave evenly."""
    linear = core + ordinal * cores
    if tla.const_expr(paired):
        prefix = (total // (cores * 2)) * 2
        if ordinal < prefix:
            linear = (core + (ordinal // 2) * cores) * 2 + ordinal % 2
        else:
            linear = prefix * cores + core + (ordinal - prefix) * cores
    return linear


@tla.jit
def _mx_repair_task(
    cursor,
    flags,
    cu_q,
    cu_k,
    hq,
    blocks,
    seq,
    prefix,
    scales,
    core,
    cores,
    total,
    paired: tla.Constexpr,
    qp: tla.Constexpr,
    kp: tla.Constexpr,
    q_major: tla.Constexpr,
    single: tla.Constexpr,
):
    task, found = tla.Int32(0), tla.Bool(False)
    while cursor < total // cores + tla.Int32(core < total % cores) and not found:
        task, seq, prefix, scales = _mx_advance_task(
            _mx_balanced_task(cursor, core, cores, total, paired),
            cu_q,
            cu_k,
            hq,
            blocks,
            seq,
            prefix,
            scales,
            qp,
            kp,
            q_major,
            single,
        )
        cursor += 1
        found = flags[task * 16] != 0 or flags[task * 16 + 8] != 0
    return task, cursor, seq, prefix, scales, found


@tla.jit
def _mx_slot(slot, flags, pipe: tla.Constexpr = None, wait: tla.Constexpr = True):
    for index in tla.range_constexpr(len(flags)):
        if True if tla.const_expr(slot is None) else slot == index:
            if tla.const_expr(pipe is None):
                operation = tla.wait_flag if wait else tla.set_flag
                operation(flags[index])
            else:
                operation = tla.cross_core_wait_flag if wait else tla.cross_core_set_flag
                operation(flags[index], pipe)


def _mx_aux_numel(shape):
    # Contiguous views may carry arbitrary strides on size-one axes. Those
    # unused strides must not increase DMA lengths or the fixed UB budget.
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _mx_aux_layouts(metadata, scalars):
    """Rebind full-cache DMA geometry without specializing on buffer lengths."""
    count = sum(2 * len(meta[1]) for meta in metadata)
    cursor, offset, result = len(scalars) - count, 0, ()
    for stored, shape, strides, wide in metadata:
        rank = len(shape)
        actual_shape = scalars[cursor : cursor + rank]
        actual_strides = scalars[cursor + rank : cursor + rank * 2]
        cursor += rank * 2
        result += ((stored if stored < 0 else offset, actual_shape, actual_strides, wide),)
        size = _mx_aux_numel(actual_shape) * (8 if wide else 4)
        offset = offset + (size + 511) // 512 * 512
    return result


@tla.jit
def _mx_stage_aux(
    source,
    cache,
    meta: tla.Constexpr,
    b,
    h,
    qstart,
    kstart,
    next_k,
    rows,
    cols,
    next_cols,
    qoff,
    koff,
    packed: tla.Constexpr,
    copy_q,
    copy_k,
    copy_next,
    tiled: tla.Constexpr,
    global_aux: tla.Constexpr,
):
    offset, shape, strides, wide = meta
    prefix = tla.Int32(0)
    if tla.const_expr(len(shape) >= 2):
        prefix = b % shape[0] * strides[0]
    if tla.const_expr(len(shape) == 3):
        prefix += h % shape[1] * strides[1]
    global_offsets = (qoff, koff, packed) if tla.const_expr(len(shape) == 1 and global_aux) else ()
    if tla.const_expr(not tiled):
        return _mx_cached_aux(
            source, cache, strides, prefix, qstart, wide, global_offsets=global_offsets
        )
    words = 2 if wide else 1
    ptr = tla.recast_ptr(source.ptr, dtype=tla.Int32 if wide else source.ptr.dtype)
    query = qstart + (qoff if tla.const_expr(len(global_offsets) > 0) else 0)
    key = kstart + (koff if tla.const_expr(len(global_offsets) > 0) else 0)
    following = next_k + (koff if tla.const_expr(len(global_offsets) > 0) else 0)
    positions, counts, enabled = (
        (query, key, following),
        (rows, cols, next_cols),
        (copy_q, copy_k, copy_next),
    )
    for i in tla.range_constexpr(3):
        if enabled[i]:
            size = counts[i] * words
            gm = tla.make_tensor(
                ptr + (prefix + positions[i]) * words,
                tla.make_layout(tla.make_shape(1, size), tla.make_stride(size, 1)),
            )
            tla.copy(tla.get_tile(cache[i], tla.make_coord(0, 0), tla.make_shape(1, size)), gm)
    return _mx_cached_aux(
        source,
        cache[0],
        strides,
        prefix,
        qstart,
        wide,
        (cache[1], cache[2]),
        (kstart, next_k),
        global_offsets,
    )


@tla.jit
def _mx_l1_cache(address, valid, tags, next_slot):
    slot = tla.Int32(-1)
    for i in tla.range_constexpr(len(tags)):
        if valid and address == tags[i]:
            slot = tla.Int32(i)
    reuse = slot >= 0
    if not reuse:
        slot = next_slot
        next_slot = (next_slot + 1) % len(tags)
        for i in tla.range_constexpr(len(tags)):
            if slot == i:
                tags = tags[:i] + (address if valid else tla.Int32(-1),) + tags[i + 1 :]
    return slot, reuse, tags, next_slot


@tla.jit
def _mx_scalar_min(a, b):
    a, b = tla.Int32(a), tla.Int32(b)
    return a + (b - a) * tla.Int32(b < a)


@tla.jit
def _mx_scalar_max(a, b):
    a, b = tla.Int32(a), tla.Int32(b)
    return a + (b - a) * tla.Int32(b > a)


@tla.jit
def _mx_task_geometry(
    task,
    cu_q,
    cu_k,
    partial_count,
    full_count,
    nq,
    nk,
    hq,
    hk,
    blocks,
    mb,
    mh,
    total_q,
    varlen: tla.Constexpr,
    sequence_major: tla.Constexpr,
    dense: tla.Constexpr,
    dense_mask: tla.Constexpr,
    dense_bounds: tla.Constexpr,
    single: tla.Constexpr,
    packed_k: tla.Constexpr,
):
    b, h, qb = task // (hq * blocks), task // blocks % hq, task % blocks
    kh = h // (hq // hk)
    qstart = qb * 128
    qoff, koff = tla.Int32(0), tla.Int32(0)
    qlen, klen = tla.Int32(nq), tla.Int32(nk)
    qbase = (
        b * nq * hq * 128
        + h * (128 if sequence_major[0] else nq * 128)
        + qstart * (hq * 128 if sequence_major[0] else 128)
    )
    kbase = b * nk * hk * 128 + kh * (128 if sequence_major[1] else nk * 128)
    if tla.const_expr(varlen[0]):
        qoff = tla.Int32(0) if tla.const_expr(single) else cu_q[b]
        qlen = total_q if tla.const_expr(single) else cu_q[b + 1] - qoff
        qbase = (qoff + qstart) * hq * 128 + h * 128
    if tla.const_expr(varlen[1]):
        koff = tla.Int32(0) if tla.const_expr(single) else cu_k[b]
        klen = tla.Int32(packed_k) if tla.const_expr(single) else cu_k[b + 1] - koff
        kbase = koff * (hk * 128 if sequence_major[1] else 128) + kh * (
            128 if sequence_major[1] else packed_k * 128
        )
    maskrow = (b % mb * mh + h % mh) * blocks + qb
    partial = tla.Int32(0)
    count = (klen + 127) // 128
    first_key_block = tla.Int32(0)
    if tla.const_expr(dense and dense_mask):
        # Dense masks evaluate callbacks on implicit tiles, without index lists.
        if tla.const_expr(len(dense_bounds) > 0):
            # q-k bounds cover the entire query tile. Round outward so every
            # potentially visible key remains in the callback's traversal.
            first_key = tla.Int32(0)
            last_key = tla.Int32(klen)
            query_end = _mx_scalar_min(qstart + 128, qlen)
            for i in tla.range_constexpr(len(dense_bounds[0])):
                c, ql, kl = dense_bounds[0][i]
                lower = c + ql * qlen + kl * klen
                last_key = _mx_scalar_min(last_key, query_end - lower)
            for i in tla.range_constexpr(len(dense_bounds[1])):
                c, ql, kl = dense_bounds[1][i]
                upper = c + ql * qlen + kl * klen
                first_key = _mx_scalar_max(first_key, qstart - upper)
            first_key = _mx_scalar_min(first_key, klen)
            last_key = _mx_scalar_max(last_key, 0)
            first_key_block = first_key // 128
            count = _mx_scalar_max(0, (last_key + 127) // 128 - first_key_block)
        partial = count
    if tla.const_expr(not dense):
        partial = partial_count[maskrow]
        count = partial + full_count[maskrow]
    return (
        b,
        h,
        kh,
        qstart,
        qbase,
        kbase,
        qlen,
        klen,
        qoff,
        koff,
        maskrow,
        partial,
        count,
        first_key_block,
    )


@tla.jit
def _mx_callback_probabilities(
    sp,
    pp,
    psp,
    mp,
    lp,
    alpha,
    bad,
    inv,
    visp,
    scale,
    b,
    h,
    qstart,
    kstart,
    cols,
    full,
    first,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    vector_score: tla.Constexpr,
    indexed: tla.Constexpr,
    ragged: tla.Constexpr,
    aux,
    scalars,
    info,
):
    scores = tla.make_tensor(sp, tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)))
    visibility = tla.make_tensor(
        tla.recast_ptr(visp, dtype=tla.Int32),
        tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
    )
    mt, lt, at, bt = (
        _mx_vector_tile(mp, 64),
        _mx_vector_tile(lp, 64),
        _mx_vector_tile(alpha, 64),
        _mx_vector_tile(bad, 64),
    )
    ph0, ph1 = (
        _mx_vector_tile(tla.recast_ptr(psp, dtype=tla.Int16), 128),
        _mx_vector_tile(tla.recast_ptr(psp + 32, dtype=tla.Int16), 128),
    )
    inverses = tla.make_tensor(inv, tla.make_layout(tla.make_shape(4, 64), tla.make_stride(64, 1)))
    if first:
        with tla.vec.func():
            mt.store(tla.full(float("-inf"), tla.Float32))
            lt.store(tla.full(0.0, tla.Float32))
            bt.store(tla.full(0.0, tla.Float32))
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    if cols > 0:
        if tla.const_expr(indexed and vector_score is None):
            prepare_scores_simt(
                scores, scale, b, h, qstart, kstart, full, score_mod, mask_mod, aux, scalars, info
            )
            tla.pipe_barrier(tla.pipes.ALL)
        elif tla.const_expr(indexed):
            if not full:
                _cache_mask(
                    mask_mod,
                    b,
                    h,
                    qstart,
                    (qstart >> 6) & 1,
                    kstart,
                    visibility,
                    aux,
                    scalars,
                    info,
                    False,
                )
                tla.pipe_barrier(tla.pipes.ALL)
    with tla.vec.func():
        bv, hv = tla.add(tla.full(0, tla.Int32), b), tla.add(tla.full(0, tla.Int32), h)
        qi = tla.arange(qstart, dtype=tla.Int32)
        maximum = tla.full(float("-inf"), tla.Float32)
        if cols > 0:
            if full:
                maximum = _score_max_pass(
                    scores,
                    scale,
                    bv,
                    hv,
                    qi,
                    kstart,
                    aux,
                    scalars,
                    vector_score if tla.const_expr(vector_score is not None) else score_mod,
                    mask_mod,
                    False,
                    visibility,
                    False,
                    info,
                    ragged,
                    indexed and vector_score is None,
                )
            else:
                maximum = _score_max_pass(
                    scores,
                    scale,
                    bv,
                    hv,
                    qi,
                    kstart,
                    aux,
                    scalars,
                    vector_score if tla.const_expr(vector_score is not None) else score_mod,
                    mask_mod,
                    not indexed,
                    visibility,
                    indexed and vector_score is not None,
                    info,
                    ragged,
                    indexed and vector_score is None,
                )
        else:
            for row in tla.range(128):
                tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(row, 0)).store(maximum)
        online_quantize_probabilities(scores, mt, lt, at, ph0, ph1, bt, inverses, maximum)
    pack_scaled_rne(sp, pp, inv)


# DMA stores do not invalidate cached scalar reads of the task and core flags.
@tla.extern(
    name="mx_refresh_guard",
    source=_MX_CPP_HEADER
    + r"""extern "C" [aicore] __attribute__((always_inline)) void mx_refresh_guard(uint64_t address) {
 AscendC::GlobalTensor<int32_t> g; g.SetGlobalBuffer((__gm__ int32_t*)address);
 AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE,AscendC::DcciDst::CACHELINE_OUT>(g);
}
""",
)
def _mx_refresh_guard(p: tla.Pointer[tla.Int32, tla.AddressSpace.gm]) -> None: ...
@tla.jit
def _attention_mxfp8_pipeline(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    qs: tla.Tensor,
    ks: tla.Tensor,
    vs: tla.Tensor,
    v_scale_offsets: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    partial_count: tla.Tensor,
    partial_index: tla.Tensor,
    full_count: tla.Tensor,
    full_index: tla.Tensor,
    cu_q: tla.Tensor,
    cu_k: tla.Tensor,
    aux_tensors: tuple,
    aux_scalars: tuple,
    batch: tla.Int32,
    q_heads: tla.Int32,
    kv_heads: tla.Int32,
    max_q: tla.Int32,
    max_k: tla.Int32,
    q_blocks: tla.Int32,
    mask_batch: tla.Int32,
    mask_heads: tla.Int32,
    scale: tla.Float32,
    total_q: tla.Int32,
    total_k: tla.Int32,
    v_scale_capacity: tla.Int32,
    varlen: tla.Constexpr,
    dense: tla.Constexpr,
    indexed: tla.Constexpr,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    sequence_major: tla.Constexpr,
    dense_mask: tla.Constexpr,
    vector_score_mod: tla.Constexpr = None,
    ragged: tla.Constexpr = True,
    identity_score: tla.Constexpr = False,
    group_size: tla.Constexpr = 4,
    mask_kind: tla.Constexpr = 0,
    features: tla.Constexpr = (),
    aux_cache_meta: tla.Constexpr = (),
    inline_quantize: tla.Constexpr = None,
    post_quant: tla.Constexpr = False,
    exact: tla.Constexpr = False,
    arenas=(),
    channels=(),
    repair_only: tla.Constexpr = False,
    generic: tla.Constexpr = False,
):
    # Head counts remain code-generation features; lengths and task geometry
    # are always read from this launch's runtime arguments.
    q_heads, kv_heads = features.q_heads, features.kv_heads
    dim, want_lse = (128, True)
    hq, hk, nq, nk, mh = (q_heads, kv_heads, max_q, max_k, mask_heads)
    ring = group_size + 2
    stream_output = features.stream_output
    bounded_quantize = features.bounded_inline or len(aux_tensors) > 0
    paired_heads = (
        sequence_major[0] and sequence_major[1] and (hq == hk) and (hq % 2 == 0)
        if tla.const_expr(exact)
        else features.paired_heads
    )
    if tla.const_expr(repair_only):
        paired_heads = features.paired_heads
    if tla.const_expr(exact):
        cache_roles = _mx_aux_cache_roles(
            mask_mod, aux_tensors, aux_scalars, aux_cache_meta, any(varlen)
        )
    compact_aux = len(aux_cache_meta) > 0
    tiled_aux = compact_aux and aux_cache_meta[0][0] < 0
    aux_layouts = _mx_aux_layouts(aux_cache_meta, aux_scalars)
    global_aux = any(varlen) or _mx_mask_depends_on_query(mask_mod, aux_tensors, aux_scalars)
    if tla.const_expr(len(features.key_aux_scope) > 0):
        global_aux = features.key_aux_scope[0]
    score_stride = 8192 if tla.const_expr(exact) else 8192 if compact_aux else 16384
    inverse_ub_offset = 57344 if compact_aux else 153600
    mask_ub_offset = (
        (212224 if generic else 91904)
        if tla.const_expr(exact)
        else 60416
        if compact_aux
        else 186368
    )
    vring, bm, bn = 2, 128, 128
    dv = dim + 32
    load_kdim = 256 if paired_heads else dim
    load_vdim = 256 if paired_heads else dv
    pv_m = dim + 32 if post_quant else dim
    half, fp8, e8m0 = 64, tla.Float8E4M3FN, tla.Float8E8M0
    base_l1_bytes = 93696 + ring * bm * 4 + ring * bm * bn + ring * 1024
    if tla.const_expr(len(arenas) > 0):
        arena_l1, arena_l0a, arena_l0b, arena_l0c, arena_ub = arenas
    else:
        arena_l1 = tla.allocate(base_l1_bytes + 236544, tla.UInt8, tla.AddressSpace.l1, 512)
        arena_l0a = tla.allocate(57344, tla.UInt8, tla.AddressSpace.l0a, 512)
        arena_l0b = tla.allocate(49152, tla.UInt8, tla.AddressSpace.l0b, 512)
        arena_l0c = tla.allocate(262144, tla.UInt8, tla.AddressSpace.l0c, 512)
        arena_ub = tla.allocate(
            252928
            if exact and not generic or inline_quantize is not None or 0 < len(aux_cache_meta) <= 4
            else 219136,
            tla.UInt8,
            tla.AddressSpace.ub,
            512,
        )
    q1p = tla.recast_ptr(arena_l1 + 0, dtype=fp8)
    k1p = tla.recast_ptr(arena_l1 + 16384, dtype=fp8)
    v1p = tla.recast_ptr(arena_l1 + 49152, dtype=fp8)
    p1p = tla.recast_ptr(arena_l1 + 93184 + ring * bm * 4, dtype=fp8)
    qs1p = tla.recast_ptr(arena_l1 + 90112, dtype=e8m0)
    ks1p = tla.recast_ptr(arena_l1 + 90624, dtype=e8m0)
    vs1p = tla.recast_ptr(arena_l1 + 91648, dtype=e8m0)
    ps1p = tla.recast_ptr(arena_l1 + 93184, dtype=e8m0)
    if tla.const_expr(paired_heads):
        k1p = tla.recast_ptr(arena_l1 + base_l1_bytes, dtype=fp8)
        v1p = tla.recast_ptr(arena_l1 + base_l1_bytes + 131072, dtype=fp8)
        ks1p = tla.recast_ptr(arena_l1 + base_l1_bytes + 229376, dtype=e8m0)
        vs1p = tla.recast_ptr(arena_l1 + base_l1_bytes + 233472, dtype=e8m0)
    inverse_l1 = arena_l1 + 93696 + ring * bm * 4 + ring * bm * bn
    q0p = tla.recast_ptr(arena_l0b + 0, dtype=fp8)
    k0p = tla.recast_ptr(arena_l0a + 0, dtype=fp8)
    p0p = tla.recast_ptr(arena_l0b + 16384, dtype=fp8)
    v0p = tla.recast_ptr(arena_l0a + 16384, dtype=fp8)
    s0p = tla.recast_ptr(arena_l0c + 0, dtype=tla.Float32)
    o0p = tla.recast_ptr(arena_l0c + 131072, dtype=tla.Float32)
    sp = tla.recast_ptr(
        arena_ub if tla.const_expr(exact) else arena_ub + 0,
        dtype=tla.Float32 if tla.const_expr(exact) else tla.Float16,
    )
    pp_stride = 8192 if tla.const_expr(exact) else 8192 if compact_aux else 32768
    pp = tla.recast_ptr(
        arena_ub + 65536 if tla.const_expr(exact) else arena_ub + (32768 if compact_aux else 65536),
        dtype=tla.UInt8,
    )
    psp = tla.recast_ptr(
        arena_ub + 81920
        if tla.const_expr(exact)
        else arena_ub + (53760 if compact_aux else 106752),
        dtype=tla.UInt8,
    )
    mp = tla.recast_ptr(
        arena_ub + 82432 if tla.const_expr(exact) else arena_ub + (49152 if compact_aux else 74240),
        dtype=tla.Float32,
    )
    lp = tla.recast_ptr(
        arena_ub + 82432 + ring * 256
        if tla.const_expr(exact)
        else arena_ub + (49152 if compact_aux else 74240) + ring * 256,
        dtype=tla.Float32,
    )
    np = tla.recast_ptr(
        arena_ub + 86528
        if tla.const_expr(exact)
        else arena_ub + (49152 if compact_aux else 74240) + ring * 512,
        dtype=tla.Float32,
    )
    if tla.const_expr(exact):
        badp = tla.recast_ptr(arena_ub + 90368, dtype=tla.Float32)
    else:
        gp = tla.recast_ptr(arena_ub + (54272 if compact_aux else 112384), dtype=tla.Float32)
    op = tla.recast_ptr(
        arena_ub + (92416 if generic else 100352) if tla.const_expr(exact) else arena_ub + 116736,
        dtype=tla.Float32,
    )
    ap = tla.recast_ptr(
        arena_ub + (157952 if generic else 165888) if tla.const_expr(exact) else arena_ub + 153600,
        dtype=tla.Float32,
    )
    ep = tla.recast_ptr(
        arena_ub + (209152 if generic else 217088) if tla.const_expr(exact) else arena_ub + 215296,
        dtype=tla.Float32,
    )
    guardp = tla.recast_ptr(
        arena_ub + 217344 if tla.const_expr(exact) else arena_ub + 215552, dtype=tla.Int32
    )
    agg = tla.recast_ptr(
        arena_ub + 217600 if tla.const_expr(exact) else arena_ub + 215808, dtype=tla.Int32
    )
    if tla.const_expr(exact):
        gp = tla.recast_ptr(arena_ub + 217856, dtype=tla.Float32)
        exact_inv = tla.recast_ptr(arena_ub + (209664 if generic else 218368), dtype=tla.Float32)
        exact_safe = tla.recast_ptr(arena_ub + (210688 if generic else 219392), dtype=tla.Float32)
        exact_gmax = tla.recast_ptr(arena_ub + (210944 if generic else 219648), dtype=tla.Float32)
        label_cache = tla.recast_ptr(arena_ub + 233472, dtype=tla.Int32)
    q1 = tla.make_tensor(q1p, tla.make_layout_with_tag((load_kdim, bm), fp8, tla.arch.nZ))
    k1 = tla.make_tensor(k1p, tla.make_layout_with_tag((bn, load_kdim), fp8, tla.arch.zN))
    v1 = tla.make_tensor(v1p, tla.make_layout_with_tag((load_vdim, bn), fp8, tla.arch.nZ))
    p1 = tla.make_tensor(p1p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.zN))
    qs1 = tla.make_tensor(
        qs1p, tla.make_layout_with_tag((load_kdim // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    ks1 = tla.make_tensor(
        ks1p, tla.make_layout_with_tag((bn, load_kdim // 32), e8m0, tla.arch.zZMxScale)
    )
    vs1 = tla.make_tensor(
        vs1p, tla.make_layout_with_tag((load_vdim, bn // 32), e8m0, tla.arch.zZMxScale)
    )
    ps1 = tla.make_tensor(ps1p, tla.make_layout_with_tag((bn // 32, bm), e8m0, tla.arch.nNMxScale))
    q0 = tla.make_tensor(q0p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k0 = tla.make_tensor(k0p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    p0 = tla.make_tensor(p0p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.nZ))
    v0 = tla.make_tensor(v0p, tla.make_layout_with_tag((pv_m, bn), fp8, tla.arch.zN))
    s0 = tla.make_tensor(s0p, tla.make_layout_with_tag((bn, bm), tla.Float32, tla.arch.L0Clayout))
    o0 = tla.make_tensor(o0p, tla.make_layout_with_tag((pv_m, bm), tla.Float32, tla.arch.L0Clayout))
    padp = tla.recast_ptr(arena_ub + 216064, dtype=tla.UInt8)
    pads = tla.recast_ptr(arena_ub + 218112, dtype=tla.UInt8)
    padref = tla.make_tensor(
        tla.recast_ptr(padp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(64, 16), tla.make_stride(16, 1)),
    )
    padu = tla.make_tensor_like(padref.ptr, padref, tla.arch.zNUnAlign)
    pad1 = tla.make_tensor(
        tla.recast_ptr(v1p + dim * bn, dtype=tla.Float16),
        tla.make_layout_with_tag((128, 16), tla.Float16, tla.arch.zN),
    )
    padsref = tla.make_tensor(
        tla.recast_ptr(pads, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(2, 16), tla.make_stride(16, 1)),
    )
    padsu = tla.make_tensor_like(padsref.ptr, padsref, tla.arch.zNUnAlign)
    pads1 = tla.make_tensor(
        tla.recast_ptr(vs1p + dim * 4, dtype=tla.Float16),
        tla.make_layout_with_tag((4, 16), tla.Float16, tla.arch.zN),
    )
    dma = (
        channels[0]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("dma", tla.arch.MTE2, tla.arch.MTE1)
    )
    load = (
        channels[1]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("load", tla.arch.MTE1, tla.arch.CUBE)
    )
    vdma = (
        channels[2]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("vdma", tla.arch.MTE2, tla.arch.MTE1)
    )
    vload = (
        channels[3]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("vload", tla.arch.MTE1, tla.arch.CUBE)
    )
    ofree = (
        channels[4]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("ofree", tla.arch.FIX, tla.arch.CUBE)
    )
    if tla.const_expr(exact):
        ofree_second = tla.flag("exact_ofree_second", tla.arch.FIX, tla.arch.CUBE)
        pvfree0 = tla.cross_flag("exact_pv_free0")
        pvfree1 = tla.cross_flag("exact_pv_free1")
        pvready0 = tla.cross_flag("exact_pv_ready0")
        pvready1 = tla.cross_flag("exact_pv_ready1")
        exact_split = tla.params.CopyL0C2DstParams(
            unit_flag=3, l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N
        )
    kfree0 = (
        channels[5]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("kfree0", tla.arch.CUBE, tla.arch.MTE1)
    )
    kfree1 = (
        channels[6]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("kfree1", tla.arch.CUBE, tla.arch.MTE1)
    )
    vfree = (
        channels[7]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("vfree", tla.arch.CUBE, tla.arch.MTE1)
    )
    vfree_second = (
        channels[8]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("vfree_second", tla.arch.CUBE, tla.arch.MTE1)
    )
    k1free0 = (
        channels[9]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("k1free0", tla.arch.MTE1, tla.arch.MTE2)
    )
    k1free1 = (
        channels[10]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("k1free1", tla.arch.MTE1, tla.arch.MTE2)
    )
    v1free0 = (
        channels[11]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("v1free0", tla.arch.MTE1, tla.arch.MTE2)
    )
    v1free1 = (
        channels[12]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("v1free1", tla.arch.MTE1, tla.arch.MTE2)
    )
    vshortfree = (
        channels[13]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("vshortfree", tla.arch.MTE1, tla.arch.MTE2)
    )
    k1free2 = (
        channels[14]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("k1free2", tla.arch.MTE1, tla.arch.MTE2)
    )
    k1free3 = (
        channels[15]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("k1free3", tla.arch.MTE1, tla.arch.MTE2)
    )
    if tla.const_expr(exact):
        exact_aux_free = tla.flag("exact_aux_free", tla.arch.VECTOR, tla.arch.MTE2)
    aux_ready = (
        channels[16]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("aux_ready", tla.arch.MTE2, tla.arch.VECTOR)
    )
    aux_cleared = (
        channels[17]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("aux_cleared", tla.arch.VECTOR, tla.arch.MTE2)
    )
    mask_done = (
        channels[18]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("mask_done", tla.arch.VECTOR, tla.arch.SCALAR)
    )
    vstore = (
        channels[19]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("vstore", tla.arch.VECTOR, tla.arch.MTE3)
    )
    stored = (
        channels[20]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("stored", tla.arch.MTE3, tla.arch.VECTOR)
    )
    ppfree0 = (
        channels[21]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("ppfree0", tla.arch.MTE3, tla.arch.VECTOR)
    )
    ppfree1 = (
        channels[22]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("ppfree1", tla.arch.MTE3, tla.arch.VECTOR)
    )
    yfree0 = (
        channels[23]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("yfree0", tla.arch.MTE3, tla.arch.VECTOR)
    )
    yfree1 = (
        channels[24]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("yfree1", tla.arch.MTE3, tla.arch.VECTOR)
    )
    sr = channels[25] if tla.const_expr(len(channels) > 0) else tla.cross_flag("score_ready")
    sr_second = (
        channels[26] if tla.const_expr(len(channels) > 0) else tla.cross_flag("score_ready_second")
    )
    ore = channels[27] if tla.const_expr(len(channels) > 0) else tla.cross_flag("output_ready")
    psready = channels[28] if tla.const_expr(len(channels) > 0) else tla.cross_flag("scales_ready")
    pfree = channels[29] if tla.const_expr(len(channels) > 0) else tla.cross_flag("prob_consumed")
    padready = channels[30] if tla.const_expr(len(channels) > 0) else tla.cross_flag("pad_ready")
    denominator_ready = ore
    inverse_written = padready
    inverse_ready = (
        channels[31]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("inverse_ready", tla.arch.MTE1, tla.arch.FIX)
    )
    pv_finished = (
        channels[32]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("pv_finished", tla.arch.CUBE, tla.arch.FIX)
    )
    denominator_ptr = tla.recast_ptr(
        arena_ub + (69632 if compact_aux else 116736), dtype=tla.Float32
    )
    denominator_tile = tla.make_tensor(
        denominator_ptr, tla.make_layout(tla.make_shape(32, 128), tla.make_stride(64, 1))
    )
    if tla.const_expr(not exact):
        repair_done = tla.cross_flag("repair_done")
        repair_go = tla.cross_flag("repair_go")
    split_o = tla.params.CopyL0C2DstParams(unit_flag=0, l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N)
    full_cached_aux = ()
    for preload_i in tla.range_constexpr(len(aux_cache_meta)):
        preload_offset, preload_shape, preload_strides, preload_wide = aux_layouts[preload_i]
        preload_words = 2 if preload_wide else 1
        preload_type = tla.Int32 if preload_wide else aux_tensors[preload_i].ptr.dtype
        if tla.const_expr(tiled_aux):
            preload_ptr = tla.recast_ptr(
                arena_ub + (221184 - preload_offset - 1)
                if tla.const_expr(exact)
                else arena_ub + (81920 - preload_offset - 1),
                dtype=preload_type,
            )
            preload_tiles = ()
            for preload_slot in tla.range_constexpr(3):
                preload_ub = tla.make_tensor(
                    preload_ptr + preload_slot * 128 * preload_words,
                    tla.make_layout(
                        tla.make_shape(1, 128 * preload_words),
                        tla.make_stride(128 * preload_words, 1),
                    ),
                )
                preload_tiles = preload_tiles + (preload_ub,)
            with tla.vector():
                mx_zero_ub(
                    tla.recast_ptr(preload_ptr, dtype=tla.UInt8), 3 * 128 * preload_words * 4
                )
            full_cached_aux = full_cached_aux + (preload_tiles,)
        else:
            preload_size = _mx_aux_numel(preload_shape) * preload_words
            preload_ub = tla.make_tensor(
                tla.recast_ptr(arena_ub + 81920 + preload_offset, dtype=preload_type),
                tla.make_layout(tla.make_shape(1, preload_size), tla.make_stride(preload_size, 1)),
            )
            preload_gm = tla.make_tensor(
                tla.recast_ptr(aux_tensors[preload_i].ptr, dtype=preload_type),
                tla.make_layout(tla.make_shape(1, preload_size), tla.make_stride(preload_size, 1)),
            )
            with tla.vector():
                tla.copy(preload_ub, preload_gm)
            full_cached_aux = full_cached_aux + (preload_ub,)
    with tla.vector():
        if tla.const_expr(tiled_aux):
            tla.set_flag(aux_cleared)
        elif tla.const_expr(compact_aux):
            tla.set_flag(aux_ready)
        if tla.const_expr(post_quant):
            if tla.const_expr(ragged or paired_heads):
                mx_zero_ub(pp, 7168 if paired_heads else 2 * bn * (dim // 32) + vring * dv * 4)
            initialize_short_pipeline(agg, gp)
            mx_pad(padp, pads)
            tla.set_flag(vstore)
            tla.wait_flag(vstore)
            if tla.arch.sub_block_idx() == 0:
                if tla.const_expr(ragged):
                    mx_zero_scale_l1(qs1p, pp, bm * (load_kdim // 32))
                    if tla.const_expr(not paired_heads):
                        mx_zero_scale_l1(ks1p, pp, 2 * bn * (dim // 32) + vring * dv * 4)
                if tla.const_expr(paired_heads):
                    mx_zero_scale_l1(ks1p, pp, 7168)
                mx_zero_scale_l1(ps1p, tla.recast_ptr(gp, dtype=tla.UInt8), bm * 4)
                for seed_slot in tla.range_constexpr(vring):
                    seed_v1 = tla.make_tensor_like(
                        tla.recast_ptr(v1p + seed_slot * bn * dv + dim * bn, dtype=tla.Float16),
                        pad1,
                        tla.arch.zN,
                    )
                    seed_vs1 = tla.make_tensor_like(
                        tla.recast_ptr(vs1p + seed_slot * dv * 4 + dim * 4, dtype=tla.Float16),
                        pads1,
                        tla.arch.zN,
                    )
                    for seed_sub in tla.range_constexpr(2):
                        tla.copy(
                            tla.tile_view(
                                seed_v1, tla.make_shape(64, 16), tla.make_coord(seed_sub, 0)
                            ),
                            padu,
                        )
                        tla.copy(
                            tla.tile_view(
                                seed_vs1, tla.make_shape(2, 16), tla.make_coord(seed_sub, 0)
                            ),
                            padsu,
                        )
        else:
            initialize_short_state(agg)
        if tla.const_expr(exact and features.periodic_prefix):
            # Every query tile starts at a multiple of 128. The first vector
            # half never exposes keys 64..127; keep both P slots zero there.
            if tla.arch.sub_block_idx() == 0:
                mx_zero_ub(pp + 4096, 4096)
                mx_zero_ub(pp + pp_stride + 4096, 4096)
        _mx_slot(None, (ppfree0, ppfree1, yfree0, yfree1), None, False)
        if tla.const_expr(post_quant):
            tla.cross_core_set_flag(padready, tla.arch.MTE3)
        tla.cross_core_set_flag(sr, tla.arch.VECTOR)
        tla.cross_core_set_flag(sr_second, tla.arch.VECTOR)
        if tla.const_expr(exact):
            tla.set_flag(exact_aux_free)
            tla.cross_core_set_flag(pvfree0, tla.arch.VECTOR)
            tla.cross_core_set_flag(pvfree1, tla.arch.VECTOR)
    with tla.cube():
        if tla.const_expr(post_quant):
            if tla.const_expr(ragged or paired_heads):
                tla.cross_core_wait_flag(padready, tla.arch.MTE2)
        else:
            if tla.const_expr(ragged):
                _mx_fill_scales(
                    tla.recast_ptr(qs1p, dtype=tla.UInt8),
                    tla.Int32(bm * (load_kdim // 32)),
                    tla.Int32(0),
                )
                if tla.const_expr(not paired_heads):
                    _mx_fill_scales(
                        tla.recast_ptr(ks1p, dtype=tla.UInt8),
                        tla.Int32(2 * bn * (dim // 32) + vring * dv * 4),
                        tla.Int32(0),
                    )
            if tla.const_expr(paired_heads):
                _mx_fill_scales(
                    tla.recast_ptr(ks1p, dtype=tla.UInt8), tla.Int32(7168), tla.Int32(0)
                )
            _mx_fill_scales(
                tla.recast_ptr(ps1p, dtype=tla.UInt8), tla.Int32(bm * 4), tla.Int32(31868)
            )
            _mx_fill_ready()
        tla.set_flag(ofree)
        if tla.const_expr(exact):
            tla.set_flag(ofree_second)
        _mx_slot(None, (kfree0, kfree1, vfree, vfree_second, k1free0, k1free1), None, False)
        if tla.const_expr(paired_heads):
            tla.set_flag(k1free2)
            tla.set_flag(k1free3)
        _mx_slot(None, (v1free0, v1free1, vshortfree), None, False)
    q1free = (
        channels[33]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("q1free", tla.arch.MTE1, tla.arch.MTE2)
    )
    qfree = (
        channels[34]
        if tla.const_expr(len(channels) > 0)
        else tla.flag("qfree", tla.arch.CUBE, tla.arch.MTE1)
    )
    with tla.cube():
        tla.set_flag(q1free)
        tla.set_flag(qfree)
    q_stride = hq * dim if tla.const_expr(sequence_major[0]) else dim
    k_stride = hk * dim if tla.const_expr(sequence_major[1]) else dim
    qs_stride = hq * 4 if tla.const_expr(sequence_major[0]) else 4
    ks_stride = hk * 4 if tla.const_expr(sequence_major[1]) else 4
    vs_stride = hk * 256 if tla.const_expr(sequence_major[1]) else 256
    q_major = (
        sequence_major[0]
        and features.query_major
        and _mx_mask_depends_on_query(mask_mod, aux_tensors, aux_scalars)
    )
    blocks = q_blocks
    total = batch * hq * blocks
    guard_capacity = total
    single_sequence = features.single_sequence
    if tla.const_expr(varlen[0]):
        packed_blocks = tla.Int32(0)
        if tla.const_expr(single_sequence):
            packed_blocks = (total_q + 127) // 128
        else:
            for sequence in tla.range(batch):
                packed_blocks += (cu_q[sequence + 1] - cu_q[sequence] + 127) // 128
        total = packed_blocks * hq
    core = tla.arch.block_idx()
    cores = tla.arch.block_num()
    task_count = total // cores + tla.Int32(core < total % cores)
    delay = min(2 if tla.const_expr(exact) else 6, ring - 2)
    if tla.const_expr(not exact):
        if tla.const_expr(features.short_delay):
            delay = min(delay, 2 if mask_kind == 2 else 1 if compact_aux else 3)
    sub = tla.arch.sub_block_idx()
    qbase, kbase, maskrow = (tla.Int32(0), tla.Int32(0), tla.Int32(0))
    pqbase, pkbase, vb, pmaskrow = (tla.Int32(0), tla.Int32(0), tla.Int32(0), tla.Int32(0))
    qpc, ppc, qcount, pcount = (tla.Int32(0), tla.Int32(0), tla.Int32(1), tla.Int32(1))
    qfirst, pfirst = tla.Int32(0), tla.Int32(0)
    plsebase, ptask = tla.Int32(0), tla.Int32(0)
    rows, valid = (tla.Int32(128), tla.Int32(64))
    b, h, qstart = (tla.Int32(0), tla.Int32(0), tla.Int32(0))
    qtile, ptile = (tla.Int32(0), tla.Int32(0))
    qordinal, pordinal, qcursor, pcursor = (tla.Int32(0),) * 4
    qlen, pqlen, klen, pklen = tla.Int32(nq), tla.Int32(nq), tla.Int32(nk), tla.Int32(nk)
    qoff, koff = (tla.Int32(0), tla.Int32(0))
    qtag, ktags, vtags = tla.Int32(-1), (tla.Int32(-1),) * 4, (tla.Int32(-1),) * 3
    knext, vnext = (tla.Int32(0), tla.Int32(0))
    q_sequence, q_prefix, q_scales = (tla.Int32(0), tla.Int32(0), tla.Int32(0))
    p_sequence, p_prefix, p_scales = (tla.Int32(0), tla.Int32(0), tla.Int32(0))
    label_geometry_ready = tla.Bool(False)
    label_rows, label_cols, tag_i = tla.Int32(-1), tla.Int32(-1), 0
    aux_qtags = ()
    aux_ktags = ()
    for tag_i in tla.range_constexpr(len(aux_cache_meta)):
        aux_qtags = aux_qtags + (tla.Int32(-1),)
        aux_ktags = aux_ktags + (tla.Int32(-1),)
    step = tla.Int32(0)
    while pordinal < task_count:
        pi = step - delay
        if qordinal < task_count and qtile == 0:
            if tla.const_expr(repair_only):
                task, qcursor, q_sequence, q_prefix, q_scales, found = _mx_repair_task(
                    qcursor,
                    v_scale_offsets,
                    cu_q,
                    cu_k,
                    hq,
                    blocks,
                    q_sequence,
                    q_prefix,
                    q_scales,
                    core,
                    cores,
                    total,
                    paired_heads,
                    varlen[0],
                    False,
                    q_major,
                    total_q if tla.const_expr(single_sequence) else None,
                )
                task_count = task_count if found else qordinal
            else:
                task, q_sequence, q_prefix, q_scales = _mx_advance_task(
                    _mx_balanced_task(qordinal, core, cores, total, paired_heads),
                    cu_q,
                    cu_k,
                    hq,
                    blocks,
                    q_sequence,
                    q_prefix,
                    q_scales,
                    varlen[0],
                    False,
                    q_major,
                    total_q if tla.const_expr(single_sequence) else None,
                )
            b, h, kh, qstart, qbase, kbase, qlen, klen, qoff, koff, maskrow, qpc, qcount, qfirst = (
                _mx_task_geometry(
                    task,
                    cu_q,
                    cu_k,
                    partial_count,
                    full_count,
                    nq,
                    nk,
                    hq,
                    hk,
                    blocks,
                    mask_batch,
                    mh,
                    total_q,
                    varlen,
                    sequence_major,
                    dense,
                    dense_mask,
                    features.dense_bounds,
                    single_sequence,
                    total_k,
                )
            )
            rows = min(128, qlen - qstart) if tla.const_expr(ragged) else 128
        if pi >= 0 and ptile == 0:
            if tla.const_expr(repair_only):
                ptask, pcursor, p_sequence, p_prefix, p_scales, found = _mx_repair_task(
                    pcursor,
                    v_scale_offsets,
                    cu_q,
                    cu_k,
                    hq,
                    blocks,
                    p_sequence,
                    p_prefix,
                    p_scales,
                    core,
                    cores,
                    total,
                    paired_heads,
                    varlen[0],
                    varlen[1],
                    q_major,
                    total_q if tla.const_expr(single_sequence) else None,
                )
            else:
                ptask, p_sequence, p_prefix, p_scales = _mx_advance_task(
                    _mx_balanced_task(pordinal, core, cores, total, paired_heads),
                    cu_q,
                    cu_k,
                    hq,
                    blocks,
                    p_sequence,
                    p_prefix,
                    p_scales,
                    varlen[0],
                    varlen[1],
                    q_major,
                    total_q if tla.const_expr(single_sequence) else None,
                )
            (
                pb,
                ph,
                pkh,
                pqstart,
                pqbase,
                pkbase,
                pqlen,
                pklen,
                pqoff,
                pkoff,
                pmaskrow,
                ppc,
                pcount,
                pfirst,
            ) = _mx_task_geometry(
                ptask,
                cu_q,
                cu_k,
                partial_count,
                full_count,
                nq,
                nk,
                hq,
                hk,
                blocks,
                mask_batch,
                mh,
                total_q,
                varlen,
                sequence_major,
                dense,
                dense_mask,
                features.dense_bounds,
                single_sequence,
                total_k,
            )
            vb = pb * ((nk + 63) // 64) * hk * 256 + pkh * (
                256 if tla.const_expr(sequence_major[1]) else (nk + 63) // 64 * 256
            )
            plsebase = (pb * hq + ph) * nq + pqstart
            if tla.const_expr(varlen[0]):
                plsebase = ph * total_q + pqoff + pqstart
            if tla.const_expr(varlen[1]):
                vb = p_scales * vs_stride + pkh * (
                    256 if tla.const_expr(sequence_major[1]) else v_scale_capacity * 256
                )
            valid = min(64, max(0, pqlen - pqstart - sub * 64)) if tla.const_expr(ragged) else 64
        qhead = tla.Int32(0)
        reuse_q = tla.Bool(False)
        if tla.const_expr(paired_heads):
            qhead = h & 1
        pair_qbase = qbase - qhead * dim
        if qordinal < task_count and qtile == 0:
            if tla.const_expr(paired_heads):
                reuse_q = pair_qbase == qtag
                qtag = pair_qbase
        qmp = mp + qordinal % ring * half
        pmp = mp + pordinal % ring * half
        qlp = lp + qordinal % ring * half
        plp = lp + pordinal % ring * half
        if qordinal < task_count:
            if qtile == 0:
                with tla.cube():
                    tla.wait_flag(qfree)
                    tla.wait_flag(q1free)
                    if not reuse_q:
                        if rows < 128:
                            q1.fill(0)
                            _mx_fill_ready()
                        qgm = tla.make_tensor(
                            q.ptr + pair_qbase,
                            tla.make_layout(
                                tla.make_shape(load_kdim, rows),
                                tla.make_stride(1, q_stride),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                q1, tla.make_shape(load_kdim, rows), tla.make_coord(0, 0)
                            ),
                            qgm,
                        )
                        qsgm = q_scale_gm(
                            qs.ptr + pair_qbase // 32, rows, qs_stride, load_kdim // 32
                        )
                        tla.copy(
                            tla.tile_view(
                                qs1, tla.make_shape(load_kdim // 32, rows), tla.make_coord(0, 0)
                            ),
                            qsgm,
                        )
                        tla.set_flag(dma)
                        tla.wait_flag(dma)
                    tla.copy(
                        q0,
                        tla.get_tile(q1, tla.make_coord(qhead * dim, 0), tla.make_shape(dim, bm)),
                        scale=tla.get_tile(
                            qs1,
                            tla.make_coord(qhead * (dim // 32), 0),
                            tla.make_shape(dim // 32, bm),
                        ),
                    )
                    tla.set_flag(q1free)
                    tla.set_flag(load)
                    tla.wait_flag(load)
        qcols, vcols = (tla.Int32(0), tla.Int32(0))
        qstart_k, vstart_k = (tla.Int32(0), tla.Int32(0))
        if qordinal < task_count and qtile < qcount:
            qblock = qfirst + qtile
            if tla.const_expr(not dense):
                qblock = (
                    partial_index[maskrow, qtile]
                    if qtile < qpc
                    else full_index[maskrow, qtile - qpc]
                )
            qstart_k = qblock * bn
            qcols = min(128, max(0, klen - qstart_k)) if tla.const_expr(ragged) else 128
        if pi >= 0 and ptile < pcount:
            pblock = pfirst + ptile
            if tla.const_expr(not dense):
                pblock = (
                    partial_index[pmaskrow, ptile]
                    if ptile < ppc
                    else full_index[pmaskrow, ptile - ppc]
                )
            vstart_k = pblock * bn
            vcols = min(128, max(0, pklen - vstart_k)) if tla.const_expr(ragged) else 128
        if tla.const_expr(exact):
            reload_q = ()
            reload_k = ()
            next_qtags = ()
            next_ktags = ()
            for tag_i in tla.range_constexpr(len(aux_cache_meta)):
                tag_offset, tag_shape, tag_strides, tag_wide = aux_layouts[tag_i]
                tag_prefix = tla.Int32(0)
                if tla.const_expr(len(tag_shape) >= 2):
                    tag_prefix = b % tag_shape[0] * tag_strides[0]
                if tla.const_expr(len(tag_shape) == 3):
                    tag_prefix += h % tag_shape[1] * tag_strides[1]
                tag_qoffset = (
                    qoff if tla.const_expr(len(tag_shape) == 1 and any(varlen)) else tla.Int32(0)
                )
                tag_koffset = (
                    koff if tla.const_expr(len(tag_shape) == 1 and any(varlen)) else tla.Int32(0)
                )
                tag_qaddress = (tag_prefix + tag_qoffset + qstart) * (2 if tag_wide else 1)
                tag_kaddress = (tag_prefix + tag_koffset + qstart_k) * (2 if tag_wide else 1)
                tag_q = aux_qtags[tag_i]
                tag_k = aux_ktags[tag_i]
                copy_q = tla.Bool(False)
                copy_k = tla.Bool(False)
                if qordinal < task_count and qcols > 0 and (qtile < qpc):
                    if tla.const_expr(cache_roles[tag_i][0]):
                        if qtile == 0 and tag_qaddress != tag_q:
                            copy_q = tla.Bool(True)
                            tag_q = tag_qaddress
                    if tla.const_expr(cache_roles[tag_i][1]):
                        if tag_kaddress != tag_k:
                            copy_k = tla.Bool(True)
                            tag_k = tag_kaddress
                reload_q = reload_q + (copy_q,)
                reload_k = reload_k + (copy_k,)
                next_qtags = next_qtags + (tag_q,)
                next_ktags = next_ktags + (tag_k,)
            if tla.const_expr(len(aux_cache_meta) > 0):
                aux_qtags = next_qtags
                aux_ktags = next_ktags
            reuse_labels = tla.Bool(False)
            if qordinal < task_count and qcols > 0 and (qtile < qpc):
                changed_labels = tla.Bool(False)
                for tag_i in tla.range_constexpr(len(aux_cache_meta)):
                    changed_labels = changed_labels or reload_q[tag_i] or reload_k[tag_i]
                reuse_labels = (
                    label_geometry_ready
                    and (not changed_labels)
                    and (label_rows == rows)
                    and (label_cols == qcols)
                )
                label_geometry_ready = tla.Bool(True)
                label_rows, label_cols = (rows, qcols)
        kslot, vslot = (tla.Int32(step % 2), tla.Int32(pi % 2))
        reuse_k, reuse_v = (tla.Bool(False), tla.Bool(False))
        khead, vhead = (tla.Int32(0), tla.Int32(0))
        address = tla.Int32(0)
        if tla.const_expr(paired_heads):
            khead = h & 1
            vhead = ptask // blocks % hq & 1
            if qordinal < task_count:
                address = kbase - khead * dim + qstart_k * k_stride
                kslot, reuse_k, ktags, knext = _mx_l1_cache(address, qcols > 0, ktags, knext)
            if pi >= 0:
                address = pkbase - vhead * dim + vstart_k * k_stride
                vslot, reuse_v, vtags, vnext = _mx_l1_cache(address, vcols > 0, vtags, vnext)
        with tla.cube():
            if qordinal < task_count:
                _mx_slot(
                    kslot,
                    (k1free0, k1free1, k1free2, k1free3)[: 4 if paired_heads else 2],
                    None,
                    True,
                )
                current_k1 = tla.make_tensor_like(k1p + kslot * bn * load_kdim, k1, tla.arch.zN)
                current_ks1 = tla.make_tensor_like(
                    ks1p + kslot * bn * (load_kdim // 32), ks1, tla.arch.zZMxScale
                )
                if not reuse_k:
                    if qcols < bn:
                        current_k1.fill(0)
                        _mx_fill_ready()
                    if qcols > 0:
                        k_copy_cols = qcols
                        kgm = tla.make_tensor(
                            k.ptr + (kbase - khead * dim + qstart_k * k_stride),
                            tla.make_layout(
                                tla.make_shape(k_copy_cols, load_kdim), tla.make_stride(k_stride, 1)
                            ),
                        )
                        ksgm = k_scale_gm(
                            ks.ptr + (kbase // 32 - khead * 4 + qstart_k * ks_stride),
                            k_copy_cols,
                            ks_stride,
                            load_kdim // 32,
                        )
                        tla.copy(
                            tla.tile_view(
                                current_k1,
                                tla.make_shape(k_copy_cols, load_kdim),
                                tla.make_coord(0, 0),
                            ),
                            kgm,
                        )
                        tla.copy(
                            tla.tile_view(
                                current_ks1,
                                tla.make_shape(k_copy_cols, load_kdim // 32),
                                tla.make_coord(0, 0),
                            ),
                            ksgm,
                        )
                tla.set_flag(dma)
            if pi >= 0:
                _mx_slot(
                    vslot, (v1free0, v1free1, vshortfree)[: 3 if paired_heads else 2], None, True
                )
                preload_vlong = tla.make_tensor_like(v1p + vslot * bn * load_vdim, v1, tla.arch.nZ)
                preload_vslong = tla.make_tensor_like(
                    vs1p + vslot * load_vdim * 4, vs1, tla.arch.zZMxScale
                )
                if not reuse_v:
                    if vcols < bn:
                        tla.get_tile(
                            preload_vlong,
                            tla.make_coord(0, 0),
                            tla.make_shape(256 if paired_heads else dim, bn),
                        ).fill(0)
                        if tla.const_expr(not dense and not ragged and not paired_heads):
                            # Empty aligned sparse tiles still run PV; seed their E8M0 scales.
                            if vcols == 0:
                                _mx_fill_scales(
                                    tla.recast_ptr(preload_vslong.ptr, dtype=tla.UInt8),
                                    tla.Int32(load_vdim * 4),
                                    tla.Int32(0x7F7F),
                                )
                        _mx_fill_ready()
                    if vcols > 0:
                        v_copy_cols = vcols
                        long_vgm = tla.make_tensor(
                            v.ptr + (pkbase - vhead * dim + vstart_k * k_stride),
                            tla.make_layout(
                                tla.make_shape(256 if paired_heads else dim, v_copy_cols),
                                tla.make_stride(1, k_stride),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                        long_groups = (v_copy_cols + 63) // 64 * 2
                        long_vsgm = v_scale_gm(
                            vs.ptr + (vb - vhead * 256 + vstart_k // 64 * vs_stride),
                            long_groups,
                            vs_stride,
                            256 if paired_heads else dim,
                        )
                        tla.copy(
                            tla.tile_view(
                                preload_vlong,
                                tla.make_shape(256 if paired_heads else dim, v_copy_cols),
                                tla.make_coord(0, 0),
                            ),
                            long_vgm,
                        )
                        tla.copy(
                            tla.tile_view(
                                preload_vslong,
                                tla.make_shape(256 if paired_heads else dim, long_groups),
                                tla.make_coord(0, 0),
                            ),
                            long_vsgm,
                        )
                tla.set_flag(vdma)
            if qordinal < task_count:
                tla.wait_flag(dma)
                tla.wait_flag(kfree0)
                current_k0 = tla.make_tensor_like(k0p, k0, tla.arch.zN)
                read_k1 = tla.make_tensor_like(k1p + kslot * bn * load_kdim, k1, tla.arch.zN)
                read_ks1 = tla.make_tensor_like(
                    ks1p + kslot * bn * (load_kdim // 32), ks1, tla.arch.zZMxScale
                )
                tla.copy(
                    current_k0,
                    tla.get_tile(read_k1, tla.make_coord(0, khead * dim), tla.make_shape(bn, dim)),
                    scale=tla.get_tile(
                        read_ks1, tla.make_coord(0, khead * 4), tla.make_shape(bn, 4)
                    ),
                )
                _mx_slot(
                    kslot,
                    (k1free0, k1free1, k1free2, k1free3)[: 4 if paired_heads else 2],
                    None,
                    False,
                )
                tla.set_flag(load)
                tla.wait_flag(load)
                current_s0 = tla.make_tensor_like(s0p + step % 2 * bm * bn, s0, tla.arch.L0Clayout)
                tla.mmad_mx(current_s0, current_k0, q0, init_c=True, unit_flag=3)
                tla.set_flag(kfree0)
                if qtile == max(1, qcount) - 1:
                    tla.set_flag(qfree)
            if pi >= 0:
                tla.wait_flag(vdma)
                _mx_slot(pi % 2, (vfree, vfree_second), None, True)
                pv_v0 = tla.make_tensor_like(v0p + pi % 2 * bn * dv, v0, tla.arch.zN)
                pv_p0 = tla.make_tensor_like(p0p + pi % 2 * bm * bn, p0, tla.arch.nZ)
                read_v1 = tla.make_tensor_like(v1p + vslot * bn * load_vdim, v1, tla.arch.nZ)
                read_vs1 = tla.make_tensor_like(
                    vs1p + vslot * load_vdim * 4, vs1, tla.arch.zZMxScale
                )
                read_p1 = tla.make_tensor_like(p1p + pi % ring * bm * bn, p1, tla.arch.zN)
                chosen_ps1p = ps1p + pi % ring * 512 if tla.const_expr(exact) else ps1p
                read_ps1 = tla.make_tensor_like(chosen_ps1p, ps1, tla.arch.nNMxScale)
                if tla.const_expr(post_quant and (not ragged) and (not paired_heads)):
                    if pi == 0:
                        tla.cross_core_wait_flag(padready, tla.arch.MTE1)
                tla.copy(
                    pv_v0,
                    tla.get_tile(read_v1, tla.make_coord(vhead * dim, 0), tla.make_shape(pv_m, bn)),
                    scale=tla.get_tile(
                        read_vs1, tla.make_coord(vhead * dim, 0), tla.make_shape(pv_m, bn // 32)
                    ),
                )
                _mx_slot(
                    vslot, (v1free0, v1free1, vshortfree)[: 3 if paired_heads else 2], None, False
                )
                tla.cross_core_wait_flag(psready, tla.arch.MTE1)
                if tla.const_expr(exact):
                    tla.copy(pv_p0, read_p1, scale=read_ps1)
                else:
                    if pi < 2:
                        tla.copy(pv_p0, read_p1, scale=read_ps1)
                    else:
                        tla.copy(pv_p0, read_p1)
                    if tla.const_expr(not post_quant):
                        if ptile == max(1, pcount) - 1:
                            tla.set_flag(inverse_ready)
                tla.cross_core_set_flag(pfree, tla.arch.MTE1)
                tla.set_flag(vload)
                tla.wait_flag(vload)
                if tla.const_expr(not exact):
                    if ptile == 0:
                        tla.wait_flag(ofree)
                    if ptile == max(1, pcount) - 1:
                        if post_quant or min(128, pqlen - ptask % blocks * 128) < 128:
                            tla.mmad_mx(o0, pv_v0, pv_p0, init_c=ptile == 0)
                            tla.set_flag(pv_finished)
                        else:
                            tla.mmad_mx(o0, pv_v0, pv_p0, init_c=ptile == 0, unit_flag=3)
                    else:
                        tla.mmad_mx(o0, pv_v0, pv_p0, init_c=ptile == 0)
                if pi % 2 == 0:
                    if tla.const_expr(exact):
                        tla.wait_flag(ofree)
                    else:
                        tla.set_flag(vfree)
                elif tla.const_expr(exact):
                    tla.wait_flag(ofree_second)
                else:
                    tla.set_flag(vfree_second)
                if tla.const_expr(exact):
                    exact_o0 = tla.make_tensor_like(o0p + pi % 2 * bm * bn, o0, tla.arch.L0Clayout)
                    tla.mmad_mx(exact_o0, pv_v0, pv_p0, init_c=True, unit_flag=3)
                    _mx_slot(pi % 2, (vfree, vfree_second), None, False)
            if qordinal < task_count:
                _mx_slot(step % 2, (sr, sr_second), tla.arch.FIX, True)
                if tla.const_expr(exact):
                    exact_su = tla.make_tensor(
                        sp + step % 2 * score_stride,
                        tla.make_layout(tla.make_shape(bn, bm), tla.make_stride(half, 1)),
                    )
                    exact_sc = tla.make_tensor_like(
                        s0p + step % 2 * bm * bn, s0, tla.arch.L0Clayout
                    )
                    tla.copy(exact_su, exact_sc, exact_split)
                else:
                    mx_score_unit(sp + step % 2 * score_stride, s0p + step % 2 * bm * bn, scale)
                _mx_slot(step % 2, (sr, sr_second), tla.arch.FIX, False)
            if pi >= 0:
                if tla.const_expr(exact):
                    _mx_slot(pi % 2, (pvfree0, pvfree1), tla.arch.FIX, True)
                    exact_pvu = tla.make_tensor(
                        op + pi % 2 * 8192,
                        tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1)),
                    )
                    exact_pvc = tla.make_tensor_like(o0p + pi % 2 * bm * bn, o0, tla.arch.L0Clayout)
                    tla.copy(exact_pvu, exact_pvc, exact_split)
                    if pi % 2 == 0:
                        tla.set_flag(ofree)
                        tla.cross_core_set_flag(pvready0, tla.arch.FIX)
                    else:
                        tla.set_flag(ofree_second)
                        tla.cross_core_set_flag(pvready1, tla.arch.FIX)
                elif ptile == max(1, pcount) - 1:
                    if tla.const_expr(post_quant):
                        tla.wait_flag(pv_finished)
                        tla.copy(
                            denominator_tile,
                            tla.get_tile(o0, tla.make_coord(128, 0), tla.make_shape(32, 128)),
                            params=split_o,
                        )
                        tla.cross_core_set_flag(denominator_ready, tla.arch.FIX)
                        tla.cross_core_wait_flag(inverse_written, tla.arch.FIX)
                        mx_output_normalized(
                            out.ptr + pqbase,
                            o0p,
                            inverse_l1 + pordinal % ring * 1024,
                            min(128, pqlen - ptask % blocks * 128),
                            q_stride,
                            tla.Int32(stream_output),
                            tla.Int32(0),
                            tla.Int32(160),
                        )
                    else:
                        if min(128, pqlen - ptask % blocks * 128) < 128:
                            tla.wait_flag(pv_finished)
                        tla.wait_flag(inverse_ready)
                        mx_output_normalized(
                            out.ptr + pqbase,
                            o0p,
                            inverse_l1 + pordinal % ring * 1024,
                            min(128, pqlen - ptask % blocks * 128),
                            q_stride,
                            tla.Int32(stream_output),
                            tla.Int32(0)
                            if min(128, pqlen - ptask % blocks * 128) < 128
                            else tla.Int32(3),
                            tla.Int32(128),
                        )
                    tla.set_flag(ofree)
        single_mask_visible = tla.Int32(-1)
        single_halves = tla.Int32(0)
        with tla.vector():
            if step == 0:
                if tla.const_expr(tiled_aux):
                    tla.wait_flag(aux_cleared)
                elif tla.const_expr(compact_aux):
                    tla.wait_flag(aux_ready)
            if qordinal < task_count:
                if tla.const_expr(exact):
                    tla.wait_flag(exact_aux_free)
                    cached_aux = ()
                    for aux_i in tla.range_constexpr(len(aux_cache_meta)):
                        cached_aux = cached_aux + (
                            _mx_stage_aux(
                                aux_tensors[aux_i],
                                full_cached_aux[aux_i],
                                aux_layouts[aux_i],
                                b,
                                h,
                                qstart,
                                qstart_k,
                                qstart_k,
                                rows,
                                qcols,
                                qcols,
                                qoff,
                                koff,
                                any(varlen),
                                qcols > 0 and qtile < qpc and reload_q[aux_i],
                                qcols > 0 and qtile < qpc and reload_k[aux_i],
                                False,
                                tiled_aux,
                                global_aux,
                            ),
                        )
                    tla.set_flag(aux_ready)
                    tla.wait_flag(aux_ready)
                elif tla.const_expr(len(features.key_aux) > 0):
                    if qcols > 0:
                        _mx_stage_aux(
                            aux_tensors[0],
                            full_cached_aux[0],
                            aux_layouts[0],
                            b,
                            h,
                            qstart,
                            qstart_k,
                            qstart_k,
                            rows,
                            qcols,
                            qcols,
                            qoff,
                            koff,
                            any(varlen),
                            False,
                            True,
                            False,
                            tiled_aux,
                            global_aux,
                        )
                        if tla.const_expr(tiled_aux):
                            tla.set_flag(aux_ready)
                            tla.wait_flag(aux_ready)
                elif tla.const_expr(mask_kind > 0 and inline_quantize is None):
                    if qcols > 0 and qtile < qpc and (qtile % 2 == 0):
                        mask0 = tla.make_tensor(
                            tla.recast_ptr(arena_ub + mask_ub_offset, dtype=tla.Int32),
                            tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                        )
                        mask1 = tla.make_tensor(
                            tla.recast_ptr(arena_ub + mask_ub_offset + 4096, dtype=tla.Int32),
                            tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                        )
                        cache_next_k = qstart_k
                        if qtile + 1 < qpc:
                            if tla.const_expr(dense):
                                cache_next_k = qstart_k + 128
                            else:
                                cache_next_k = partial_index[maskrow, qtile + 1] * 128
                        cached_aux = ()
                        for aux_i in tla.range_constexpr(len(aux_tensors)):
                            if tla.const_expr(compact_aux):
                                cached_aux = cached_aux + (
                                    _mx_stage_aux(
                                        aux_tensors[aux_i],
                                        full_cached_aux[aux_i],
                                        aux_layouts[aux_i],
                                        b,
                                        h,
                                        qstart,
                                        qstart_k,
                                        cache_next_k,
                                        rows,
                                        qcols,
                                        min(128, klen - cache_next_k),
                                        qoff,
                                        koff,
                                        any(varlen),
                                        qtile == 0,
                                        True,
                                        qtile + 1 < qpc,
                                        tiled_aux,
                                        global_aux,
                                    ),
                                )
                            else:
                                cached_aux = cached_aux + (aux_tensors[aux_i],)
                        if tla.const_expr(tiled_aux):
                            tla.set_flag(aux_ready)
                            tla.wait_flag(aux_ready)
                        uniform_code = tla.Int32(0)
                        label_equality = _mx_trace_label_equality(
                            mask_mod,
                            b,
                            h,
                            sub,
                            qstart_k,
                            cached_aux,
                            aux_scalars,
                            (qlen, klen, qoff, koff),
                        )
                        if tla.const_expr(label_equality is not None):
                            if qcount == 1 and qcols == 128 and (rows == 128):
                                mx_label_probe(
                                    label_equality.q,
                                    label_equality.k,
                                    guardp,
                                    tla.Int32(label_equality.wide),
                                )
                                tla.set_flag(mask_done)
                                tla.wait_flag(mask_done)
                                uniform_flag = tla.make_tensor(
                                    guardp, tla.make_layout(tla.make_shape(1), tla.make_stride(1))
                                )
                                uniform_code = uniform_flag[0]
                        if uniform_code > 0:
                            if tla.const_expr(label_equality is not None):
                                single_halves = uniform_code >> 10
                                if single_halves == 0:
                                    mx_label_uniform(
                                        label_equality.q,
                                        label_equality.k,
                                        mask0.ptr,
                                        tla.Int32(label_equality.wide),
                                    )
                                single_mask_visible = uniform_code >> 1 & 255
                        elif qtile + 1 < qpc:
                            next_k = cache_next_k
                            if tla.const_expr(mask_kind == 2):
                                if rows == 128 and qcols == 128 and (next_k + 128 <= klen):
                                    _mx_cache_packed_pair(
                                        mask_mod,
                                        b,
                                        h,
                                        qstart,
                                        sub,
                                        qstart_k,
                                        next_k,
                                        mask0,
                                        mask1,
                                        cached_aux,
                                        aux_scalars,
                                        (qlen, klen, qoff, koff),
                                    )
                                else:
                                    _cache_mask_pair(
                                        mask_mod,
                                        b,
                                        h,
                                        qstart,
                                        sub,
                                        qstart_k,
                                        next_k,
                                        mask0,
                                        mask1,
                                        cached_aux,
                                        aux_scalars,
                                        (qlen, klen, qoff, koff),
                                        False,
                                        True,
                                    )
                            elif tla.const_expr(len(aux_cache_meta) > 0):
                                _mx_cache_aux_pair(
                                    mask_mod,
                                    b,
                                    h,
                                    qstart,
                                    sub,
                                    qstart_k,
                                    next_k,
                                    mask0,
                                    mask1,
                                    cached_aux,
                                    aux_scalars,
                                    (qlen, klen, qoff, koff),
                                    not ragged,
                                    True,
                                )
                            else:
                                _cache_mask_pair(
                                    mask_mod,
                                    b,
                                    h,
                                    qstart,
                                    sub,
                                    qstart_k,
                                    next_k,
                                    mask0,
                                    mask1,
                                    cached_aux,
                                    aux_scalars,
                                    (qlen, klen, qoff, koff),
                                    not ragged,
                                    mask_kind == 2,
                                )
                        elif tla.const_expr(mask_kind == 2):
                            if rows == 128 and qcols == 128:
                                _mx_cache_packed_pair(
                                    mask_mod,
                                    b,
                                    h,
                                    qstart,
                                    sub,
                                    qstart_k,
                                    qstart_k,
                                    mask0,
                                    mask1,
                                    cached_aux,
                                    aux_scalars,
                                    (qlen, klen, qoff, koff),
                                )
                            else:
                                _cache_mask(
                                    mask_mod,
                                    b,
                                    h,
                                    qstart,
                                    sub,
                                    qstart_k,
                                    mask0,
                                    cached_aux,
                                    aux_scalars,
                                    (qlen, klen, qoff, koff),
                                    False,
                                )
                        elif tla.const_expr(len(aux_cache_meta) > 0):
                            _mx_cache_aux_pair(
                                mask_mod,
                                b,
                                h,
                                qstart,
                                sub,
                                qstart_k,
                                qstart_k,
                                mask0,
                                mask0,
                                cached_aux,
                                aux_scalars,
                                (qlen, klen, qoff, koff),
                                not ragged,
                                False,
                            )
                        else:
                            _cache_mask(
                                mask_mod,
                                b,
                                h,
                                qstart,
                                sub,
                                qstart_k,
                                mask0,
                                cached_aux,
                                aux_scalars,
                                (qlen, klen, qoff, koff),
                                not ragged,
                            )
                        tla.set_flag(mask_done)
                        tla.wait_flag(mask_done)
                _mx_slot(step % 2, (sr, sr_second), tla.arch.VECTOR, True)
                _mx_slot(step % 2, (ppfree0, ppfree1), None, True)
                if tla.const_expr(exact):
                    if tla.const_expr(features.periodic_prefix):
                        if qtile < qpc and qcols > 0:
                            _mx_exact_periodic_prefix(
                                sp + step % 2 * score_stride,
                                pp + step % 2 * pp_stride,
                                psp + step % 2 * 256,
                                qmp,
                                qlp,
                                np + step % ring * half,
                                badp + qordinal % ring * half,
                                scale,
                                max(0, min(64, qlen - qstart - sub * 64)),
                                qcols,
                                tla.Int32(qtile == 0),
                                (qstart + sub * 64) % 128,
                            )
                        else:
                            _mx_exact_probabilities_generic(
                                sp + step % 2 * score_stride,
                                pp + step % 2 * pp_stride,
                                psp + step % 2 * 256,
                                qmp,
                                qlp,
                                np + step % ring * half,
                                badp + qordinal % ring * half,
                                exact_inv,
                                exact_safe,
                                exact_gmax,
                                scale,
                                qcols,
                                False,
                                qtile == 0,
                                mask_mod,
                                b,
                                h,
                                qstart,
                                sub,
                                qstart_k,
                                cached_aux,
                                aux_scalars,
                                (qlen, klen, qoff, koff),
                                False,
                            )
                    elif tla.const_expr(generic):
                        _mx_callback_probabilities(
                            sp + step % 2 * score_stride,
                            pp + step % 2 * pp_stride,
                            psp + step % 2 * 256,
                            qmp,
                            qlp,
                            np + step % ring * half,
                            badp + qordinal % ring * half,
                            exact_inv,
                            arena_ub + mask_ub_offset,
                            scale,
                            b,
                            h,
                            qstart + sub * 64,
                            qstart_k,
                            qcols,
                            qtile >= qpc if tla.const_expr(not dense) else not dense_mask,
                            qtile == 0,
                            score_mod,
                            mask_mod,
                            vector_score_mod,
                            indexed,
                            ragged,
                            aux_tensors,
                            aux_scalars,
                            (qlen, klen, qoff, koff),
                        )
                    else:
                        if rows == 128 and qcols == 128:
                            _mx_exact_probabilities(
                                sp + step % 2 * score_stride,
                                pp + step % 2 * pp_stride,
                                psp + step % 2 * 256,
                                qmp,
                                qlp,
                                np + step % ring * half,
                                badp + qordinal % ring * half,
                                exact_inv,
                                exact_safe,
                                exact_gmax,
                                scale,
                                qcols,
                                qtile < qpc,
                                qtile == 0,
                                mask_mod,
                                b,
                                h,
                                qstart,
                                sub,
                                qstart_k,
                                cached_aux,
                                aux_scalars,
                                (qlen, klen, qoff, koff),
                                True,
                                label_cache,
                                reuse_labels,
                            )
                        else:
                            _mx_exact_probabilities(
                                sp + step % 2 * score_stride,
                                pp + step % 2 * pp_stride,
                                psp + step % 2 * 256,
                                qmp,
                                qlp,
                                np + step % ring * half,
                                badp + qordinal % ring * half,
                                exact_inv,
                                exact_safe,
                                exact_gmax,
                                scale,
                                qcols,
                                qtile < qpc,
                                qtile == 0,
                                mask_mod,
                                b,
                                h,
                                qstart,
                                sub,
                                qstart_k,
                                cached_aux,
                                aux_scalars,
                                (qlen, klen, qoff, koff),
                                False,
                                label_cache,
                                reuse_labels,
                            )
                    tla.set_flag(exact_aux_free)
                else:
                    if qtile == 0 and qcols == 0:
                        with tla.vec.func():
                            tla.make_tensor(
                                qmp, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                            ).store(tla.full(0.0, tla.Float32))
                            tla.make_tensor(
                                qlp, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                            ).store(tla.full(0.0, tla.Float32))
                    if qcols > 0:
                        if tla.const_expr(len(features.key_aux) > 0):
                            key_ptr = (
                                full_cached_aux[0][1].ptr
                                if tla.const_expr(tiled_aux)
                                else full_cached_aux[0].ptr + qstart_k
                            )
                            _mx_quantize_key_aux(
                                sp + step % 2 * score_stride,
                                pp + step % 2 * pp_stride,
                                qmp,
                                qlp,
                                qtile,
                                qcols,
                                tla.recast_ptr(key_ptr, dtype=tla.Int32),
                                np + qordinal % ring * half,
                                tla.Int32(qcount == 1),
                                qstart_k,
                                klen,
                            )
                        elif qcount == 1:
                            if tla.const_expr(inline_quantize is not None):
                                if qtile < qpc:
                                    _mx_quantize_masked_inline(
                                        sp + step % 2 * score_stride,
                                        pp + step % 2 * pp_stride,
                                        qmp,
                                        qlp,
                                        tla.Int32(0),
                                        qstart + sub * 64,
                                        qstart_k,
                                        qlen,
                                        klen,
                                        tla.recast_ptr(qmp, dtype=tla.Int32),
                                        inline_quantize,
                                        post_quant,
                                        np + qordinal % ring * half,
                                        True,
                                    )
                                else:
                                    _mx_short_quantize(
                                        sp + step % 2 * score_stride,
                                        pp + step % 2 * pp_stride,
                                        qmp,
                                        qlp,
                                        tla.Int32(0),
                                        qcols,
                                        tla.recast_ptr(arena_ub + mask_ub_offset, dtype=tla.Int32),
                                        np + qordinal % ring * half,
                                        tla.Int32(0),
                                        tla.Int32(0),
                                        tla.Int32(-1),
                                        tla.Int32(128),
                                        post_quant,
                                    )
                            else:
                                needs_maximum = tla.Int32(qcols < 32)
                                if tla.const_expr(mask_kind > 0):
                                    if qtile < qpc:
                                        if single_mask_visible >= 0:
                                            needs_maximum = tla.Int32(single_mask_visible < 32)
                                        else:
                                            mask_has_32_visible(
                                                tla.recast_ptr(
                                                    arena_ub + mask_ub_offset, dtype=tla.Int32
                                                )
                                                + sub * 64 * 8,
                                                guardp,
                                            )
                                            tla.set_flag(mask_done)
                                            tla.wait_flag(mask_done)
                                            probe = tla.make_tensor(
                                                guardp,
                                                tla.make_layout(
                                                    tla.make_shape(1), tla.make_stride(1)
                                                ),
                                            )
                                            needs_maximum = tla.Int32(probe[0] != 286331153)
                                if single_halves > 0:
                                    _mx_short_quantize(
                                        sp + step % 2 * score_stride,
                                        pp + step % 2 * pp_stride,
                                        qmp,
                                        qlp,
                                        tla.Int32(0),
                                        tla.Int32(128),
                                        tla.recast_ptr(qmp, dtype=tla.Int32),
                                        np + qordinal % ring * half,
                                        tla.Int32(0),
                                        tla.Int32(0),
                                        tla.Int32(64) if single_halves == 2 else tla.Int32(0),
                                        tla.Int32(64) if single_halves == 1 else tla.Int32(128),
                                        post_quant,
                                    )
                                else:
                                    _mx_short_quantize(
                                        sp + step % 2 * score_stride,
                                        pp + step % 2 * pp_stride,
                                        qmp,
                                        qlp,
                                        tla.Int32(0),
                                        qcols,
                                        tla.recast_ptr(arena_ub + mask_ub_offset, dtype=tla.Int32),
                                        np + qordinal % ring * half,
                                        tla.Int32(qtile < qpc)
                                        if tla.const_expr(mask_kind > 0)
                                        else tla.Int32(0),
                                        needs_maximum,
                                        tla.Int32(-1),
                                        tla.Int32(128),
                                        post_quant,
                                    )
                        elif tla.const_expr(mask_kind > 0):
                            if qtile < qpc:
                                mask0 = tla.make_tensor(
                                    tla.recast_ptr(arena_ub + mask_ub_offset, dtype=tla.Int32),
                                    tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                                )
                                mask1 = tla.make_tensor(
                                    tla.recast_ptr(
                                        arena_ub + mask_ub_offset + 4096, dtype=tla.Int32
                                    ),
                                    tla.make_layout(tla.make_shape(128, 8), tla.make_stride(8, 1)),
                                )
                                visible_ptr = mask0.ptr + qtile % 2 * 1024
                                _mx_quantize_masked_inline(
                                    sp + step % 2 * score_stride,
                                    pp + step % 2 * pp_stride,
                                    qmp,
                                    qlp,
                                    qtile,
                                    qstart + sub * 64,
                                    qstart_k,
                                    qlen,
                                    klen,
                                    visible_ptr,
                                    inline_quantize,
                                    post_quant,
                                )
                            else:
                                _mx_short_quantize(
                                    sp + step % 2 * score_stride,
                                    pp + step % 2 * pp_stride,
                                    qmp,
                                    qlp,
                                    qtile,
                                    qcols,
                                    tla.recast_ptr(qmp, dtype=tla.Int32),
                                    qmp,
                                    tla.Int32(0),
                                    tla.Int32(-1),
                                    tla.Int32(-1),
                                    tla.Int32(128),
                                    post_quant,
                                )
                        else:
                            _mx_short_quantize(
                                sp + step % 2 * score_stride,
                                pp + step % 2 * pp_stride,
                                qmp,
                                qlp,
                                qtile,
                                qcols,
                                tla.recast_ptr(qmp, dtype=tla.Int32),
                                qmp,
                                tla.Int32(0),
                                tla.Int32(-1),
                                tla.Int32(-1),
                                tla.Int32(128),
                                post_quant,
                            )
                    else:
                        mx_zero_ub(pp + step % 2 * pp_stride, half * bn)
                    if tla.const_expr(not post_quant):
                        if qtile == max(1, qcount) - 1:
                            short_statistics(
                                ap,
                                qmp,
                                arena_ub + inverse_ub_offset + qordinal % ring * 512,
                                np + qordinal % ring * half,
                                tla.recast_ptr(qmp, dtype=tla.Int32),
                                agg,
                                qcount,
                                qlp,
                                np + qordinal % ring * half,
                                min(64, max(0, rows - sub * 64)),
                                tla.Int32(bounded_quantize),
                            )
                _mx_slot(step % 2, (sr, sr_second), tla.arch.VECTOR, False)
            if pi >= 0:
                tla.cross_core_wait_flag(pfree, tla.arch.MTE3)
            if qordinal < task_count:
                tla.set_flag(vstore)
                tla.wait_flag(vstore)
                if tla.const_expr(exact):
                    probability_to_l1(
                        p1p + step % ring * bm * bn,
                        ps1p + step % ring * 512,
                        pp + step % 2 * pp_stride,
                        psp + step % 2 * 256,
                        sub,
                    )
                else:
                    mx_prob64_to_l1(
                        p1p + step % ring * bm * bn + sub * half * bn, pp + step % 2 * pp_stride
                    )
                    if tla.const_expr(not post_quant):
                        if qtile == max(1, qcount) - 1:
                            mx_zero_scale_l1(
                                inverse_l1 + qordinal % ring * 1024 + sub * 512,
                                arena_ub + inverse_ub_offset + qordinal % ring * 512,
                                512,
                            )
                _mx_slot(step % 2, (ppfree0, ppfree1), None, False)
                tla.cross_core_set_flag(psready, tla.arch.MTE3)
            if pi >= 0:
                if tla.const_expr(exact):
                    _mx_slot(pi % 2, (pvready0, pvready1), tla.arch.VECTOR, True)
                if ptile == max(1, pcount) - 1:
                    if tla.const_expr(exact):
                        _mx_slot(0 if generic else pi % 2, (yfree0, yfree1), None, True)
                        exact_y = tla.recast_ptr(
                            arena_ub + (190720 if generic else 198656 + pi % 2 * 35840),
                            dtype=tla.BFloat16,
                        )
                        exact_e = ep + (0 if generic else pi % 2) * 64
                        _mx_exact_finish(
                            ap,
                            op + pi % 2 * 8192,
                            np + pi % ring * half,
                            ptile == 0,
                            pmp,
                            plp,
                            badp + pordinal % ring * half,
                            exact_e,
                            exact_y,
                            out,
                            lse,
                            pqbase + sub * half * q_stride,
                            plsebase + sub * half,
                            q_stride,
                            valid,
                            stream_output,
                        )
                        _mx_slot(0 if generic else pi % 2, (yfree0, yfree1), None, False)
                    elif ptile == max(1, pcount) - 1:
                        if tla.const_expr(post_quant):
                            tla.cross_core_wait_flag(denominator_ready, tla.arch.VECTOR)
                            short_statistics(
                                ap,
                                pmp,
                                arena_ub + inverse_ub_offset + pordinal % ring * 512,
                                np + pordinal % ring * half,
                                tla.recast_ptr(pmp, dtype=tla.Int32),
                                agg,
                                pcount,
                                denominator_ptr,
                                np + pordinal % ring * half,
                                valid,
                                tla.Int32(bounded_quantize),
                            )
                            tla.set_flag(vstore)
                            tla.wait_flag(vstore)
                            mx_zero_scale_l1(
                                inverse_l1 + pordinal % ring * 1024 + sub * 512,
                                arena_ub + inverse_ub_offset + pordinal % ring * 512,
                                512,
                            )
                            tla.cross_core_set_flag(inverse_written, tla.arch.MTE3)
                        current_eu = tla.make_tensor(
                            np + pordinal % ring * half,
                            tla.make_layout(tla.make_shape(half), tla.make_stride(1)),
                        )
                        tla.set_flag(vstore)
                        tla.wait_flag(vstore)
                        if valid > 0 and want_lse:
                            egm = tla.make_tensor(
                                lse.ptr + plsebase + sub * half,
                                tla.make_layout(tla.make_shape(valid), tla.make_stride(1)),
                            )
                            tla.copy(
                                egm,
                                tla.tile_view(current_eu, tla.make_shape(valid), tla.make_coord(0)),
                            )
                        guard_gm = tla.make_tensor(
                            v_scale_offsets.ptr + ptask * 16 + sub * 8,
                            tla.make_layout(tla.make_shape(1), tla.make_stride(1)),
                        )
                        guard_ub = tla.make_tensor(
                            tla.recast_ptr(pmp, dtype=tla.Int32),
                            tla.make_layout(tla.make_shape(1), tla.make_stride(1)),
                        )
                        tla.copy(guard_gm, guard_ub)
                elif tla.const_expr(exact):
                    _mx_exact_accumulate(ap, op + pi % 2 * 8192, np + pi % ring * half, ptile == 0)
                if tla.const_expr(exact):
                    _mx_slot(pi % 2, (pvfree0, pvfree1), tla.arch.VECTOR, False)
        if qordinal < task_count:
            qtile += 1
            if qtile == max(1, qcount):
                qtile = tla.Int32(0)
                qordinal += 1
        if pi >= 0:
            ptile += 1
            if ptile == max(1, pcount):
                ptile = tla.Int32(0)
                pordinal += 1
        step += 1
    with tla.cube():
        _mx_slot(None, (qfree, q1free, ofree), None, True)
        if tla.const_expr(exact):
            tla.wait_flag(ofree_second)
            tla.cross_core_wait_flag(pvfree0, tla.arch.FIX)
            tla.cross_core_wait_flag(pvfree1, tla.arch.FIX)
        _mx_slot(None, (kfree0, kfree1, vfree, vfree_second, k1free0, k1free1), None, True)
        if tla.const_expr(paired_heads):
            tla.wait_flag(k1free2)
            tla.wait_flag(k1free3)
        _mx_slot(None, (v1free0, v1free1, vshortfree), None, True)
        tla.cross_core_wait_flag(sr, tla.arch.FIX)
        tla.cross_core_wait_flag(sr_second, tla.arch.FIX)
    with tla.vector():
        if tla.const_expr(exact):
            tla.wait_flag(exact_aux_free)
        _mx_slot(None, (ppfree0, ppfree1, yfree0, yfree1), None, True)
    repair_needed = tla.Int32(0)
    if tla.const_expr(not exact):
        with tla.vector():
            tla.set_flag(vstore)
            tla.wait_flag(vstore)
            aggregate_gm = tla.make_tensor(
                v_scale_offsets.ptr + (guard_capacity + core) * 16 + sub * 8,
                tla.make_layout(tla.make_shape(1), tla.make_stride(1)),
            )
            aggregate_ub = tla.make_tensor(
                agg, tla.make_layout(tla.make_shape(1), tla.make_stride(1))
            )
            tla.copy(aggregate_gm, aggregate_ub)
            tla.set_flag(stored)
            tla.wait_flag(stored)
        with tla.vector():
            tla.set_flag(stored)
            tla.wait_flag(stored)
            tla.cross_core_set_flag(repair_done, tla.arch.MTE3)
            tla.cross_core_wait_flag(repair_go, tla.arch.SCALAR)
            if tla.const_expr(not dense or any(varlen)):
                _mx_refresh_guard(v_scale_offsets.ptr)
        with tla.cube():
            tla.cross_core_wait_flag(repair_done, tla.arch.SCALAR)
            tla.cross_core_set_flag(repair_go, tla.arch.SCALAR)
            if tla.const_expr(not dense or any(varlen)):
                _mx_refresh_guard(v_scale_offsets.ptr)
        repair_needed = tla.Int32(
            v_scale_offsets[(guard_capacity + core) * 16] > 0
            or v_scale_offsets[(guard_capacity + core) * 16 + 8] > 0
        )
    return (
        repair_needed,
        (arena_l1, arena_l0a, arena_l0b, arena_l0c, arena_ub),
        (
            dma,
            load,
            vdma,
            vload,
            ofree,
            kfree0,
            kfree1,
            vfree,
            vfree_second,
            k1free0,
            k1free1,
            v1free0,
            v1free1,
            vshortfree,
            k1free2,
            k1free3,
            aux_ready,
            aux_cleared,
            mask_done,
            vstore,
            stored,
            ppfree0,
            ppfree1,
            yfree0,
            yfree1,
            sr,
            sr_second,
            ore,
            psready,
            pfree,
            padready,
            inverse_ready,
            pv_finished,
            q1free,
            qfree,
        ),
    )


@tla.jit
def _mx_masked_group_max_aux(
    scores,
    scale,
    groupmax,
    mask_mod: tla.Constexpr,
    b,
    h,
    qstart,
    sub,
    kstart,
    aux,
    scalars,
    info,
    checked: tla.Constexpr,
    first_key,
    last_key,
):
    """Scale/mask FP32 scores and save all group32 maxima in the same pass."""
    query = _mx_query_vector(qstart, sub)
    values = _mx_load_query_aux(aux, sub)
    query_valid = tla.cmp(query.value, info[0], mode="lt")
    neginf = tla.full(float("-inf"), tla.Float32)
    blockm = neginf
    full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    for group in tla.range_constexpr(4):
        maxima = (neginf,) * 8
        begin = max(0, min(4, (first_key - group * 32) // 8))
        end = max(0, min(4, (last_key - group * 32 + 7) // 8))
        for subgroup in tla.range(begin, end):
            for lane in tla.range_constexpr(8):
                row = group * 32 + subgroup * 8 + lane
                item = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(row, 0))
                key = kstart + row
                visible = _mx_apply_vector_mask(
                    mask_mod, b, h, query, key, aux, values, scalars, info, 1
                )
                if tla.const_expr(not checked):
                    visible = tla.bitwise_and(visible, query_valid)
                    visible = tla.bitwise_and(
                        visible, tla.cmp(tla.add(tla.full(0, tla.Int32), key), info[1], mode="lt")
                    )
                z = tla.where(visible, item.load() * scale, neginf)
                item.store(z, mask=full)
                maxima = maxima[:lane] + (tla.max(maxima[lane], z),) + maxima[lane + 1 :]
        maximum = tla.max(
            tla.max(tla.max(maxima[0], maxima[1]), tla.max(maxima[2], maxima[3])),
            tla.max(tla.max(maxima[4], maxima[5]), tla.max(maxima[6], maxima[7])),
        )
        tla.tile_view(groupmax, tla.make_shape(1, 64), tla.make_coord(group, 0)).store(maximum)
        blockm = tla.max(blockm, maximum)
    return blockm


_CPP_MX_SCALE = (
    _MX_CPP_HEADER
    + r"""#ifndef FLEX_MX_SCALE_DEFINED
#define FLEX_MX_SCALE_DEFINED
constexpr CastTrait mx_scale_cast={RegLayout::ZERO,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
__simd_callee__ inline void mx_group_scale(RegTensor<float>& maximum,RegTensor<float>& sm,RegTensor<float>& inv,RegTensor<int32_t>& encoded) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),condition;
 RegTensor<float> p,zero;Duplicate(zero,0.f);
 FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(p,maximum,sm,mf);
 Compares<float,CMPMODE::LT>(condition,p,__builtin_inff(),mf);Select(p,p,zero,condition);
 RegTensor<int32_t> power,mantissa,one,none,increment,mask,invbits;
 Duplicate(one,int32_t(1));Duplicate(none,int32_t(0));Duplicate(mask,int32_t(0x7fffff));
 ShiftRights(power,reinterpret_cast<RegTensor<int32_t>&>(p),int16_t(23),mf);Adds(power,power,int32_t(-135),mf);
 And(mantissa,reinterpret_cast<RegTensor<int32_t>&>(p),mask,mf);
 Compares<int32_t,CMPMODE::GT>(condition,mantissa,int32_t(0x600000),mf);Select(increment,one,none,condition);Add(power,power,increment,mf);
 Maxs(power,power,int32_t(-127),mf);Mins(power,power,int32_t(127),mf);
 Compares<float,CMPMODE::EQ>(condition,p,0.f,mf);Select(power,none,power,condition);
 Duplicate(invbits,int32_t(127));Sub(invbits,invbits,power,mf);ShiftLefts(invbits,invbits,int16_t(23),mf);inv=reinterpret_cast<RegTensor<float>&>(invbits);
 Adds(encoded,power,int32_t(127),mf);
}
__simd_callee__ inline void mx_scale_pair(__ubuf__ uint8_t* dst,RegTensor<int32_t>& lo,RegTensor<int32_t>& hi) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>();uint32_t count=64;MaskReg mh=UpdateMask<int16_t>(count);
 RegTensor<int32_t> joined;RegTensor<int16_t> words,packed,unused;
 ShiftLefts(joined,hi,int16_t(8),mf);Or(joined,joined,lo,mf);
 Cast<int16_t,int32_t,mx_scale_cast>(words,joined,mf);DeInterleave(packed,unused,words,words);
 StoreAlign<int16_t,DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ int16_t*)dst,packed,2,mh);
}
#endif
"""
)

_MX_LABEL_SOURCE = (
    _CPP_MX_SCALE
    + r"""constexpr CastTrait r0={RegLayout::ZERO,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait r1={RegLayout::ONE,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait r2={RegLayout::TWO,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait r3={RegLayout::THREE,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
template<bool WIDE,bool FULL>
__simd_callee__ inline void label_max(RegTensor<float>& result,__ubuf__ float* src,
 __ubuf__ uint32_t* keys,RegTensor<uint32_t>& qlo,RegTensor<uint32_t>& qhi,
 MaskReg& valid,int32_t start,int32_t cols,float scale) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),visible,hi;
 RegTensor<float> m0,m1,m2,m3,negative,x;
 RegTensor<uint32_t> k;
 Duplicate(negative,-__builtin_inff());m0=negative;m1=negative;m2=negative;m3=negative;
 #pragma unroll 2
 for(uint16_t j=0;j<32;j+=4) {
#define LABEL_MAX(LANE, MAXIMUM) \
  LoadAlign(x,src+(start+j+LANE)*64);Muls(x,x,scale,mf); \
  LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(k,keys+(start+j+LANE)*(WIDE?2:1)); \
  Compare<uint32_t,CMPMODE::EQ>(visible,qlo,k,mf); \
  if constexpr(WIDE) {LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(k,keys+(start+j+LANE)*2+1);Compare<uint32_t,CMPMODE::EQ>(hi,qhi,k,mf);And(visible,visible,hi,mf);} \
  if constexpr(!FULL) And(visible,visible,valid,mf); \
  Select(x,x,negative,visible); \
  if constexpr(!FULL) {if(start+j+LANE>=cols) x=negative;} \
  StoreAlign(src+(start+j+LANE)*64,x,mf);Max(MAXIMUM,MAXIMUM,x,mf);
  LABEL_MAX(0,m0) LABEL_MAX(1,m1) LABEL_MAX(2,m2) LABEL_MAX(3,m3)
#undef LABEL_MAX
 }
 Max(m0,m0,m1,mf);Max(m2,m2,m3,mf);Max(result,m0,m2,mf);
}
__simd_callee__ inline void group_pack(__ubuf__ float* src,__ubuf__ uint8_t* dst,
 RegTensor<float>& maximum,RegTensor<float>& sm,RegTensor<uint8_t>& index,
 RegTensor<int32_t>& encoded,RegTensor<float>& sum0,RegTensor<float>& sum1,
 RegTensor<float>& sum2,RegTensor<float>& sum3,int32_t start) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),mb=CreateMask<uint8_t,MaskPattern::ALL>(),condition;
 RegTensor<float> inv;
 mx_group_scale(maximum,sm,inv,encoded);
 #pragma unroll 4
 for(uint16_t j=0;j<32;j+=4) {
  RegTensor<float> x0,x1,x2,x3;
  RegTensor<fp8_e4m3fn_t> p0,p1,p2,p3;
  RegTensor<uint8_t> packed;
#define EXACT_P(LANE, SUM, PACK, TRAIT) \
  LoadAlign(x##LANE,src+(start+j+LANE)*64); \
  FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x##LANE,x##LANE,sm,mf); \
  Add(SUM,SUM,x##LANE,mf);Mul(x##LANE,x##LANE,inv,mf); \
  Cast<fp8_e4m3fn_t,float,TRAIT>(PACK,x##LANE,mf);
  EXACT_P(0,sum0,p0,r0) EXACT_P(1,sum1,p1,r1) EXACT_P(2,sum2,p2,r2) EXACT_P(3,sum3,p3,r3)
#undef EXACT_P
  Or(packed,reinterpret_cast<RegTensor<uint8_t>&>(p0),reinterpret_cast<RegTensor<uint8_t>&>(p1),mb);
  Or(packed,packed,reinterpret_cast<RegTensor<uint8_t>&>(p2),mb);Or(packed,packed,reinterpret_cast<RegTensor<uint8_t>&>(p3),mb);
  Gather(packed,packed,index);StoreAlign(dst+(start+j)*64,packed,mb);
 }
}
template<bool WIDE,bool FULL>
__simd_vf__ inline void label_quant(__ubuf__ float* src,__ubuf__ uint8_t* dst,__ubuf__ uint8_t* scales,
 __ubuf__ float* mp,__ubuf__ float* lp,__ubuf__ float* ap,__ubuf__ float* bp,
 __ubuf__ uint32_t* q,__ubuf__ uint32_t* k,float scale,int32_t rows,int32_t cols,int32_t first) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),mb=CreateMask<uint8_t,MaskPattern::ALL>(),valid,condition;
 RegTensor<uint32_t> qlo,qhi;
 if constexpr(WIDE) {
  RegTensor<uint32_t> a,b,index,even,odd,idxmask;RegTensor<int32_t> lane;MaskReg lower;
  LoadAlign(a,q);LoadAlign(b,q+64);Arange(lane,int32_t(0));index=reinterpret_cast<RegTensor<uint32_t>&>(lane);
  Compares<int32_t,CMPMODE::LT>(lower,lane,int32_t(32),mf);Duplicate(idxmask,uint32_t(31));And(even,index,idxmask,mf);ShiftLefts(even,even,int16_t(1),mf);Adds(odd,even,uint32_t(1),mf);
  RegTensor<uint32_t> al,bl,ah,bh;Gather(al,a,even);Gather(bl,b,even);Gather(ah,a,odd);Gather(bh,b,odd);Select(qlo,al,bl,lower);Select(qhi,ah,bh,lower);
 } else {LoadAlign(qlo,q);Duplicate(qhi,uint32_t(0));}
 RegTensor<int32_t> qi;Arange(qi,int32_t(0));Compares<int32_t,CMPMODE::LT>(valid,qi,rows,mf);
 RegTensor<float> g0,g1,g2,g3;
 label_max<WIDE,FULL>(g0,src,k,qlo,qhi,valid,0,cols,scale);
 label_max<WIDE,FULL>(g1,src,k,qlo,qhi,valid,32,cols,scale);
 label_max<WIDE,FULL>(g2,src,k,qlo,qhi,valid,64,cols,scale);
 label_max<WIDE,FULL>(g3,src,k,qlo,qhi,valid,96,cols,scale);
 RegTensor<float> m,sm,alpha,oldm,oldl,bad,zero,one,tmp;
 Duplicate(zero,0.f);Duplicate(one,1.f);
 Max(m,g0,g1,mf);Max(tmp,g2,g3,mf);Max(m,m,tmp,mf);
 if(first) {Duplicate(oldm,-__builtin_inff());oldl=zero;bad=zero;}else {LoadAlign(oldm,mp);LoadAlign(oldl,lp);LoadAlign(bad,bp);}
 Max(m,m,oldm,mf);Compares<float,CMPMODE::EQ>(condition,m,-__builtin_inff(),mf);Select(sm,zero,m,condition);
 FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(alpha,oldm,sm,mf);Compares<float,CMPMODE::GT>(condition,oldl,0.f,mf);Select(alpha,alpha,zero,condition);
 StoreAlign(mp,m,mf);StoreAlign(ap,alpha,mf);
 RegTensor<int8_t> seq;RegTensor<uint8_t> index,upper;Arange(seq,int8_t(0));
 ShiftLefts(index,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(3),mb);ShiftRights(index,index,int16_t(1),mb);
 ShiftLefts(upper,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(1),mb);ShiftRights(upper,upper,int16_t(6),mb);Add(index,index,upper,mb);
 ShiftRights(upper,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(7),mb);ShiftLefts(upper,upper,int16_t(7),mb);Or(index,index,upper,mb);
 RegTensor<float> sum0,sum1,sum2,sum3;sum0=zero;sum1=zero;sum2=zero;sum3=zero;
 RegTensor<int32_t> e0,e1,e2,e3;
 LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
 group_pack(src,dst,g0,sm,index,e0,sum0,sum1,sum2,sum3,0);
 group_pack(src,dst,g1,sm,index,e1,sum0,sum1,sum2,sum3,32);mx_scale_pair(scales,e0,e1);
 group_pack(src,dst,g2,sm,index,e2,sum0,sum1,sum2,sum3,64);
 group_pack(src,dst,g3,sm,index,e3,sum0,sum1,sum2,sum3,96);mx_scale_pair(scales+32,e2,e3);
 Add(sum0,sum0,sum1,mf);Add(sum2,sum2,sum3,mf);Add(sum0,sum0,sum2,mf);
 Mul(oldl,oldl,alpha,mf);Add(oldl,oldl,sum0,mf);StoreAlign(lp,oldl,mf);
 Compares<float,CMPMODE::LT>(condition,sum0,__builtin_inff(),mf);Select(bad,bad,one,condition);StoreAlign(bp,bad,mf);
 LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
}
extern "C" [aicore] __attribute__((always_inline))
void flex_exact_label_quant(uint64_t s,uint64_t p,uint64_t ps,uint64_t m,uint64_t l,uint64_t a,uint64_t b,uint64_t q,uint64_t k,float scale,int32_t rows,int32_t cols,int32_t first,int32_t wide,int32_t full) {
#define CALL(W,F) label_quant<W,F>((__ubuf__ float*)s,(__ubuf__ uint8_t*)p,(__ubuf__ uint8_t*)ps,(__ubuf__ float*)m,(__ubuf__ float*)l,(__ubuf__ float*)a,(__ubuf__ float*)b,(__ubuf__ uint32_t*)q,(__ubuf__ uint32_t*)k,scale,rows,cols,first)
 if(wide) {if(full) CALL(true,true);else CALL(true,false);}else {if(full) CALL(false,true);else CALL(false,false);}
#undef CALL
}
"""
)


# Interval tiles skip inactive groups and retain the fixed half-tile specializations.
_MX_LABEL_INTERVAL_SOURCE = r"""#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;using namespace AscendC::Reg;
#ifndef FLEX_MX_SCALE_DEFINED
#define FLEX_MX_SCALE_DEFINED
constexpr CastTrait mx_scale_cast={RegLayout::ZERO,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
__simd_callee__ inline void mx_group_scale(RegTensor<float>& maximum,RegTensor<float>& sm,RegTensor<float>& inv,RegTensor<int32_t>& encoded) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),condition;
 RegTensor<float> p,zero;Duplicate(zero,0.f);
 FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(p,maximum,sm,mf);
 Compares<float,CMPMODE::LT>(condition,p,__builtin_inff(),mf);Select(p,p,zero,condition);
 RegTensor<int32_t> power,mantissa,one,none,increment,mask,invbits;
 Duplicate(one,int32_t(1));Duplicate(none,int32_t(0));Duplicate(mask,int32_t(0x7fffff));
 ShiftRights(power,reinterpret_cast<RegTensor<int32_t>&>(p),int16_t(23),mf);Adds(power,power,int32_t(-135),mf);
 And(mantissa,reinterpret_cast<RegTensor<int32_t>&>(p),mask,mf);
 Compares<int32_t,CMPMODE::GT>(condition,mantissa,int32_t(0x600000),mf);Select(increment,one,none,condition);Add(power,power,increment,mf);
 Maxs(power,power,int32_t(-127),mf);Mins(power,power,int32_t(127),mf);
 Compares<float,CMPMODE::EQ>(condition,p,0.f,mf);Select(power,none,power,condition);
 Duplicate(invbits,int32_t(127));Sub(invbits,invbits,power,mf);ShiftLefts(invbits,invbits,int16_t(23),mf);inv=reinterpret_cast<RegTensor<float>&>(invbits);
 Adds(encoded,power,int32_t(127),mf);
}
__simd_callee__ inline void mx_scale_pair(__ubuf__ uint8_t* dst,RegTensor<int32_t>& lo,RegTensor<int32_t>& hi) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>();uint32_t count=64;MaskReg mh=UpdateMask<int16_t>(count);
 RegTensor<int32_t> joined;RegTensor<int16_t> words,packed,unused;
 ShiftLefts(joined,hi,int16_t(8),mf);Or(joined,joined,lo,mf);
 Cast<int16_t,int32_t,mx_scale_cast>(words,joined,mf);DeInterleave(packed,unused,words,words);
 StoreAlign<int16_t,DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ int16_t*)dst,packed,2,mh);
}
#endif
constexpr CastTrait ri_r0={RegLayout::ZERO,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait ri_r1={RegLayout::ONE,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait ri_r2={RegLayout::TWO,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait ri_r3={RegLayout::THREE,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
template<bool FULL,bool MASKED=true>
__simd_callee__ inline void prefix_range_max8(RegTensor<float>& result,__ubuf__ float* src,
 MaskReg& valid,int32_t start,int32_t first_key,int32_t last_key,float scale,int32_t prefix_base) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),visible;
 RegTensor<int32_t> query;
 if constexpr(MASKED) {Arange(query,int32_t(0));Adds(query,query,prefix_base,mf);}
 RegTensor<float> m0,m1,m2,m3,m4,m5,m6,m7,negative;
 Duplicate(negative,-__builtin_inff());m0=negative;m1=negative;m2=negative;m3=negative;m4=negative;m5=negative;m6=negative;m7=negative;
 uint16_t begin=first_key<=start?0:first_key>=start+32?32:((first_key-start)/8)*8;
 uint16_t end=last_key<=start?0:last_key>=start+32?32:((last_key-start+7)/8)*8;
 for(uint16_t j=begin;j<end;j+=8) {
  RegTensor<float> x0,x1,x2,x3,x4,x5,x6,x7;
#define LOAD_X(I) LoadAlign(x##I,src+(start+j+I)*64);
  LOAD_X(0) LOAD_X(1) LOAD_X(2) LOAD_X(3) LOAD_X(4) LOAD_X(5) LOAD_X(6) LOAD_X(7)
#undef LOAD_X
#define SCALE_X(I) Muls(x##I,x##I,scale,mf);
  SCALE_X(0) SCALE_X(1) SCALE_X(2) SCALE_X(3) SCALE_X(4) SCALE_X(5) SCALE_X(6) SCALE_X(7)
#undef SCALE_X
#define SELECT_X(I) \
  if constexpr(MASKED) {Compares<int32_t,CMPMODE::GE>(visible,query,int32_t(start+j+I),mf); \
   if constexpr(!FULL) And(visible,visible,valid,mf); \
   Select(x##I,x##I,negative,visible);} \
  else if constexpr(!FULL) Select(x##I,x##I,negative,valid); \
  if constexpr(!FULL) {if(start+j+I<first_key || start+j+I>=last_key) x##I=negative;}
  SELECT_X(0) SELECT_X(1) SELECT_X(2) SELECT_X(3) SELECT_X(4) SELECT_X(5) SELECT_X(6) SELECT_X(7)
#undef SELECT_X
#define STORE_X(I) StoreAlign(src+(start+j+I)*64,x##I,mf);
  STORE_X(0) STORE_X(1) STORE_X(2) STORE_X(3) STORE_X(4) STORE_X(5) STORE_X(6) STORE_X(7)
#undef STORE_X
#define MAX_X(I) Max(m##I,m##I,x##I,mf);
  MAX_X(0) MAX_X(1) MAX_X(2) MAX_X(3) MAX_X(4) MAX_X(5) MAX_X(6) MAX_X(7)
#undef MAX_X
 }
 Max(m0,m0,m1,mf);Max(m2,m2,m3,mf);Max(m4,m4,m5,mf);Max(m6,m6,m7,mf);
 Max(m0,m0,m2,mf);Max(m4,m4,m6,mf);Max(result,m0,m4,mf);
}
template<bool FULL,bool PREFIX=false,bool ALL_VISIBLE=false>
__simd_callee__ inline void range_max(RegTensor<float>& result,__ubuf__ float* src,
 MaskReg& valid,int32_t start,int32_t first_key,int32_t last_key,float scale,int32_t prefix_base=0) {
 if constexpr(PREFIX) {prefix_range_max8<FULL,!ALL_VISIBLE>(result,src,valid,start,first_key,last_key,scale,prefix_base);return;}
 MaskReg mf=CreateMask<float,MaskPattern::ALL>();
 RegTensor<float> m0,m1,m2,m3,negative,x;
 Duplicate(negative,-__builtin_inff());m0=negative;m1=negative;m2=negative;m3=negative;
 uint16_t begin=first_key<=start?0:first_key>=start+32?32:((first_key-start)/4)*4;
 uint16_t end=last_key<=start?0:last_key>=start+32?32:((last_key-start+3)/4)*4;
 for(uint16_t j=begin;j<end;j+=4) {
#define RANGE_MAX(LANE, MAXIMUM) \
  LoadAlign(x,src+(start+j+LANE)*64);Muls(x,x,scale,mf); \
  if constexpr(!FULL) Select(x,x,negative,valid); \
  if(start+j+LANE<first_key || start+j+LANE>=last_key) x=negative; \
  StoreAlign(src+(start+j+LANE)*64,x,mf);Max(MAXIMUM,MAXIMUM,x,mf);
  RANGE_MAX(0,m0) RANGE_MAX(1,m1) RANGE_MAX(2,m2) RANGE_MAX(3,m3)
#undef RANGE_MAX
 }
 Max(m0,m0,m1,mf);Max(m2,m2,m3,mf);Max(result,m0,m2,mf);
}
__simd_callee__ inline void range_pack(__ubuf__ float* src,__ubuf__ uint8_t* dst,
 RegTensor<float>& maximum,RegTensor<float>& sm,RegTensor<uint8_t>& index,
 RegTensor<int32_t>& encoded,RegTensor<float>& sum0,RegTensor<float>& sum1,
 RegTensor<float>& sum2,RegTensor<float>& sum3,int32_t start,int32_t first_key,int32_t last_key) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),mb=CreateMask<uint8_t,MaskPattern::ALL>(),condition;
 RegTensor<float> inv;
 mx_group_scale(maximum,sm,inv,encoded);
 uint16_t begin=first_key<=start?0:first_key>=start+32?32:((first_key-start)/4)*4;
 uint16_t end=last_key<=start?0:last_key>=start+32?32:((last_key-start+3)/4)*4;
 for(uint16_t j=begin;j<end;j+=4) {
  RegTensor<float> x0,x1,x2,x3;
  RegTensor<fp8_e4m3fn_t> p0,p1,p2,p3;
  RegTensor<uint8_t> packed;
#define EXACT_P(LANE, SUM, PACK, TRAIT) \
  LoadAlign(x##LANE,src+(start+j+LANE)*64); \
  FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x##LANE,x##LANE,sm,mf); \
  Add(SUM,SUM,x##LANE,mf);Mul(x##LANE,x##LANE,inv,mf); \
  Cast<fp8_e4m3fn_t,float,TRAIT>(PACK,x##LANE,mf);
  EXACT_P(0,sum0,p0,ri_r0) EXACT_P(1,sum1,p1,ri_r1) EXACT_P(2,sum2,p2,ri_r2) EXACT_P(3,sum3,p3,ri_r3)
#undef EXACT_P
  Or(packed,reinterpret_cast<RegTensor<uint8_t>&>(p0),reinterpret_cast<RegTensor<uint8_t>&>(p1),mb);
  Or(packed,packed,reinterpret_cast<RegTensor<uint8_t>&>(p2),mb);Or(packed,packed,reinterpret_cast<RegTensor<uint8_t>&>(p3),mb);
  Gather(packed,packed,index);StoreAlign(dst+(start+j)*64,packed,mb);
 }
}
template<bool WIDE,bool FULL,int BEGIN=-1,int END=-1,bool PREFIX=false,int QUERY_BASE=-1>
__simd_vf__ inline void range_quant(__ubuf__ float* src,__ubuf__ uint8_t* dst,__ubuf__ uint8_t* scales,
 __ubuf__ float* mp,__ubuf__ float* lp,__ubuf__ float* ap,__ubuf__ float* bp,
 __ubuf__ uint32_t* q,__ubuf__ uint32_t* k,float scale,int32_t rows,int32_t cols,int32_t first,int32_t first_key,int32_t last_key,int32_t prefix_base=0) {
 if constexpr(BEGIN>=0) { first_key=BEGIN;last_key=END; }
 if constexpr(PREFIX && QUERY_BASE>=0) prefix_base=QUERY_BASE;
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),mb=CreateMask<uint8_t,MaskPattern::ALL>(),valid,condition;
 RegTensor<int32_t> qi;Arange(qi,int32_t(0));Compares<int32_t,CMPMODE::LT>(valid,qi,rows,mf);
 RegTensor<float> g0,g1,g2,g3;
 range_max<FULL,PREFIX,(PREFIX && QUERY_BASE>=32)>(g0,src,valid,0,first_key,last_key,scale,prefix_base);
 range_max<FULL,PREFIX,(PREFIX && QUERY_BASE>=64)>(g1,src,valid,32,first_key,last_key,scale,prefix_base);
 range_max<FULL,PREFIX,(PREFIX && QUERY_BASE>=96)>(g2,src,valid,64,first_key,last_key,scale,prefix_base);
 range_max<FULL,PREFIX,(PREFIX && QUERY_BASE>=128)>(g3,src,valid,96,first_key,last_key,scale,prefix_base);
 RegTensor<float> m,sm,alpha,oldm,oldl,bad,zero,one,tmp;
 Duplicate(zero,0.f);Duplicate(one,1.f);
 Max(m,g0,g1,mf);Max(tmp,g2,g3,mf);Max(m,m,tmp,mf);
 if(first) {Duplicate(oldm,-__builtin_inff());oldl=zero;bad=zero;}else {LoadAlign(oldm,mp);LoadAlign(oldl,lp);LoadAlign(bad,bp);}
 Max(m,m,oldm,mf);Compares<float,CMPMODE::EQ>(condition,m,-__builtin_inff(),mf);Select(sm,zero,m,condition);
 FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(alpha,oldm,sm,mf);Compares<float,CMPMODE::GT>(condition,oldl,0.f,mf);Select(alpha,alpha,zero,condition);
 StoreAlign(mp,m,mf);StoreAlign(ap,alpha,mf);
 RegTensor<int8_t> seq;RegTensor<uint8_t> index,upper;Arange(seq,int8_t(0));
 ShiftLefts(index,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(3),mb);ShiftRights(index,index,int16_t(1),mb);
 ShiftLefts(upper,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(1),mb);ShiftRights(upper,upper,int16_t(6),mb);Add(index,index,upper,mb);
 ShiftRights(upper,reinterpret_cast<RegTensor<uint8_t>&>(seq),int16_t(7),mb);ShiftLefts(upper,upper,int16_t(7),mb);Or(index,index,upper,mb);
 RegTensor<float> sum0,sum1,sum2,sum3;sum0=zero;sum1=zero;sum2=zero;sum3=zero;
 RegTensor<int32_t> e0,e1,e2,e3;
 RegTensor<uint8_t> zeros;Duplicate(zeros,uint8_t(0));
 if constexpr(!(PREFIX && BEGIN==0 && END==64)) {
 for(uint16_t i=0;i<first_key/4;i++) StoreAlign(dst+i*256,zeros,mb);
 for(uint16_t i=(last_key+3)/4;i<32;i++) StoreAlign(dst+i*256,zeros,mb);
 }
 LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
 range_pack(src,dst,g0,sm,index,e0,sum0,sum1,sum2,sum3,0,first_key,last_key);
 range_pack(src,dst,g1,sm,index,e1,sum0,sum1,sum2,sum3,32,first_key,last_key);mx_scale_pair(scales,e0,e1);
 range_pack(src,dst,g2,sm,index,e2,sum0,sum1,sum2,sum3,64,first_key,last_key);
 range_pack(src,dst,g3,sm,index,e3,sum0,sum1,sum2,sum3,96,first_key,last_key);mx_scale_pair(scales+32,e2,e3);
 Add(sum0,sum0,sum1,mf);Add(sum2,sum2,sum3,mf);Add(sum0,sum0,sum2,mf);
 Mul(oldl,oldl,alpha,mf);Add(oldl,oldl,sum0,mf);StoreAlign(lp,oldl,mf);
 Compares<float,CMPMODE::LT>(condition,sum0,__builtin_inff(),mf);Select(bad,bad,one,condition);StoreAlign(bp,bad,mf);
 LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
}
extern "C" [aicore] __attribute__((always_inline))
void flex_exact_label_interval_quant(uint64_t s,uint64_t p,uint64_t ps,uint64_t m,uint64_t l,uint64_t a,uint64_t b,uint64_t q,uint64_t k,float scale,int32_t rows,int32_t cols,int32_t first,int32_t wide,int32_t full,int32_t first_key,int32_t last_key) {
#define CALL(W,F) range_quant<W,F>((__ubuf__ float*)s,(__ubuf__ uint8_t*)p,(__ubuf__ uint8_t*)ps,(__ubuf__ float*)m,(__ubuf__ float*)l,(__ubuf__ float*)a,(__ubuf__ float*)b,(__ubuf__ uint32_t*)q,(__ubuf__ uint32_t*)k,scale,rows,cols,first,first_key,last_key)
#define FIXED(L,R) range_quant<false,true,L,R>((__ubuf__ float*)s,(__ubuf__ uint8_t*)p,(__ubuf__ uint8_t*)ps,(__ubuf__ float*)m,(__ubuf__ float*)l,(__ubuf__ float*)a,(__ubuf__ float*)b,(__ubuf__ uint32_t*)q,(__ubuf__ uint32_t*)k,scale,rows,cols,first,first_key,last_key)
 if(full && first_key==0 && last_key==64) FIXED(0,64);
 else if(full && first_key==64 && last_key==128) FIXED(64,128);
 else if(full && first_key==0 && last_key==128) FIXED(0,128);
 else if(wide) {if(full) CALL(true,true);else CALL(true,false);}else {if(full) CALL(false,true);else CALL(false,false);}
#undef FIXED
#undef CALL
}
extern "C" [aicore] __attribute__((always_inline))
void mx_exact_periodic_prefix(uint64_t s,uint64_t p,uint64_t ps,uint64_t m,uint64_t l,uint64_t a,uint64_t b,float scale,int32_t rows,int32_t cols,int32_t first,int32_t prefix_base) {
 int32_t last=prefix_base+rows;last=last<cols?last:cols;
#define PREFIX(F,L,R,Q) range_quant<false,F,L,R,true,Q>((__ubuf__ float*)s,(__ubuf__ uint8_t*)p,(__ubuf__ uint8_t*)ps,(__ubuf__ float*)m,(__ubuf__ float*)l,(__ubuf__ float*)a,(__ubuf__ float*)b,nullptr,nullptr,scale,rows,cols,first,0,last,prefix_base)
 if(rows==64 && prefix_base==0 && cols>=64) PREFIX(true,0,64,0);
 else if(rows==64 && prefix_base==64 && cols==128) PREFIX(true,0,128,64);
 else PREFIX(false,-1,-1,-1);
#undef PREFIX
}
"""


@tla.extern(name="mx_exact_periodic_prefix", source=_MX_LABEL_INTERVAL_SOURCE)
def _mx_exact_periodic_prefix(
    s: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    p: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    ps: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    l: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    a: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    bad: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    scale: tla.Float32,
    rows: tla.Int32,
    cols: tla.Int32,
    first: tla.Int32,
    prefix_base: tla.Int32,
) -> None: ...


@tla.extern(name="flex_exact_label_quant", source=_MX_LABEL_SOURCE)
def _mx_exact_label_quant(
    s: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    p: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    ps: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    l: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    a: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    bad: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    q: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    k: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    scale: tla.Float32,
    rows: tla.Int32,
    cols: tla.Int32,
    first: tla.Int32,
    wide: tla.Int32,
    full: tla.Int32,
) -> None: ...


_MX_EXACT_EXP_PACK_SOURCE = r"""#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;using namespace AscendC::Reg;
constexpr CastTrait rne0 = {RegLayout::ZERO, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne1 = {RegLayout::ONE, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne2 = {RegLayout::TWO, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne3 = {RegLayout::THREE, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
__simd_vf__ inline void flex_exact_exp_pack_update(__ubuf__ float* src, __ubuf__ uint8_t* dst,
                                __ubuf__ float* inverses, __ubuf__ float* safemstate,
                                __ubuf__ float* lstate, __ubuf__ float* astate, __ubuf__ float* badstate, float scale) {
  MaskReg mf = CreateMask<float, MaskPattern::ALL>();
  MaskReg mb = CreateMask<uint8_t, MaskPattern::ALL>();
  RegTensor<int8_t> seq;
  RegTensor<uint8_t> loidx, hiidx, upper;
  Arange(seq, int8_t(0));
  ShiftLefts(loidx, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(3), mb);
  ShiftRights(loidx, loidx, int16_t(1), mb);
  ShiftLefts(upper, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(1), mb);
  ShiftRights(upper, upper, int16_t(6), mb);
  Add(loidx, loidx, upper, mb);
  ShiftRights(upper, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(7), mb);
  ShiftLefts(upper, upper, int16_t(7), mb);
  Or(loidx, loidx, upper, mb);
  RegTensor<float> sm, inv, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
  LoadAlign(sm, safemstate);
  Duplicate(sum0, 0.0f, mf); Duplicate(sum1, 0.0f, mf);
  Duplicate(sum2, 0.0f, mf); Duplicate(sum3, 0.0f, mf);
  Duplicate(sum4, 0.0f, mf); Duplicate(sum5, 0.0f, mf);
  Duplicate(sum6, 0.0f, mf); Duplicate(sum7, 0.0f, mf);
  for (uint16_t row = 0; row < 128; row += 8) {
    LoadAlign(inv, inverses + (row / 32) * 64);
    {
      RegTensor<float> x0, x1, x2, x3;
      RegTensor<fp8_e4m3fn_t> p0, p1, p2, p3;
      RegTensor<uint8_t> packed, lo;
      LoadAlign(x0, src + (row + 0) * 64);
      Muls(x0, x0, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x0,x0,sm,mf);
      Add(sum0, sum0, x0, mf);
      Mul(x0, x0, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne0>(p0, x0, mf);
      LoadAlign(x1, src + (row + 1) * 64);
      Muls(x1, x1, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x1,x1,sm,mf);
      Add(sum1, sum1, x1, mf);
      Mul(x1, x1, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne1>(p1, x1, mf);
      LoadAlign(x2, src + (row + 2) * 64);
      Muls(x2, x2, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x2,x2,sm,mf);
      Add(sum2, sum2, x2, mf);
      Mul(x2, x2, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne2>(p2, x2, mf);
      LoadAlign(x3, src + (row + 3) * 64);
      Muls(x3, x3, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x3,x3,sm,mf);
      Add(sum3, sum3, x3, mf);
      Mul(x3, x3, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne3>(p3, x3, mf);
      Or(packed, reinterpret_cast<RegTensor<uint8_t>&>(p0), reinterpret_cast<RegTensor<uint8_t>&>(p1), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p2), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p3), mb);
      Gather(lo, packed, loidx);
      StoreAlign(dst + (row + 0) * 64, lo, mb);
    }
    {
      RegTensor<float> x0, x1, x2, x3;
      RegTensor<fp8_e4m3fn_t> p0, p1, p2, p3;
      RegTensor<uint8_t> packed, lo;
      LoadAlign(x0, src + (row + 4) * 64);
      Muls(x0, x0, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x0,x0,sm,mf);
      Add(sum4, sum4, x0, mf);
      Mul(x0, x0, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne0>(p0, x0, mf);
      LoadAlign(x1, src + (row + 5) * 64);
      Muls(x1, x1, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x1,x1,sm,mf);
      Add(sum5, sum5, x1, mf);
      Mul(x1, x1, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne1>(p1, x1, mf);
      LoadAlign(x2, src + (row + 6) * 64);
      Muls(x2, x2, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x2,x2,sm,mf);
      Add(sum6, sum6, x2, mf);
      Mul(x2, x2, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne2>(p2, x2, mf);
      LoadAlign(x3, src + (row + 7) * 64);
      Muls(x3, x3, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x3,x3,sm,mf);
      Add(sum7, sum7, x3, mf);
      Mul(x3, x3, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne3>(p3, x3, mf);
      Or(packed, reinterpret_cast<RegTensor<uint8_t>&>(p0), reinterpret_cast<RegTensor<uint8_t>&>(p1), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p2), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p3), mb);
      Gather(lo, packed, loidx);
      StoreAlign(dst + (row + 4) * 64, lo, mb);
    }
  }
  Add(sum0, sum0, sum1, mf); Add(sum2, sum2, sum3, mf);
  Add(sum4, sum4, sum5, mf); Add(sum6, sum6, sum7, mf);
  Add(sum0, sum0, sum2, mf); Add(sum4, sum4, sum6, mf);
  Add(sum0, sum0, sum4, mf);
  RegTensor<float> oldl, alpha, bad, one;
  LoadAlign(oldl, lstate); LoadAlign(alpha, astate); LoadAlign(bad, badstate);
  Mul(oldl, oldl, alpha, mf); Add(oldl, oldl, sum0, mf);
  StoreAlign(lstate, oldl, mf);
  Duplicate(one, 1.0f, mf);
  MaskReg finite;
  Compares<float, CMPMODE::LT>(finite, sum0, __builtin_inff(), mf);
  Select(bad, bad, one, finite); StoreAlign(badstate, bad, mf);
  LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}
extern "C" [aicore] __attribute__((always_inline))
void flex_mx_exact_exp_pack_update(uint64_t src, uint64_t dst, uint64_t inverses,
                      uint64_t safemstate, uint64_t lstate, uint64_t astate, uint64_t badstate, float scale) {
  flex_exact_exp_pack_update(reinterpret_cast<__ubuf__ float*>(src),
                reinterpret_cast<__ubuf__ uint8_t*>(dst),
                reinterpret_cast<__ubuf__ float*>(inverses),
                reinterpret_cast<__ubuf__ float*>(safemstate),
                reinterpret_cast<__ubuf__ float*>(lstate), reinterpret_cast<__ubuf__ float*>(astate),
                reinterpret_cast<__ubuf__ float*>(badstate), scale);
}
"""


@tla.extern(name="flex_mx_exact_exp_pack_update", source=_MX_EXACT_EXP_PACK_SOURCE)
def _mx_exact_exp_pack_update(
    src: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    dst: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    inverses: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    safe: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    lstate: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    astate: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    badstate: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    scale: tla.Float32,
) -> None: ...


class _MxQueryExpr(_MxComparable):
    def __init__(self, tree):
        self.tree = tree

    @staticmethod
    def lift(value):
        if isinstance(value, _MxQueryExpr):
            return value
        if type(value) is int and -(2**31) <= value < 2**31:
            return _MxQueryExpr(("const", value))
        raise NotImplementedError("query mask requires int32 expressions")

    def binary(self, other, op):
        return _MxQueryExpr((op, self.tree, self.lift(other).tree))

    def __add__(self, other):
        return self.binary(other, "add")

    __radd__ = __add__

    def __sub__(self, other):
        return self.binary(other, "sub")

    def __rsub__(self, other):
        return self.lift(other).binary(self, "sub")

    def compare(self, other, mode):
        return _MxAffineMask((mode, self.tree, self.lift(other).tree))

    def to(self, dtype):
        if dtype != tla.Int32:
            raise NotImplementedError("query arithmetic must remain int32")
        return self


class _MxQueryAuxProbe:
    def __init__(self, tensor, index, b, h):
        self.tensor, self.index, self.b, self.h = tensor, index, b, h
        self.dtype = tensor.dtype

    def load_cached(self, indices):
        t = self.tensor
        prefix = (
            (self.b, self.h) if len(t.strides) == 3 else (self.b,) if len(t.strides) == 2 else ()
        )
        if (
            t.wide
            or len(indices) != len(t.strides)
            or not all(_mx_same_index(a, b) for a, b in zip(indices[:-1], prefix))
        ):
            raise NotImplementedError("query mask requires direct int32 query auxiliary indexing")
        last = indices[-1]
        query = ("var", "q")
        accepted = () if t.global_offsets and t.global_offsets[2] else (query,)
        if t.global_offsets:
            accepted += (("add", query, ("var", "qoff")), ("add", ("var", "qoff"), query))
        if not isinstance(last, _MxQueryExpr) or last.tree not in accepted:
            raise NotImplementedError("query mask requires direct query auxiliary indexing")
        return _MxQueryExpr(("aux", self.index))


def _mx_trace_query_aux_expression(mask_mod, b, h, sub, kstart, aux, scalars, info):
    if not aux or not all(isinstance(t, _MxCachedAux) and not t.wide for t in aux):
        return None
    q, k = (_MxQueryExpr(("var", name)) for name in ("q", "k"))
    sequence = tuple(_MxQueryExpr(("var", name)) for name in ("qlen", "klen", "qoff", "koff"))
    probes = tuple(_MxQueryAuxProbe(t, i, b, h) for i, t in enumerate(aux))
    try:
        result = _mx_trace_callback(mask_mod)(0.0, b, h, q, k, probes, scalars, sequence, True)
    except Exception:
        return None
    if not isinstance(result, _MxAffineMask):
        return None
    pointers = tuple(t.cache.ptr + t.query_index(sub) for t in aux)
    return result.tree, pointers + (pointers[0],) * (4 - len(pointers))


def _mx_trace_query_window(mask_mod, b, h, sub, kstart, aux, scalars, info, pure=False):
    expression = _mx_trace_query_aux_expression(mask_mod, b, h, sub, kstart, aux, scalars, info)
    if expression is None:
        return None
    leaves = []

    def flatten(tree):
        if tree[0] == "and":
            flatten(tree[1])
            flatten(tree[2])
        else:
            leaves.append(tree)

    flatten(expression[0])
    if pure and len(leaves) != 2:
        return None
    q, k = ("var", "q"), ("var", "k")
    if ("LE", k, q) not in leaves and ("GE", q, k) not in leaves:
        return None
    for leaf in leaves:
        mode, left, right = leaf
        if mode == "GE":
            left, right = right, left
            mode = "LE"
        if mode == "LE" and left == ("sub", q, k) and right[0] == "aux":
            return expression[1][right[1]]
    return None


@tla.extern(
    name="flex_mx_radius_max",
    source=_MX_CPP_HEADER
    + r"""__simd_vf__ inline void mx_radius_max_vf(__ubuf__ int32_t* dst,__ubuf__ int32_t* src,int32_t rows) {
 MaskReg all=CreateMask<int32_t,MaskPattern::ALL>(),valid;
 RegTensor<int32_t> value,lanes,minimum;
 LoadAlign(value,src);Arange(lanes,int32_t(0));Duplicate(minimum,int32_t(0x80000000u));
 Compares<int32_t,CMPMODE::LT>(valid,lanes,rows,all);Select(value,value,minimum,valid);
 ReduceMax(value,value,all);StoreAlign(dst,value,all);
}
extern "C" [aicore] __attribute__((always_inline))
void flex_mx_radius_max(uint64_t d,uint64_t s,int32_t rows) {
 mx_radius_max_vf((__ubuf__ int32_t*)d,(__ubuf__ int32_t*)s,rows);
}
""",
)
def _mx_radius_max(
    dst: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    src: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    rows: tla.Int32,
) -> None: ...


@tla.jit
def _mx_query_window_bounds(radius, scratch, qstart, kstart, rows, cols, checked: tla.Constexpr):
    word = tla.recast_ptr(scratch, dtype=tla.Int32)
    _mx_radius_max(word, radius, rows)
    ready = tla.flag(
        "exact_radius_full" if checked else "exact_radius_tail", tla.arch.VECTOR, tla.arch.SCALAR
    )
    tla.set_flag(ready)
    tla.wait_flag(ready)
    scalar = tla.make_tensor(word, tla.make_layout(tla.make_shape(1), tla.make_stride(1)))
    maximum = scalar[0].to(tla.Int64)
    lower = qstart.to(tla.Int64) - maximum - kstart.to(tla.Int64)
    lower = max(tla.Int64(0), min(cols.to(tla.Int64), lower))
    upper = max(0, min(cols, qstart + rows - kstart))
    return ((lower // tla.Int64(8)) * tla.Int64(8)).to(tla.Int32), ((upper + 7) // 8) * 8, maximum


_MX_EXACT_RANGE_SOURCE = r"""#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;using namespace AscendC::Reg;
constexpr CastTrait rne0 = {RegLayout::ZERO, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne1 = {RegLayout::ONE, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne2 = {RegLayout::TWO, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait rne3 = {RegLayout::THREE, SatMode::NO_SAT,
  MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
__simd_vf__ inline void flex_exact_exp_pack_range(__ubuf__ float* src, __ubuf__ uint8_t* dst,
                                __ubuf__ float* inverses, __ubuf__ float* safemstate,
                                __ubuf__ float* lstate, __ubuf__ float* astate, __ubuf__ float* badstate, float scale,int32_t first_key,int32_t last_key) {
  MaskReg mf = CreateMask<float, MaskPattern::ALL>();
  MaskReg mb = CreateMask<uint8_t, MaskPattern::ALL>();
  RegTensor<int8_t> seq;
  RegTensor<uint8_t> loidx, hiidx, upper;
  Arange(seq, int8_t(0));
  ShiftLefts(loidx, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(3), mb);
  ShiftRights(loidx, loidx, int16_t(1), mb);
  ShiftLefts(upper, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(1), mb);
  ShiftRights(upper, upper, int16_t(6), mb);
  Add(loidx, loidx, upper, mb);
  ShiftRights(upper, reinterpret_cast<RegTensor<uint8_t>&>(seq), int16_t(7), mb);
  ShiftLefts(upper, upper, int16_t(7), mb);
  Or(loidx, loidx, upper, mb);
  RegTensor<float> sm, inv, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
  LoadAlign(sm, safemstate);
  Duplicate(sum0, 0.0f, mf); Duplicate(sum1, 0.0f, mf);
  Duplicate(sum2, 0.0f, mf); Duplicate(sum3, 0.0f, mf);
  Duplicate(sum4, 0.0f, mf); Duplicate(sum5, 0.0f, mf);
  Duplicate(sum6, 0.0f, mf); Duplicate(sum7, 0.0f, mf);
  for (uint16_t row = first_key; row < last_key; row += 8) {
    LoadAlign(inv, inverses + (row / 32) * 64);
    {
      RegTensor<float> x0, x1, x2, x3;
      RegTensor<fp8_e4m3fn_t> p0, p1, p2, p3;
      RegTensor<uint8_t> packed, lo;
      LoadAlign(x0, src + (row + 0) * 64);
      Muls(x0, x0, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x0,x0,sm,mf);
      Add(sum0, sum0, x0, mf);
      Mul(x0, x0, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne0>(p0, x0, mf);
      LoadAlign(x1, src + (row + 1) * 64);
      Muls(x1, x1, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x1,x1,sm,mf);
      Add(sum1, sum1, x1, mf);
      Mul(x1, x1, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne1>(p1, x1, mf);
      LoadAlign(x2, src + (row + 2) * 64);
      Muls(x2, x2, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x2,x2,sm,mf);
      Add(sum2, sum2, x2, mf);
      Mul(x2, x2, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne2>(p2, x2, mf);
      LoadAlign(x3, src + (row + 3) * 64);
      Muls(x3, x3, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x3,x3,sm,mf);
      Add(sum3, sum3, x3, mf);
      Mul(x3, x3, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne3>(p3, x3, mf);
      Or(packed, reinterpret_cast<RegTensor<uint8_t>&>(p0), reinterpret_cast<RegTensor<uint8_t>&>(p1), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p2), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p3), mb);
      Gather(lo, packed, loidx);
      StoreAlign(dst + (row + 0) * 64, lo, mb);
    }
    {
      RegTensor<float> x0, x1, x2, x3;
      RegTensor<fp8_e4m3fn_t> p0, p1, p2, p3;
      RegTensor<uint8_t> packed, lo;
      LoadAlign(x0, src + (row + 4) * 64);
      Muls(x0, x0, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x0,x0,sm,mf);
      Add(sum4, sum4, x0, mf);
      Mul(x0, x0, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne0>(p0, x0, mf);
      LoadAlign(x1, src + (row + 5) * 64);
      Muls(x1, x1, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x1,x1,sm,mf);
      Add(sum5, sum5, x1, mf);
      Mul(x1, x1, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne1>(p1, x1, mf);
      LoadAlign(x2, src + (row + 6) * 64);
      Muls(x2, x2, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x2,x2,sm,mf);
      Add(sum6, sum6, x2, mf);
      Mul(x2, x2, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne2>(p2, x2, mf);
      LoadAlign(x3, src + (row + 7) * 64);
      Muls(x3, x3, scale, mf);
      FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x3,x3,sm,mf);
      Add(sum7, sum7, x3, mf);
      Mul(x3, x3, inv, mf);
      Cast<fp8_e4m3fn_t, float, rne3>(p3, x3, mf);
      Or(packed, reinterpret_cast<RegTensor<uint8_t>&>(p0), reinterpret_cast<RegTensor<uint8_t>&>(p1), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p2), mb);
      Or(packed, packed, reinterpret_cast<RegTensor<uint8_t>&>(p3), mb);
      Gather(lo, packed, loidx);
      StoreAlign(dst + (row + 4) * 64, lo, mb);
    }
  }
  Add(sum0, sum0, sum1, mf); Add(sum2, sum2, sum3, mf);
  Add(sum4, sum4, sum5, mf); Add(sum6, sum6, sum7, mf);
  Add(sum0, sum0, sum2, mf); Add(sum4, sum4, sum6, mf);
  Add(sum0, sum0, sum4, mf);
  RegTensor<float> oldl, alpha, bad, one;
  LoadAlign(oldl, lstate); LoadAlign(alpha, astate); LoadAlign(bad, badstate);
  Mul(oldl, oldl, alpha, mf); Add(oldl, oldl, sum0, mf);
  StoreAlign(lstate, oldl, mf);
  Duplicate(one, 1.0f, mf);
  MaskReg finite;
  Compares<float, CMPMODE::LT>(finite, sum0, __builtin_inff(), mf);
  Select(bad, bad, one, finite); StoreAlign(badstate, bad, mf);
  LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}
extern "C" [aicore] __attribute__((always_inline))
void flex_mx_exact_exp_pack_range(uint64_t src, uint64_t dst, uint64_t inverses,
                      uint64_t safemstate, uint64_t lstate, uint64_t astate, uint64_t badstate, float scale,int32_t first_key,int32_t last_key) {
  flex_exact_exp_pack_range(reinterpret_cast<__ubuf__ float*>(src),
                reinterpret_cast<__ubuf__ uint8_t*>(dst),
                reinterpret_cast<__ubuf__ float*>(inverses),
                reinterpret_cast<__ubuf__ float*>(safemstate),
                reinterpret_cast<__ubuf__ float*>(lstate), reinterpret_cast<__ubuf__ float*>(astate),
                reinterpret_cast<__ubuf__ float*>(badstate), scale, first_key, last_key);
}
"""


@tla.extern(name="flex_mx_exact_exp_pack_range", source=_MX_EXACT_RANGE_SOURCE)
def _mx_exact_exp_pack_range(
    src: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    dst: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    inverses: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    safe: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    lstate: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    astate: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    badstate: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    scale: tla.Float32,
    first_key: tla.Int32,
    last_key: tla.Int32,
) -> None: ...


@tla.extern(
    name="flex_mx_scatter",
    source=_CPP_MX_SCALE
    + r"""constexpr CastTrait scatter_r0={RegLayout::ZERO,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
template<bool Labels>
__simd_callee__ inline void scatter_gather(RegTensor<float>& x,RegTensor<int32_t>& key,MaskReg& valid,
 __ubuf__ float* src,RegTensor<int32_t>& lane,RegTensor<int32_t>& radius,RegTensor<int32_t>& starts,int32_t delta,int32_t relative,int32_t rows,int32_t cols,float scale) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),condition;
 RegTensor<int32_t> index;RegTensor<float> negative;Duplicate(negative,-__builtin_inff());
 if constexpr(Labels) Adds(key,starts,delta,mf);
 else Adds(key,lane,relative-delta,mf);
 Compares<int32_t,CMPMODE::LT>(valid,lane,rows,mf);
 Compares<int32_t,CMPMODE::GE>(condition,key,int32_t(0),mf);And(valid,valid,condition,mf);
 Compares<int32_t,CMPMODE::LT>(condition,key,cols,mf);And(valid,valid,condition,mf);
 if constexpr(Labels) Compares<int32_t,CMPMODE::GT>(condition,radius,delta,mf);
 else Compares<int32_t,CMPMODE::GE>(condition,radius,delta,mf);
 And(valid,valid,condition,mf);
 Maxs(index,key,int32_t(0),mf);Mins(index,index,int32_t(127),mf);ShiftLefts(index,index,int16_t(6),mf);Add(index,index,lane,mf);
 Gather(x,src,reinterpret_cast<RegTensor<uint32_t>&>(index),mf);Muls(x,x,scale,mf);Select(x,x,negative,valid);
}
template<bool Labels>
__simd_vf__ inline void scatter_quant(__ubuf__ float* src,__ubuf__ uint8_t* dst,__ubuf__ uint8_t* scales,
 __ubuf__ float* mp,__ubuf__ float* lp,__ubuf__ float* ap,__ubuf__ float* bp,
 __ubuf__ int32_t* radiusp,int32_t relative,int32_t rows,int32_t cols,int32_t first,int32_t maximum,float scale) {
 MaskReg mf=CreateMask<float,MaskPattern::ALL>(),mb=CreateMask<uint8_t,MaskPattern::ALL>(),valid,condition;
 RegTensor<int32_t> lane,radius,key,group,starts;Arange(lane,int32_t(0));
 if constexpr(Labels) {LoadAlign(starts,radiusp);LoadAlign(radius,radiusp+64);}
 else LoadAlign(radius,radiusp);
 RegTensor<float> g0,g1,g2,g3,negative,x,candidate;
 Duplicate(negative,-__builtin_inff());g0=negative;g1=negative;g2=negative;g3=negative;
 for(uint16_t d=0;d<maximum+(Labels?0:1);d++) {
  scatter_gather<Labels>(x,key,valid,src,lane,radius,starts,d,relative,rows,cols,scale);
  ShiftRights(group,key,int16_t(5),mf);
#define WD_MAX(G) Compares<int32_t,CMPMODE::EQ>(condition,group,int32_t(G),mf);Select(candidate,x,negative,condition);Max(g##G,g##G,candidate,mf);
  WD_MAX(0) WD_MAX(1) WD_MAX(2) WD_MAX(3)
#undef WD_MAX
 }
 RegTensor<float> m,sm,alpha,oldm,oldl,bad,zero,one,tmp;
 Duplicate(zero,0.f);Duplicate(one,1.f);
 Max(m,g0,g1,mf);Max(tmp,g2,g3,mf);Max(m,m,tmp,mf);
 if(first) {oldm=negative;oldl=zero;bad=zero;}else {LoadAlign(oldm,mp);LoadAlign(oldl,lp);LoadAlign(bad,bp);}
 Max(m,m,oldm,mf);Compares<float,CMPMODE::EQ>(condition,m,-__builtin_inff(),mf);Select(sm,zero,m,condition);
 FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(alpha,oldm,sm,mf);Compares<float,CMPMODE::GT>(condition,oldl,0.f,mf);Select(alpha,alpha,zero,condition);
 StoreAlign(mp,m,mf);StoreAlign(ap,alpha,mf);
 RegTensor<float> inv0,inv1,inv2,inv3,inv,sum;sum=zero;
 RegTensor<int32_t> e0,e1,e2,e3;
 mx_group_scale(g0,sm,inv0,e0);mx_group_scale(g1,sm,inv1,e1);mx_scale_pair(scales,e0,e1);
 mx_group_scale(g2,sm,inv2,e2);mx_group_scale(g3,sm,inv3,e3);mx_scale_pair(scales+32,e2,e3);
 RegTensor<uint8_t> zeros;Duplicate(zeros,uint8_t(0));
 for(uint16_t i=0;i<32;i++) StoreAlign(dst+i*256,zeros,mb);
 LocalMemBar<MemType::VEC_STORE,MemType::VEC_STORE>();
 for(uint16_t d=0;d<maximum+(Labels?0:1);d++) {
  scatter_gather<Labels>(x,key,valid,src,lane,radius,starts,d,relative,rows,cols,scale);
  FusedExpSub<float,float,RegLayout::ZERO,MaskMergeMode::ZEROING>(x,x,sm,mf);
  Select(x,x,zero,valid);Add(sum,sum,x,mf);
  ShiftRights(group,key,int16_t(5),mf);inv=inv0;
  Compares<int32_t,CMPMODE::EQ>(condition,group,int32_t(1),mf);Select(inv,inv1,inv,condition);
  Compares<int32_t,CMPMODE::EQ>(condition,group,int32_t(2),mf);Select(inv,inv2,inv,condition);
  Compares<int32_t,CMPMODE::EQ>(condition,group,int32_t(3),mf);Select(inv,inv3,inv,condition);
  Mul(x,x,inv,mf);
  RegTensor<fp8_e4m3fn_t> fp8;
  RegTensor<uint16_t> packed,unused;
  Cast<fp8_e4m3fn_t,float,scatter_r0>(fp8,x,mf);
  // B8 scatter consumes the low byte of each B16 slot.
  DeInterleave(packed,unused,reinterpret_cast<RegTensor<uint16_t>&>(fp8),reinterpret_cast<RegTensor<uint16_t>&>(fp8));
  RegTensor<int32_t> address,tmp,mask;
  Maxs(key,key,int32_t(0),mf);Mins(key,key,int32_t(127),mf);
  {
   ShiftRights(address,key,int16_t(2),mf);ShiftLefts(address,address,int16_t(8),mf);
   Duplicate(mask,int32_t(3));And(tmp,key,mask,mf);ShiftLefts(tmp,tmp,int16_t(5),mf);Or(address,address,tmp,mf);
   ShiftRights(tmp,lane,int16_t(5),mf);ShiftLefts(tmp,tmp,int16_t(7),mf);Or(address,address,tmp,mf);
  }
  Duplicate(mask,int32_t(31));And(tmp,lane,mask,mf);Or(address,address,tmp,mf);
  RegTensor<int16_t> index,word;
  Cast<int16_t,int32_t,scatter_r0>(word,address,mf);DeInterleave(index,reinterpret_cast<RegTensor<int16_t>&>(unused),word,word);
  MaskReg mh;MaskPack(mh,valid);
  Scatter(dst,reinterpret_cast<RegTensor<uint8_t>&>(packed),reinterpret_cast<RegTensor<uint16_t>&>(index),mh);
 }
 Mul(oldl,oldl,alpha,mf);Add(oldl,oldl,sum,mf);StoreAlign(lp,oldl,mf);
 Compares<float,CMPMODE::LT>(condition,sum,__builtin_inff(),mf);Select(bad,bad,one,condition);StoreAlign(bp,bad,mf);
 LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
}
extern "C" [aicore] __attribute__((always_inline))
void flex_mx_scatter(uint64_t s,uint64_t p,uint64_t ps,uint64_t m,uint64_t l,uint64_t a,uint64_t b,uint64_t radius,int32_t relative,int32_t rows,int32_t cols,int32_t first,int32_t maximum,int32_t labels,float scale) {
#define SCATTER(Labels) scatter_quant<Labels>((__ubuf__ float*)s,(__ubuf__ uint8_t*)p,(__ubuf__ uint8_t*)ps,(__ubuf__ float*)m,(__ubuf__ float*)l,(__ubuf__ float*)a,(__ubuf__ float*)b,(__ubuf__ int32_t*)radius,relative,rows,cols,first,maximum,scale);
 if(labels) {SCATTER(true)} else {SCATTER(false)}
#undef SCATTER
}
""",
)
def _mx_scatter(
    s: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    p: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    ps: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    l: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    a: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    bad: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    radius: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    relative: tla.Int32,
    rows: tla.Int32,
    cols: tla.Int32,
    first: tla.Int32,
    maximum: tla.Int32,
    labels: tla.Int32,
    scale: tla.Float32,
) -> None: ...


@tla.extern(
    name="flex_mx_label_ranges",
    source=_MX_CPP_HEADER
    + r"""__simd_vf__ inline void label_ranges_vf(__ubuf__ int32_t* query,__ubuf__ int32_t* keys,
 __ubuf__ int32_t* starts,__ubuf__ int32_t* maximum,int32_t rows,int32_t cols) {
 MaskReg mf=CreateMask<int32_t,MaskPattern::ALL>(),condition,valid;
 RegTensor<int32_t> q,lane,zero,unsafe,bad,limit,next,index,left,right;
 LoadAlign(q,query);Arange(lane,int32_t(0));Duplicate(zero,int32_t(0));Duplicate(bad,int32_t(129));Duplicate(limit,cols);unsafe=zero;
 for(uint16_t half=0;half<2;half++) {
  LoadAlign(left,keys+half*64);
  Adds(index,lane,int32_t(half*64+1),mf);Mins(index,index,int32_t(127),mf);
  Gather(right,keys,reinterpret_cast<RegTensor<uint32_t>&>(index),mf);
  Compare<int32_t,CMPMODE::GT>(condition,left,right,mf);
  Adds(index,lane,int32_t(half*64),mf);Compares<int32_t,CMPMODE::LT>(valid,index,cols-1,mf);And(condition,condition,valid,mf);
  Select(unsafe,bad,unsafe,condition);
 }
 RegTensor<int32_t> lo,hi,upperlo,upperhi,mid,midupper,k,kupper;
 lo=zero;hi=limit;upperlo=zero;upperhi=limit;
 for(uint16_t step=0;step<8;step++) {
  Add(mid,lo,hi,mf);ShiftRights(mid,mid,int16_t(1),mf);Mins(index,mid,int32_t(127),mf);
  Gather(k,keys,reinterpret_cast<RegTensor<uint32_t>&>(index),mf);
  Compare<int32_t,CMPMODE::LT>(condition,k,q,mf);Compares<int32_t,CMPMODE::LT>(valid,mid,cols,mf);And(condition,condition,valid,mf);
  Adds(next,mid,int32_t(1),mf);Select(lo,next,lo,condition);Select(hi,hi,mid,condition);
  Add(midupper,upperlo,upperhi,mf);ShiftRights(midupper,midupper,int16_t(1),mf);Mins(index,midupper,int32_t(127),mf);
  Gather(kupper,keys,reinterpret_cast<RegTensor<uint32_t>&>(index),mf);
  Compare<int32_t,CMPMODE::LE>(condition,kupper,q,mf);Compares<int32_t,CMPMODE::LT>(valid,midupper,cols,mf);And(condition,condition,valid,mf);
  Adds(next,midupper,int32_t(1),mf);Select(upperlo,next,upperlo,condition);Select(upperhi,upperhi,midupper,condition);
 }
 Sub(upperlo,upperlo,lo,mf);
 Compares<int32_t,CMPMODE::LT>(valid,lane,rows,mf);Select(upperlo,upperlo,zero,valid);
 StoreAlign(starts,lo,mf);StoreAlign(starts+64,upperlo,mf);
 Max(upperlo,upperlo,unsafe,mf);ReduceMax(upperlo,upperlo,mf);StoreAlign(maximum,upperlo,mf);
 LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
}
extern "C" [aicore] __attribute__((always_inline))
void flex_mx_label_ranges(uint64_t q,uint64_t k,uint64_t starts,uint64_t maximum,int32_t rows,int32_t cols) {
 label_ranges_vf((__ubuf__ int32_t*)q,(__ubuf__ int32_t*)k,(__ubuf__ int32_t*)starts,(__ubuf__ int32_t*)maximum,rows,cols);
}
""",
)
def _mx_label_ranges(
    q: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    k: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    starts: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    maximum: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    rows: tla.Int32,
    cols: tla.Int32,
) -> None: ...


@tla.extern(
    name="flex_mx_common_label_interval",
    source=_MX_CPP_HEADER
    + r"""__simd_vf__ inline void common_interval_vf(__ubuf__ int32_t* ranges,__ubuf__ int32_t* dst,int32_t rows) {
 MaskReg mf=CreateMask<int32_t,MaskPattern::ALL>(),valid,a,b;
 RegTensor<int32_t> starts,counts,first,count,lane,zero,one,bad,encoded,invalid;
 LoadAlign(starts,ranges);LoadAlign(counts,ranges+64);
 LoadAlign<int32_t,LoadDist::DIST_BRC_B32>(first,ranges);LoadAlign<int32_t,LoadDist::DIST_BRC_B32>(count,ranges+64);
 Arange(lane,int32_t(0));Duplicate(zero,int32_t(0));Duplicate(one,int32_t(1));Duplicate(invalid,int32_t(-1));
 Compares<int32_t,CMPMODE::LT>(valid,lane,rows,mf);
 Compare<int32_t,CMPMODE::NE>(a,starts,first,mf);Compare<int32_t,CMPMODE::NE>(b,counts,count,mf);Or(a,a,b,mf);And(a,a,valid,mf);
 Select(bad,one,zero,a);ReduceMax(bad,bad,mf);
 Add(count,count,first,mf);ShiftLefts(encoded,count,int16_t(8),mf);Or(encoded,encoded,first,mf);
 Compares<int32_t,CMPMODE::EQ>(a,bad,int32_t(0),mf);Select(encoded,encoded,invalid,a);
 if(rows==0) encoded=invalid;
 StoreAlign(dst,encoded,mf);LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
}
extern "C" [aicore] __attribute__((always_inline)) void flex_mx_common_label_interval(uint64_t src,uint64_t dst,int32_t rows) {
 common_interval_vf((__ubuf__ int32_t*)src,(__ubuf__ int32_t*)dst,rows);
}
""",
)
def _mx_common_label_interval(
    src: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    dst: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    rows: tla.Int32,
) -> None: ...


@tla.jit
def _mx_common_label_bounds(inv, safe, rows, checked: tla.Constexpr):
    word = tla.recast_ptr(safe, dtype=tla.Int32)
    _mx_common_label_interval(tla.recast_ptr(inv, dtype=tla.Int32), word, rows)
    ready = tla.flag(
        "exact_labels_common_full" if checked else "exact_labels_common_tail",
        tla.arch.VECTOR,
        tla.arch.SCALAR,
    )
    tla.set_flag(ready)
    tla.wait_flag(ready)
    scalar = tla.make_tensor(word, tla.make_layout(tla.make_shape(1), tla.make_stride(1)))
    return scalar[0]


@tla.extern(name="flex_exact_label_interval_quant", source=_MX_LABEL_INTERVAL_SOURCE)
def _mx_exact_label_interval_quant(
    s: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    p: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    ps: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    l: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    a: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    bad: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    q: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    k: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    scale: tla.Float32,
    rows: tla.Int32,
    cols: tla.Int32,
    first: tla.Int32,
    wide: tla.Int32,
    full: tla.Int32,
    first_key: tla.Int32,
    last_key: tla.Int32,
) -> None: ...


@tla.jit
def _mx_reused_label_bounds(q, k, cache, rows, cols, checked: tla.Constexpr, reuse):
    if not reuse:
        _mx_label_ranges(q, k, cache, cache + 128, rows, cols)
        ready = tla.flag(
            "label_bounds_reused_full" if checked else "label_bounds_reused_tail",
            tla.arch.VECTOR,
            tla.arch.SCALAR,
        )
        tla.set_flag(ready)
        tla.wait_flag(ready)
    state = tla.make_tensor(cache, tla.make_layout(tla.make_shape(192), tla.make_stride(1)))
    return state[128]


@tla.jit
def _mx_reused_label_halves(q, k, cache, reuse):
    if not reuse:
        mx_label_probe(q, k, cache, tla.Int32(1))
        ready = tla.flag("label_halves_reused", tla.arch.VECTOR, tla.arch.SCALAR)
        tla.set_flag(ready)
        tla.wait_flag(ready)
    state = tla.make_tensor(cache, tla.make_layout(tla.make_shape(64), tla.make_stride(1)))
    return (state[0] >> 10) & 3


@tla.jit
def _mx_exact_probabilities(
    sp,
    pp,
    psp,
    mp,
    lp,
    alpha,
    bad,
    inv,
    safe,
    gmax,
    scale,
    cols,
    partial,
    first,
    mask_mod: tla.Constexpr,
    b,
    h,
    qstart,
    sub,
    kstart,
    aux,
    scalars,
    info,
    checked: tla.Constexpr,
    label_cache,
    reuse_labels,
):
    label = _mx_trace_label_equality(mask_mod, b, h, sub, kstart, aux, scalars, info)
    if tla.const_expr(label is not None):
        if partial and cols > 0:
            if tla.const_expr(not label.wide):
                valid = max(0, min(64, info[0] - qstart - sub * 64))
                maximum = _mx_reused_label_bounds(
                    label.q, label.k, label_cache, valid, cols, checked, reuse_labels
                )
                if maximum <= 16:
                    _mx_scatter(
                        sp,
                        pp,
                        psp,
                        mp,
                        lp,
                        alpha,
                        bad,
                        label_cache,
                        tla.Int32(0),
                        valid,
                        cols,
                        tla.Int32(first),
                        maximum,
                        tla.Int32(1),
                        scale,
                    )
                else:
                    common = tla.Int32(-1)
                    if maximum <= 128:
                        common = _mx_common_label_bounds(label_cache, safe, valid, checked)
                    if common >= 0:
                        _mx_exact_label_interval_quant(
                            sp,
                            pp,
                            psp,
                            mp,
                            lp,
                            alpha,
                            bad,
                            label.q,
                            label.k,
                            scale,
                            valid,
                            cols,
                            tla.Int32(first),
                            tla.Int32(0),
                            tla.Int32(checked),
                            common & 255,
                            (common >> 8) & 255,
                        )
                    else:
                        _mx_exact_label_quant(
                            sp,
                            pp,
                            psp,
                            mp,
                            lp,
                            alpha,
                            bad,
                            label.q,
                            label.k,
                            scale,
                            valid,
                            cols,
                            tla.Int32(first),
                            tla.Int32(label.wide),
                            tla.Int32(checked),
                        )
            else:
                halves = tla.Int32(0)
                if tla.const_expr(checked):
                    halves = _mx_reused_label_halves(label.q, label.k, label_cache, reuse_labels)
                if halves > 0:
                    _mx_exact_label_interval_quant(
                        sp,
                        pp,
                        psp,
                        mp,
                        lp,
                        alpha,
                        bad,
                        label.q,
                        label.k,
                        scale,
                        tla.Int32(64),
                        cols,
                        tla.Int32(first),
                        tla.Int32(0),
                        tla.Int32(1),
                        tla.Int32(64) if halves == 2 else tla.Int32(0),
                        tla.Int32(64) if halves == 1 else tla.Int32(128),
                    )
                else:
                    _mx_exact_label_quant(
                        sp,
                        pp,
                        psp,
                        mp,
                        lp,
                        alpha,
                        bad,
                        label.q,
                        label.k,
                        scale,
                        max(0, min(64, info[0] - qstart - sub * 64)),
                        cols,
                        tla.Int32(first),
                        tla.Int32(label.wide),
                        tla.Int32(checked),
                    )
        else:
            _mx_exact_probabilities_generic(
                sp,
                pp,
                psp,
                mp,
                lp,
                alpha,
                bad,
                inv,
                safe,
                gmax,
                scale,
                cols,
                partial,
                first,
                mask_mod,
                b,
                h,
                qstart,
                sub,
                kstart,
                aux,
                scalars,
                info,
                checked,
            )
    else:
        radius = _mx_trace_query_window(mask_mod, b, h, sub, kstart, aux, scalars, info)
        if tla.const_expr(radius is not None):
            if partial and cols > 0:
                first_key, last_key, maximum = _mx_query_window_bounds(
                    radius,
                    gmax,
                    qstart + sub * 64,
                    kstart,
                    max(0, min(64, info[0] - qstart - sub * 64)),
                    cols,
                    checked,
                )
                pure_radius = _mx_trace_query_window(
                    mask_mod, b, h, sub, kstart, aux, scalars, info, True
                )
                if tla.const_expr(pure_radius is not None):
                    if maximum <= tla.Int64(16):
                        _mx_scatter(
                            sp,
                            pp,
                            psp,
                            mp,
                            lp,
                            alpha,
                            bad,
                            radius,
                            qstart + sub * 64 - kstart,
                            max(0, min(64, info[0] - qstart - sub * 64)),
                            cols,
                            tla.Int32(first),
                            maximum.to(tla.Int32),
                            tla.Int32(0),
                            scale,
                        )
                    else:
                        _mx_exact_probabilities_generic(
                            sp,
                            pp,
                            psp,
                            mp,
                            lp,
                            alpha,
                            bad,
                            inv,
                            safe,
                            gmax,
                            scale,
                            cols,
                            partial,
                            first,
                            mask_mod,
                            b,
                            h,
                            qstart,
                            sub,
                            kstart,
                            aux,
                            scalars,
                            info,
                            checked,
                            first_key,
                            last_key,
                            True,
                        )
                else:
                    _mx_exact_probabilities_generic(
                        sp,
                        pp,
                        psp,
                        mp,
                        lp,
                        alpha,
                        bad,
                        inv,
                        safe,
                        gmax,
                        scale,
                        cols,
                        partial,
                        first,
                        mask_mod,
                        b,
                        h,
                        qstart,
                        sub,
                        kstart,
                        aux,
                        scalars,
                        info,
                        checked,
                        first_key,
                        last_key,
                        True,
                    )
            else:
                _mx_exact_probabilities_generic(
                    sp,
                    pp,
                    psp,
                    mp,
                    lp,
                    alpha,
                    bad,
                    inv,
                    safe,
                    gmax,
                    scale,
                    cols,
                    partial,
                    first,
                    mask_mod,
                    b,
                    h,
                    qstart,
                    sub,
                    kstart,
                    aux,
                    scalars,
                    info,
                    checked,
                )
        else:
            _mx_exact_probabilities_generic(
                sp,
                pp,
                psp,
                mp,
                lp,
                alpha,
                bad,
                inv,
                safe,
                gmax,
                scale,
                cols,
                partial,
                first,
                mask_mod,
                b,
                h,
                qstart,
                sub,
                kstart,
                aux,
                scalars,
                info,
                checked,
            )


@tla.jit
def _mx_exact_probabilities_generic(
    sp,
    pp,
    psp,
    mp,
    lp,
    alpha,
    bad,
    inv,
    safe,
    gmax,
    scale,
    cols,
    partial,
    first,
    mask_mod: tla.Constexpr,
    b,
    h,
    qstart,
    sub,
    kstart,
    aux,
    scalars,
    info,
    checked: tla.Constexpr,
    first_key=0,
    last_key=128,
    windowed: tla.Constexpr = False,
):
    scores = tla.make_tensor(sp, tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)))
    mt = _mx_vector_tile(mp, 64)
    lt = _mx_vector_tile(lp, 64)
    at = _mx_vector_tile(alpha, 64)
    bt = _mx_vector_tile(bad, 64)
    st = _mx_vector_tile(safe, 64)
    gt = tla.make_tensor(gmax, tla.make_layout(tla.make_shape(4, 64), tla.make_stride(64, 1)))
    it = tla.make_tensor(inv, tla.make_layout(tla.make_shape(4, 64), tla.make_stride(64, 1)))
    ph0 = _mx_vector_tile(tla.recast_ptr(psp, dtype=tla.Int16), 128)
    ph1 = _mx_vector_tile(tla.recast_ptr(psp + 32, dtype=tla.Int16), 128)
    if first:
        with tla.vec.func():
            mt.store(tla.full(float("-inf"), tla.Float32))
            lt.store(tla.full(0.0, tla.Float32))
            bt.store(tla.full(0.0, tla.Float32))
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    if tla.const_expr(windowed):
        mx_zero_ub(pp, 8192)
    if cols > 0:
        pack_scale = scale if not partial and cols == 128 else tla.Float32(1.0)
        positive = scale.bitcast(tla.Int32) >= 0
        with tla.vec.func():
            blockm = tla.full(float("-inf"), tla.Float32)
            if partial:
                blockm = _mx_masked_group_max_aux(
                    scores,
                    scale,
                    gt,
                    mask_mod,
                    b,
                    h,
                    qstart,
                    sub,
                    kstart,
                    aux,
                    scalars,
                    info,
                    checked,
                    first_key,
                    last_key,
                )
            else:
                if cols == 128:
                    if positive:
                        blockm = scaled_group_max_readonly(scores, scale, gt, True)
                    else:
                        blockm = scaled_group_max_readonly(scores, scale, gt, False)
                else:
                    for key in tla.range(128):
                        item = tla.tile_view(scores, tla.make_shape(1, 64), tla.make_coord(key, 0))
                        if key < cols:
                            item.store(item.load() * scale)
                        else:
                            item.store(tla.full(float("-inf"), tla.Float32))
                    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                    blockm = scaled_group_max_readonly(scores, tla.Float32(1.0), gt, True)
            prepare_group_scales(mt, lt, at, st, ph0, ph1, it, blockm, gt)
        if tla.const_expr(windowed):
            _mx_exact_exp_pack_range(
                sp, pp, inv, safe, lp, alpha, bad, pack_scale, first_key, last_key
            )
        else:
            _mx_exact_exp_pack_update(sp, pp, inv, safe, lp, alpha, bad, pack_scale)
    else:
        mx_zero_ub(pp, 8192)
        ps = _mx_vector_tile(tla.recast_ptr(psp, dtype=tla.Int8), 256)
        with tla.vec.func():
            ps.store(tla.full(127, tla.Int8))
            at.store(tla.full(0.0, tla.Float32))


@tla.jit
def _mx_exact_accumulate(acc, pv, alpha, first):
    a = tla.make_tensor(acc, tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)))
    p = tla.make_tensor(pv, tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)))
    at = _mx_vector_tile(alpha, 64)
    if first:
        with tla.vec.func():
            for d in tla.range(128):
                dst = tla.tile_view(a, tla.make_shape(1, 64), tla.make_coord(d, 0))
                src = tla.tile_view(p, tla.make_shape(1, 64), tla.make_coord(d, 0))
                dst.store(src.load())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    else:
        with tla.vec.func():
            multiplier = at.load()
            for d in tla.range(128):
                dst = tla.tile_view(a, tla.make_shape(1, 64), tla.make_coord(d, 0))
                src = tla.tile_view(p, tla.make_shape(1, 64), tla.make_coord(d, 0))
                dst.store(dst.load() * multiplier + src.load())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


# Keep loads, normalization, casts and stores in separate register batches.
_MX_FINISH_SOURCE = r"""#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;using namespace AscendC::Reg;
constexpr CastTrait mx_short_out0={RegLayout::ZERO,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr CastTrait mx_short_out1={RegLayout::ONE,SatMode::NO_SAT,MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
template<bool FIRST>
__simd_vf__ inline void mx_exact_finish_vf(__ubuf__ float* acc,__ubuf__ float* pv,
 __ubuf__ float* alphap,__ubuf__ float* mp,__ubuf__ float* lp,__ubuf__ float* bp,
 __ubuf__ bfloat16_t* out,__ubuf__ float* lse) {
  using OUT=bfloat16_t;
  constexpr int32_t dim=128;
  MaskReg af=CreateMask<float,MaskPattern::ALL>();
  MaskReg ah=CreateMask<uint16_t,MaskPattern::ALL>();
  // Online normalization has denominator >= 1 on valid rows. Empty rows
  // substitute one, and the explicit invalid flag preserves NaN outputs.
  constexpr static LnSpecificMode logarithm={MaskMergeMode::ZEROING,LnAlgo::PRECISION_1ULP_FTZ_TRUE};
  constexpr static DivSpecificMode divide={MaskMergeMode::ZEROING,false,DivAlgo::PRECISION_1ULP_FTZ_TRUE};
  MaskReg nonempty,invalid,active;
  RegTensor<float> den,inv,one,zero,nan,m,l,bad,alpha,previous;
  LoadAlign(den,lp);LoadAlign(m,mp);LoadAlign(bad,bp);LoadAlign(alpha,alphap);
  Duplicate(one,1.f);Duplicate(zero,0.f);Duplicate(nan,__builtin_nanf(""));
  Compares<float,CMPMODE::GT>(nonempty,den,0.f,af);
  Compares<float,CMPMODE::GT>(invalid,bad,0.f,af);
  Select(den,den,one,nonempty);Div<float,&divide>(inv,one,den,af);Select(inv,nan,inv,invalid);Or(active,nonempty,invalid,af);
  Ln<float,&logarithm>(l,den,af);Add(l,l,m,af);Select(l,nan,l,invalid);StoreAlign(lse,l,af);
  for(uint16_t d=0;d<dim;d+=16) {
    RegTensor<float> r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
    RegTensor<OUT> low,high;
    LoadAlign(r0,pv+(d+0)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+0)*64);Mul(previous,previous,alpha,af);Add(r0,previous,r0,af); }
    LoadAlign(r1,pv+(d+1)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+1)*64);Mul(previous,previous,alpha,af);Add(r1,previous,r1,af); }
    LoadAlign(r2,pv+(d+2)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+2)*64);Mul(previous,previous,alpha,af);Add(r2,previous,r2,af); }
    LoadAlign(r3,pv+(d+3)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+3)*64);Mul(previous,previous,alpha,af);Add(r3,previous,r3,af); }
    LoadAlign(r4,pv+(d+4)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+4)*64);Mul(previous,previous,alpha,af);Add(r4,previous,r4,af); }
    LoadAlign(r5,pv+(d+5)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+5)*64);Mul(previous,previous,alpha,af);Add(r5,previous,r5,af); }
    LoadAlign(r6,pv+(d+6)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+6)*64);Mul(previous,previous,alpha,af);Add(r6,previous,r6,af); }
    LoadAlign(r7,pv+(d+7)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+7)*64);Mul(previous,previous,alpha,af);Add(r7,previous,r7,af); }
    LoadAlign(r8,pv+(d+8)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+8)*64);Mul(previous,previous,alpha,af);Add(r8,previous,r8,af); }
    LoadAlign(r9,pv+(d+9)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+9)*64);Mul(previous,previous,alpha,af);Add(r9,previous,r9,af); }
    LoadAlign(r10,pv+(d+10)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+10)*64);Mul(previous,previous,alpha,af);Add(r10,previous,r10,af); }
    LoadAlign(r11,pv+(d+11)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+11)*64);Mul(previous,previous,alpha,af);Add(r11,previous,r11,af); }
    LoadAlign(r12,pv+(d+12)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+12)*64);Mul(previous,previous,alpha,af);Add(r12,previous,r12,af); }
    LoadAlign(r13,pv+(d+13)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+13)*64);Mul(previous,previous,alpha,af);Add(r13,previous,r13,af); }
    LoadAlign(r14,pv+(d+14)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+14)*64);Mul(previous,previous,alpha,af);Add(r14,previous,r14,af); }
    LoadAlign(r15,pv+(d+15)*64);
    if constexpr(!FIRST) { LoadAlign(previous,acc+(d+15)*64);Mul(previous,previous,alpha,af);Add(r15,previous,r15,af); }
    Mul(r0,r0,inv,af);
    Select(r0,r0,zero,active);
    Mul(r1,r1,inv,af);
    Select(r1,r1,zero,active);
    Mul(r2,r2,inv,af);
    Select(r2,r2,zero,active);
    Mul(r3,r3,inv,af);
    Select(r3,r3,zero,active);
    Mul(r4,r4,inv,af);
    Select(r4,r4,zero,active);
    Mul(r5,r5,inv,af);
    Select(r5,r5,zero,active);
    Mul(r6,r6,inv,af);
    Select(r6,r6,zero,active);
    Mul(r7,r7,inv,af);
    Select(r7,r7,zero,active);
    Mul(r8,r8,inv,af);
    Select(r8,r8,zero,active);
    Mul(r9,r9,inv,af);
    Select(r9,r9,zero,active);
    Mul(r10,r10,inv,af);
    Select(r10,r10,zero,active);
    Mul(r11,r11,inv,af);
    Select(r11,r11,zero,active);
    Mul(r12,r12,inv,af);
    Select(r12,r12,zero,active);
    Mul(r13,r13,inv,af);
    Select(r13,r13,zero,active);
    Mul(r14,r14,inv,af);
    Select(r14,r14,zero,active);
    Mul(r15,r15,inv,af);
    Select(r15,r15,zero,active);
    Cast<OUT,float,mx_short_out0>(low,r0,af);Cast<OUT,float,mx_short_out1>(high,r1,af);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(r0),reinterpret_cast<RegTensor<uint16_t>&>(low),reinterpret_cast<RegTensor<uint16_t>&>(high),ah);
    Cast<OUT,float,mx_short_out0>(low,r2,af);Cast<OUT,float,mx_short_out1>(high,r3,af);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(r1),reinterpret_cast<RegTensor<uint16_t>&>(low),reinterpret_cast<RegTensor<uint16_t>&>(high),ah);
    Cast<OUT,float,mx_short_out0>(low,r4,af);Cast<OUT,float,mx_short_out1>(high,r5,af);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(r2),reinterpret_cast<RegTensor<uint16_t>&>(low),reinterpret_cast<RegTensor<uint16_t>&>(high),ah);
    Cast<OUT,float,mx_short_out0>(low,r6,af);Cast<OUT,float,mx_short_out1>(high,r7,af);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(r3),reinterpret_cast<RegTensor<uint16_t>&>(low),reinterpret_cast<RegTensor<uint16_t>&>(high),ah);
    Cast<OUT,float,mx_short_out0>(low,r8,af);Cast<OUT,float,mx_short_out1>(high,r9,af);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(r4),reinterpret_cast<RegTensor<uint16_t>&>(low),reinterpret_cast<RegTensor<uint16_t>&>(high),ah);
    Cast<OUT,float,mx_short_out0>(low,r10,af);Cast<OUT,float,mx_short_out1>(high,r11,af);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(r5),reinterpret_cast<RegTensor<uint16_t>&>(low),reinterpret_cast<RegTensor<uint16_t>&>(high),ah);
    Cast<OUT,float,mx_short_out0>(low,r12,af);Cast<OUT,float,mx_short_out1>(high,r13,af);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(r6),reinterpret_cast<RegTensor<uint16_t>&>(low),reinterpret_cast<RegTensor<uint16_t>&>(high),ah);
    Cast<OUT,float,mx_short_out0>(low,r14,af);Cast<OUT,float,mx_short_out1>(high,r15,af);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(r7),reinterpret_cast<RegTensor<uint16_t>&>(low),reinterpret_cast<RegTensor<uint16_t>&>(high),ah);
    Interleave(r0,r4,r0,r4);
    Interleave(r1,r5,r1,r5);
    Interleave(r2,r6,r2,r6);
    Interleave(r3,r7,r3,r7);
    Interleave(r0,r2,r0,r2);
    Interleave(r1,r3,r1,r3);
    Interleave(r4,r6,r4,r6);
    Interleave(r5,r7,r5,r7);
    Interleave(r0,r1,r0,r1);
    Interleave(r2,r3,r2,r3);
    Interleave(r4,r5,r4,r5);
    Interleave(r6,r7,r6,r7);
    StoreAlign<float,DataCopyMode::DATA_BLOCK_COPY>(reinterpret_cast<__ubuf__ float*>(out)+d/2+0*8*(dim+16)/2,r0,(dim+16)/16,af);
    StoreAlign<float,DataCopyMode::DATA_BLOCK_COPY>(reinterpret_cast<__ubuf__ float*>(out)+d/2+1*8*(dim+16)/2,r1,(dim+16)/16,af);
    StoreAlign<float,DataCopyMode::DATA_BLOCK_COPY>(reinterpret_cast<__ubuf__ float*>(out)+d/2+2*8*(dim+16)/2,r2,(dim+16)/16,af);
    StoreAlign<float,DataCopyMode::DATA_BLOCK_COPY>(reinterpret_cast<__ubuf__ float*>(out)+d/2+3*8*(dim+16)/2,r3,(dim+16)/16,af);
    StoreAlign<float,DataCopyMode::DATA_BLOCK_COPY>(reinterpret_cast<__ubuf__ float*>(out)+d/2+4*8*(dim+16)/2,r4,(dim+16)/16,af);
    StoreAlign<float,DataCopyMode::DATA_BLOCK_COPY>(reinterpret_cast<__ubuf__ float*>(out)+d/2+5*8*(dim+16)/2,r5,(dim+16)/16,af);
    StoreAlign<float,DataCopyMode::DATA_BLOCK_COPY>(reinterpret_cast<__ubuf__ float*>(out)+d/2+6*8*(dim+16)/2,r6,(dim+16)/16,af);
    StoreAlign<float,DataCopyMode::DATA_BLOCK_COPY>(reinterpret_cast<__ubuf__ float*>(out)+d/2+7*8*(dim+16)/2,r7,(dim+16)/16,af);
  }
}
extern "C" [aicore] __attribute__((always_inline))
void flex_mx_exact_finish(uint64_t a,uint64_t pv,uint64_t alpha,uint64_t m,uint64_t l,uint64_t b,uint64_t y,uint64_t e,int32_t first) {
 if(first) mx_exact_finish_vf<true>((__ubuf__ float*)a,(__ubuf__ float*)pv,(__ubuf__ float*)alpha,(__ubuf__ float*)m,(__ubuf__ float*)l,(__ubuf__ float*)b,(__ubuf__ bfloat16_t*)y,(__ubuf__ float*)e);
 else mx_exact_finish_vf<false>((__ubuf__ float*)a,(__ubuf__ float*)pv,(__ubuf__ float*)alpha,(__ubuf__ float*)m,(__ubuf__ float*)l,(__ubuf__ float*)b,(__ubuf__ bfloat16_t*)y,(__ubuf__ float*)e);
}
"""


@tla.extern(name="flex_mx_exact_finish", source=_MX_FINISH_SOURCE)
def _mx_exact_finish_vf(
    acc: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    pv: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    alpha: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    mp: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    lp: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    bad: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    out: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    lse: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    first: tla.Int32,
) -> None: ...


@tla.extern(
    name="flex_mx_stream_output",
    source=_MX_CPP_HEADER
    + r"""extern "C" [aicore] __attribute__((always_inline)) void flex_mx_stream_output(uint64_t dst,uint64_t src,int32_t rows,int32_t pitch) {
 AscendC::GlobalTensor<bfloat16_t> d;d.SetGlobalBuffer((__gm__ bfloat16_t*)dst);
 d.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
 AscendC::LocalTensor<bfloat16_t> s(AscendC::TPosition::VECCALC,uint32_t(src),64*144);
 AscendC::DataCopyExtParams p;
 p.blockCount=rows;p.blockLen=128*sizeof(bfloat16_t);p.srcStride=1;p.dstStride=(pitch-128)*sizeof(bfloat16_t);
 AscendC::DataCopyPad(d,s,p);
}
""",
)
def _mx_stream_output(
    dst: tla.Pointer[tla.BFloat16, tla.AddressSpace.gm],
    src: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    rows: tla.Int32,
    pitch: tla.Int32,
) -> None: ...


@tla.jit
def _mx_exact_finish(
    acc,
    pv,
    alpha,
    first,
    mp,
    lp,
    bad,
    ep,
    scratch,
    out,
    lse,
    qoffset,
    eoffset,
    stride,
    rows,
    streaming: tla.Constexpr = False,
):
    _mx_exact_finish_vf(acc, pv, alpha, mp, lp, bad, scratch, ep, tla.Int32(first))
    ready = tla.flag("exact_finish_ready", tla.arch.VECTOR, tla.arch.MTE3)
    tla.set_flag(ready)
    tla.wait_flag(ready)
    if rows > 0:
        output = tla.make_tensor(
            scratch, tla.make_layout(tla.make_shape(rows, 128), tla.make_stride(144, 1))
        )
        log = tla.make_tensor(ep, tla.make_layout(tla.make_shape(rows), tla.make_stride(1)))
        gm = tla.make_tensor(
            out.ptr + qoffset,
            tla.make_layout(tla.make_shape(rows, 128), tla.make_stride(stride, 1)),
        )
        lg = tla.make_tensor(
            lse.ptr + eoffset, tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
        )
        if tla.const_expr(streaming):
            _mx_stream_output(out.ptr + qoffset, scratch, rows, stride)
        else:
            tla.copy(gm, output)
        tla.copy(lg, log)


def _mx_repair_aux(meta):
    sizes = tuple(3072 if t[3] else 1536 for t in meta)
    return (
        tuple((-1 - sum(sizes[:i]), *t[1:]) for i, t in enumerate(meta)) if len(meta) <= 4 else ()
    )


@tla.kernel
def flash_fwd_mxfp8_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    qs: tla.Tensor,
    ks: tla.Tensor,
    vs: tla.Tensor,
    v_scale_offsets: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    partial_count: tla.Tensor,
    partial_index: tla.Tensor,
    full_count: tla.Tensor,
    full_index: tla.Tensor,
    cu_q: tla.Tensor,
    cu_k: tla.Tensor,
    aux_tensors: tuple,
    aux_scalars: tuple,
    batch: tla.Int32,
    q_heads: tla.Int32,
    kv_heads: tla.Int32,
    max_q: tla.Int32,
    max_k: tla.Int32,
    q_blocks: tla.Int32,
    mask_batch: tla.Int32,
    mask_heads: tla.Int32,
    scale: tla.Float32,
    total_q: tla.Int32,
    total_k: tla.Int32,
    v_scale_capacity: tla.Int32,
    varlen: tla.Constexpr,
    dense: tla.Constexpr,
    indexed: tla.Constexpr,
    score_mod: tla.Constexpr,
    mask_mod: tla.Constexpr,
    sequence_major: tla.Constexpr,
    dense_mask: tla.Constexpr,
    vector_score_mod: tla.Constexpr = None,
    ragged: tla.Constexpr = True,
    identity_score: tla.Constexpr = False,
    repair_only: tla.Constexpr = False,
    pipeline: tla.Constexpr = "general",
    group_size: tla.Constexpr = 16,
    mask_kind: tla.Constexpr = 0,
    features: tla.Constexpr = (),
    aux_cache_meta: tla.Constexpr = (),
    inline_quantize: tla.Constexpr = None,
    post_quant: tla.Constexpr = False,
):
    repair_aux = _mx_repair_aux(aux_cache_meta)
    repair_needed, arenas, channels = _attention_mxfp8_pipeline(
        q,
        k,
        v,
        qs,
        ks,
        vs,
        v_scale_offsets,
        out,
        lse,
        partial_count,
        partial_index,
        full_count,
        full_index,
        cu_q,
        cu_k,
        aux_tensors,
        aux_scalars,
        batch,
        q_heads,
        kv_heads,
        max_q,
        max_k,
        q_blocks,
        mask_batch,
        mask_heads,
        scale,
        total_q,
        total_k,
        v_scale_capacity,
        varlen,
        dense,
        indexed,
        score_mod,
        mask_mod,
        sequence_major,
        dense_mask,
        vector_score_mod,
        ragged,
        identity_score,
        min(6, group_size),
        mask_kind,
        features,
        aux_cache_meta,
        inline_quantize,
        post_quant,
        pipeline != "continuous",
        (),
        (),
        repair_only,
        pipeline == "general",
    )
    if tla.const_expr(pipeline == "continuous"):
        if repair_needed > 0:
            with tla.cube():
                tla.pipe_barrier("all")
            with tla.vector():
                tla.pipe_barrier("all")
            _attention_mxfp8_pipeline(
                q,
                k,
                v,
                qs,
                ks,
                vs,
                v_scale_offsets,
                out,
                lse,
                partial_count,
                partial_index,
                full_count,
                full_index,
                cu_q,
                cu_k,
                aux_tensors,
                aux_scalars,
                batch,
                q_heads,
                kv_heads,
                max_q,
                max_k,
                q_blocks,
                mask_batch,
                mask_heads,
                scale,
                total_q,
                total_k,
                v_scale_capacity,
                varlen,
                dense,
                indexed,
                score_mod,
                mask_mod,
                sequence_major,
                dense_mask,
                vector_score_mod,
                ragged,
                identity_score,
                min(6, group_size),
                mask_kind,
                features,
                repair_aux,
                inline_quantize,
                False,
                True,
                arenas,
                channels,
                True,
                inline_quantize is None and not repair_aux,
            )
