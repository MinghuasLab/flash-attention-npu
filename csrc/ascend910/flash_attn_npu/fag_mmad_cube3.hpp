/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE3_HPP
#define CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE3_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "fag_block.h"
#include "fag_common/common_header.h"

////////////////////////////////////////////////////////////////////
namespace Catlass::Gemm::Block
{
    ////////////////////////////////////////////////////////////////////

    template <
        class L1TileShape_,
        class L0TileShape_,
        class AType_,
        class BType_,
        class CType_,
        class BiasType_,
        class TileCopy_,
        class TileMmad_>
    struct BlockMmad<
        MmadAtlasA2FAGCube3,
        L1TileShape_,
        L0TileShape_,
        AType_,
        BType_,
        CType_,
        BiasType_,
        TileCopy_,
        TileMmad_>
    {
    public:
        // Type Aliases
        using DispatchPolicy = MmadAtlasA2FAGCube3;
        using ArchTag = typename DispatchPolicy::ArchTag;
        using L1TileShape = L1TileShape_;
        using L0TileShape = L0TileShape_;
        using ElementA = typename AType_::Element;
        using LayoutA = typename AType_::Layout;
        using ElementB = typename BType_::Element;
        using LayoutB = typename BType_::Layout;
        using ElementC = typename CType_::Element;
        using LayoutC = typename CType_::Layout;
        using TileMmad = TileMmad_;
        using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
        using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
        using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
        using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
        using ElementAccumulator =
            typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
        using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
        using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
        using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
        using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
        using LayoutCInL0 = layout::zN;

        using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
        using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

        static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
        static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
        static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
        static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
        static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
        static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
        static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;

        static const uint32_t C0_SIZE = 16;
        static const uint32_t SIZE_16 = 16;
        static const uint32_t SIZE_32 = 32;
        static const uint32_t SIZE_64 = 64;
        static const uint32_t SIZE_128 = 128;
        static const uint32_t SIZE_256 = 256;
        static const uint32_t SIZE_LONG_BLOCK = 16384;
        static const uint32_t SIZE_384 = 384;
        static const uint32_t SIZE_ONE_K = 1024;
        static const uint32_t BLOCK_WORKSPACE = 16 * 128 * 128;

        /// Construct
        CATLASS_DEVICE
        BlockMmad(Arch::Resource<ArchTag> &resource, uint64_t nheadsIn, uint64_t nheadsKIn, uint64_t headDimIn)
        {
            nheads = nheadsIn;
            nheads_k = nheadsKIn;
            headdim = headDimIn;
            globalBlockOffset = GetBlockIdx() * 16 * 128 * 128;

            AscendC::SetLoadDataPaddingValue<uint64_t>(0);
            uint64_t config = 0x1;
            AscendC::SetNdParaImpl(config);

            // init L1 tensor
            l1_a_ping_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1_a_pong_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_128 * SIZE_ONE_K);
            l1_b_ping_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_256 * SIZE_ONE_K);
            l1_b_pong_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_384 * SIZE_ONE_K);

            // init L0A/L0B/L0C tensor
            l0_a_ping_tensor = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0_a_pong_tensor = resource.l0ABuf.template GetBufferByByte<ElementA>(SIZE_32 * SIZE_ONE_K);
            l0_b_ping_tensor = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0_b_pong_tensor = resource.l0BBuf.template GetBufferByByte<ElementB>(SIZE_32 * SIZE_ONE_K);

            l0_c_ping_tensor = resource.l0CBuf.template GetBufferByByte<float>(0);
            l0_c_pong_tensor = resource.l0CBuf.template GetBufferByByte<float>(SIZE_64 * SIZE_ONE_K);
        }

        /// Destructor
        CATLASS_DEVICE
        ~BlockMmad()
        {
        }

        CATLASS_DEVICE
        void operator()(const CubeAddrInfo &addrs, __gm__ ElementA *left, __gm__ ElementB *right, __gm__ float *out,
                        uint32_t &pingpongFlagL1A, uint32_t &pingpongFlagL0A, uint32_t &pingpongFlagL1B,
                        uint32_t &pingpongFlagL0B, uint32_t &pingpongFlagC)
        {
            pingPongIdx = addrs.taskId % 2;
            globalBlockOffset =  GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

            for (uint32_t i = 0; i < addrs.blockLength; ++i) {
                pingpongFlagL0B = 0;
                int32_t ping_pong_flag_l0_b_last = 0;
                int32_t skip_num = 0;    
                auto &shapeInfo = addrs.addrInfo[i];

                uint32_t kn = shapeInfo.kx;
                uint32_t km = shapeInfo.ky;
                uint32_t lineStride = shapeInfo.lineStride;

                uint32_t l1_m_size = km;
                uint32_t l1_n_size = kn;
                uint32_t l1_k_size = headdim;

                uint32_t l1_m_size_align = RoundUp<C0_SIZE>(l1_m_size);
                uint32_t l1_n_size_align = RoundUp<C0_SIZE>(l1_n_size);
                uint32_t l1_m_block_size_tail = (l1_m_size % 128) == 0 ? 128 : (l1_m_size % 128);
                uint32_t l1_n_block_size_tail = (l1_n_size % 128) == 0 ? 128 : (l1_n_size % 128);
                uint32_t l1_m_block_size_align_tail = (l1_m_size_align % 128) == 0 ? 128 : (l1_m_size_align % 128);
                uint32_t l1_n_block_size_align_tail = (l1_n_size_align % 128) == 0 ? 128 : (l1_n_size_align % 128);

                uint32_t m_loop = CeilDiv<SIZE_128>(km);
                uint32_t n_loop = CeilDiv<SIZE_128>(kn);

                // *2: half* view into float mm workspace (basic cube3_op.h); P/dS overwrite mm1/mm2.
                __gm__ ElementA* gm_a = left + (shapeInfo.out + globalBlockOffset) * 2;
                __gm__ ElementB* gm_b = right + shapeInfo.left;
                __gm__ float* gm_out = out + shapeInfo.right;

                AscendC::GlobalTensor<ElementA> gLeft;
                gLeft.SetGlobalBuffer((__gm__ ElementA *)gm_a);

                AscendC::GlobalTensor<ElementB> gRight;
                gRight.SetGlobalBuffer((__gm__ ElementB *)gm_b);

                AscendC::GlobalTensor<ElementC> gOut;
                gOut.SetGlobalBuffer((__gm__ ElementC *)gm_out);

                bool lowerLeft = !shapeInfo.lowerLeft;
                bool upperRight = !shapeInfo.upperRight;

                LocalTensor<ElementB> *l1_b_buf_tensor = pingpongFlagL1B ? &l1_b_pong_tensor : &l1_b_ping_tensor;
                
                LayoutB layoutB(km, headdim, nheads * headdim);
                // move right matrix Q(ky, headdim) from GM to L1B
                // Use aligned M in L1 (basic dstNzC0Stride) so ky=1 pads to C0=16.
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);

                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(l1_m_size, l1_k_size));
                LayoutBInL1 layoutBInL1 =
                    LayoutBInL1::template MakeLayout<ElementB>(l1_m_size_align, l1_k_size);
                copyGmToL1B(*l1_b_buf_tensor, gRight, layoutBInL1, layoutTileB);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);
                
                // Match basic cube3_op.h: load L0B per (n,m) from L1B (no one-shot preload).
                AscendC::LoadData2dParams loadParamsB{0, 128, 128, 0, 0, true, 0};
                for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
                    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                        LocalTensor<ElementA>* l1_a_buf_tensor = pingpongFlagL1A ? &l1_a_pong_tensor : &l1_a_ping_tensor;
                        LocalTensor<ElementA>* l0_a_buf_tensor = pingpongFlagL0A ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                        LocalTensor<ElementB>* l0_b_buf_tensor =
                            ping_pong_flag_l0_b_last ? &l0_b_ping_tensor : &l0_b_pong_tensor;
                        LocalTensor<float>* l0_c_buf_tensor = pingpongFlagC ? &l0_c_pong_tensor : &l0_c_ping_tensor;

                        uint32_t real_m = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
                        uint32_t real_n = (n_loop_index == n_loop - 1) ? l1_n_block_size_tail : 128;
                        uint32_t real_m_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                        uint32_t real_n_align = (n_loop_index == n_loop - 1) ? l1_n_block_size_align_tail : 128;

                        bool init_c = (m_loop_index == 0);
                        bool out_c = (m_loop_index == (m_loop - 1));
                        if (m_loop_index != 0 && n_loop_index == n_loop - 1 && upperRight) {
                            init_c = true;
                        }

                        bool is_skip = false;
                        if (n_loop_index == (n_loop - 1) && m_loop_index == 0 && upperRight) {
                            is_skip = true;
                        }
                        if (n_loop_index == 0 && m_loop_index == (m_loop - 1) && lowerLeft) {
                            is_skip = true;
                        }
                        if (is_skip) {
                            skip_num++;
                        }

                        if (!is_skip) {
                            // L1B → L0B (basic: srcStride = l1_m_size_align/16)
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_last + 2 + FLAG_SHIFT);
                            int32_t m_remain_align =
                                (m_loop_index == m_loop - 1) ? static_cast<int32_t>(l1_m_block_size_align_tail) : 128;
                            int32_t l1_b_buf_offset = 128 * 16 * static_cast<int32_t>(m_loop_index);
                            loadParamsB.repeatTimes = static_cast<uint16_t>(l1_k_size / C0_SIZE);
                            loadParamsB.srcStride = static_cast<uint16_t>(l1_m_size_align / C0_SIZE);
                            for (int32_t j = 0; j < m_remain_align / static_cast<int32_t>(C0_SIZE); ++j) {
                                AscendC::LoadData(
                                    (*l0_b_buf_tensor)[j * static_cast<int32_t>(l1_k_size) * static_cast<int32_t>(C0_SIZE)],
                                    (*l1_b_buf_tensor)[l1_b_buf_offset + j * 16 * 16], loadParamsB);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ping_pong_flag_l0_b_last + 2);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ping_pong_flag_l0_b_last + 2);
                        }

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);
                        if (!is_skip) {
                            // half alias: mUp/mDown + 2*tile (basic vec/cube3)
                            uint32_t tileIdx = n_loop_index * m_loop + m_loop_index - skip_num;
                            int32_t mUp = (static_cast<int32_t>(real_m) + 1) / 2;
                            int32_t mDown = static_cast<int32_t>(real_m) - mUp;
                            auto gmTile = gLeft[2 * tileIdx * SIZE_128 * SIZE_128];
                            AscendC::Nd2NzParams nd2nzParams;
                            nd2nzParams.ndNum = 1;
                            nd2nzParams.nValue = mUp;
                            nd2nzParams.dValue = real_n;
                            nd2nzParams.srcNdMatrixStride = 0;
                            nd2nzParams.srcDValue = 128;
                            nd2nzParams.dstNzC0Stride = real_m_align;
                            nd2nzParams.dstNzNStride = 1;
                            nd2nzParams.dstNzMatrixStride = 0;
                            AscendC::DataCopy(*l1_a_buf_tensor, gmTile, nd2nzParams);
                            if (mDown > 0) {
                                nd2nzParams.nValue = mDown;
                                AscendC::DataCopy((*l1_a_buf_tensor)[mUp * C0_SIZE],
                                    gmTile[2 * mUp * 128], nd2nzParams);
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0A + FLAG_SHIFT);
                        if (!is_skip) {
                            // Match basic: fractal LoadData (handles real_m=1 / ky=1).
                            AscendC::LoadData2dParams loadParamsA{0, 1, 1, 0, 0, true, 0};
                            uint32_t m_c0_loop = (real_m + C0_SIZE - 1) / C0_SIZE;
                            uint32_t n_c0_loop = (real_n + C0_SIZE - 1) / C0_SIZE;
                            loadParamsA.repeatTimes = static_cast<uint16_t>(m_c0_loop * n_c0_loop);
                            loadParamsA.srcStride = 1;
                            AscendC::LoadData(*l0_a_buf_tensor, *l1_a_buf_tensor, loadParamsA);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                        if (!is_skip) {
                            int unit_flag = out_c ? 0b11 : 0b10;
                            // Preserve the corrected baseline matrix-mode padding for one-row tails.
                            uint32_t m_mad = (real_n == 1) ? 16 : real_n;
                            tileMmad(*l0_c_buf_tensor, *l0_a_buf_tensor, *l0_b_buf_tensor, m_mad, l1_k_size, real_m,
                                init_c, unit_flag);
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_last + 2 + FLAG_SHIFT);
                            AscendC::PipeBarrier<PIPE_M>();
                            ping_pong_flag_l0_b_last = 1 - ping_pong_flag_l0_b_last;
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0A + FLAG_SHIFT);

                        if (!is_skip && out_c) {
                            AscendC::SetAtomicType<float>();
                            auto blockShape = MakeCoord(real_n, l1_k_size);
                            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
                            LayoutC layoutC(real_n, l1_k_size, nheads_k * headdim);
                            copyL0CToGm(gOut[n_loop_index * 128 * nheads_k * headdim], *l0_c_buf_tensor, layoutC,
                                layoutInL0C, 3);
                            AscendC::SetAtomicNone();
                        }
                        pingpongFlagL1A = 1 - pingpongFlagL1A;
                        pingpongFlagL0A = 1 - pingpongFlagL0A;
                    }
                    pingpongFlagC = 1 - pingpongFlagC;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);
                pingpongFlagL1B = 1 - pingpongFlagL1B;
            }
        }

    protected:
        /// Data members
        TileMmad tileMmad;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;

        LocalTensor<ElementA> l1_a_ping_tensor;
        LocalTensor<ElementA> l1_a_pong_tensor;
        LocalTensor<ElementB> l1_b_ping_tensor;
        LocalTensor<ElementB> l1_b_pong_tensor;

        // L0A L0B
        LocalTensor<ElementA> l0_a_ping_tensor;
        LocalTensor<ElementA> l0_a_pong_tensor;
        LocalTensor<ElementB> l0_b_ping_tensor;
        LocalTensor<ElementB> l0_b_pong_tensor;

        // L0C
        LocalTensor<float> l0_c_ping_tensor;
        LocalTensor<float> l0_c_pong_tensor;

        uint32_t FLAG_SHIFT = 3;

        uint64_t nheads;
        uint64_t nheads_k;
        uint64_t headdim;

        uint32_t pingPongIdx = 0;
        uint64_t globalBlockOffset = 0;
    };
} // namespace Catlass::Gemm::Block

#endif  // ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE3_HPP```
