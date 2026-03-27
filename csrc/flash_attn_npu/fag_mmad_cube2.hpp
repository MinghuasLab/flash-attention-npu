/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE2_HPP
#define CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE2_HPP

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
        MmadAtlasA2FAGCube2,
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
        using DispatchPolicy = MmadAtlasA2FAGCube2;
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
        using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
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
        void operator()(
            const CubeAddrInfo &addrs, __gm__ ElementA *left, __gm__ ElementB *right, __gm__ float *out,
            uint32_t &pingpongFlagL1A, uint32_t &pingpongFlagL0A, uint32_t &pingpongFlagL1B,
            uint32_t &pingpongFlagL0B)
        {
            pingPongIdx = addrs.taskId % 2;
            globalBlockOffset =  GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

            for (uint32_t i = 0; i < addrs.blockLength; ++i) {
                auto &shapeInfo = addrs.addrInfo[i];

                auto gm_a = left + (shapeInfo.out + globalBlockOffset);
                auto gm_b = right + shapeInfo.right;
                auto gm_c = out + shapeInfo.left;

                AscendC::GlobalTensor<ElementA> gLeft;
                gLeft.SetGlobalBuffer((__gm__ ElementA *)gm_a);

                AscendC::GlobalTensor<ElementB> gRight;
                gRight.SetGlobalBuffer((__gm__ ElementB *)gm_b);

                AscendC::GlobalTensor<ElementC> gOut;
                gOut.SetGlobalBuffer((__gm__ ElementC *)gm_c);

                // left matrix is (ky, kx)
                // right matrix is (kx, headdim)
                uint32_t kn = shapeInfo.kx;
                uint32_t km = shapeInfo.ky;
                uint32_t lineStride = shapeInfo.lineStride;

                int32_t l1_m_size = km;
                int32_t l1_n_size = kn;
                int32_t l1_k_size = headdim;

                int32_t l1_m_size_align = RoundUp<C0_SIZE>(l1_m_size);
                int32_t l1_n_size_align = RoundUp<C0_SIZE>(l1_n_size);
                int32_t l1_m_block_size_tail = (l1_m_size % 128) == 0 ? 128 : (l1_m_size % 128);
                int32_t l1_n_block_size_tail = (l1_n_size % 128) == 0 ? 128 : (l1_n_size % 128);
                int32_t l1_m_block_size_align_tail = (l1_m_size_align % 128) == 0 ? 128 : (l1_m_size_align % 128);
                int32_t l1_n_block_size_align_tail = (l1_n_size_align % 128) == 0 ? 128 : (l1_n_size_align % 128);

                int32_t m_loop = CeilDiv<SIZE_128>(km);
                int32_t n_loop = CeilDiv<SIZE_128>(kn);
                bool upperRight = !shapeInfo.upperRight;

                LayoutA layoutA(km, kn, 128);
                int32_t skip_num = 0;
                for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
                    int32_t n_remain = (n_loop_index == n_loop - 1) ? l1_n_block_size_tail : 128;
                    int32_t n_remain_align = (n_loop_index == n_loop - 1) ? l1_n_block_size_align_tail : 128;
                    bool l0_c_init_flag = (n_loop_index == 0);

                    LayoutB layoutB(headdim, n_remain, nheads_k * headdim);

                    // load right matrix gm (kx, headdim)-> L1B
                    AscendC::LocalTensor<ElementB>* l1_b_buf_tensor = pingpongFlagL1B ? &l1_b_pong_tensor : &l1_b_ping_tensor;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);

                    auto layoutTileB = layoutB.GetTileLayout(MakeCoord(static_cast<uint32_t>(n_remain), static_cast<uint32_t>(headdim)));
                    LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(n_remain, headdim);
                    copyGmToL1B(*l1_b_buf_tensor, gRight[n_loop_index * nheads_k * 128 * headdim], layoutBInL1, layoutTileB);

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);

                    // Load Left_GM matrix A -> L1A
                    pingpongFlagL1A = 0;
                    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                        AscendC::LocalTensor<ElementA>* l1_a_buf_tensor = pingpongFlagL1A ? &l1_a_pong_tensor : &l1_a_ping_tensor;
                        int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
                        int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                        bool is_skip = false;

                        if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
                            skip_num++;
                            is_skip = true;
                        } 

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);

                        if (!is_skip) {
                            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(static_cast<uint32_t>(m_remain), static_cast<uint32_t>(n_remain)));
                            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(m_remain, n_remain);
                            copyGmToL1A(*l1_a_buf_tensor, gLeft[(m_loop * n_loop_index + m_loop_index - skip_num) * 128 * 128], layoutAInL1, layoutTileA);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);
                        pingpongFlagL1A = 1 - pingpongFlagL1A;
                    }

                    // load L1B (n, headdim) -> L0B
                    AscendC::LocalTensor<ElementB>* l0_b_buf_tensor = pingpongFlagL0B ? &l0_b_pong_tensor : &l0_b_ping_tensor;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0B + 2 + FLAG_SHIFT);

                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(n_remain, headdim);
                    copyL1ToL0B(*l0_b_buf_tensor, *l1_b_buf_tensor, layoutBInL0, layoutBInL1);

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);

                    pingpongFlagL1A = 0;
                    pingpongFlagL0A = 0;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);

                    // do m_loop times mad with l0B常驻
                    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                        AscendC::LocalTensor<ElementA>* l1_a_buf_tensor = pingpongFlagL1A ? &l1_a_pong_tensor : &l1_a_ping_tensor;
                        AscendC::LocalTensor<ElementA>* l0_a_buf_tensor = pingpongFlagL0A ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                        AscendC::LocalTensor<float>* l0_c_buf_tensor = m_loop_index ? &l0_c_pong_tensor : &l0_c_ping_tensor;

                        int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
                        int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                        bool is_skip = false;

                        if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
                            is_skip = true;
                        }
                        // load L1A[m_loop_index] -> L0A[m_loop_index]
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0A + FLAG_SHIFT);
                        if (!is_skip) {
                            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(m_remain, n_remain);
                            LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(m_remain, n_remain);
                            copyL1ToL0A(*l0_a_buf_tensor, *l1_a_buf_tensor, layoutAInL0, layoutAInL1);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);

                        // mad (m_remain, n_remain) x (n_remain, headdim)
                        bool last_k = false;
                        last_k = (m_loop_index == 0 && upperRight) ? n_loop_index == n_loop - 2 : n_loop_index == n_loop - 1;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                        if (!is_skip) {
                            uint16_t m_modify = (m_remain == 1) ? 2 : m_remain;

                            tileMmad(*l0_c_buf_tensor, *l0_a_buf_tensor, *l0_b_buf_tensor, m_modify, headdim, n_remain, l0_c_init_flag, last_k ? 3 : 2);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0A + FLAG_SHIFT);

                        // fixp in n_loop tail block
                        if (!is_skip && last_k) {
                            AscendC::SetAtomicType<float>();
                            auto blockShape = MakeCoord(static_cast<uint32_t>(m_remain), static_cast<uint32_t>(headdim));
                            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
                            LayoutC layoutC(m_remain, headdim, nheads * headdim);
                            copyL0CToGm((gOut)[m_loop_index * nheads * 128 * headdim], *l0_c_buf_tensor, layoutC, layoutInL0C, 3);
                            AscendC::SetAtomicNone();
                        }
                        pingpongFlagL1A = 1 - pingpongFlagL1A;
                        pingpongFlagL0A = 1 - pingpongFlagL0A;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0B + 2 + FLAG_SHIFT);

                    pingpongFlagL0B = 1 - pingpongFlagL0B;
                    pingpongFlagL1B = 1 - pingpongFlagL1B;
                }
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

////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif  // ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE2_HPP
