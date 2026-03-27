/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE1_HPP
#define CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE1_HPP

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
        MmadAtlasA2FAGCube1,
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
        using DispatchPolicy = MmadAtlasA2FAGCube1;
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
            cube1Cnt = 0;
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
        void cube1_base_matmul(
            LocalTensor<ElementA> *l1_a_tensor, LocalTensor<ElementB> *l1_b_tensor,
            GlobalTensor<ElementC> *gOut, uint32_t &pingpongFlagL1A,
            uint32_t &pingpongFlagL0A, uint32_t &pingpongFlagL1B,
            uint32_t &pingpongFlagL0B, uint32_t &pingpongFlagC,
            int32_t l1_m_size, int32_t l1_n_size, bool upper_right_flag)
        {
            int32_t l1_m_size_ = l1_m_size;
            int32_t l1_n_size_ = l1_n_size;

            uint32_t l1_m_size_align_ = RoundUp<C0_SIZE>(l1_m_size_);
            uint32_t l1_n_size_align_ = RoundUp<C0_SIZE>(l1_n_size_);

            uint32_t m0_ = 128;
            uint32_t n0_ = 128;
            uint32_t k0_ = 128;

            uint32_t m_mad_ = 128;
            uint32_t n_mad_ = 128;
            uint32_t k_mad_ = 128;

            int32_t dst_n_size_ = 128;

            for (int n_offset = 0; n_offset < l1_n_size_; n_offset += 128) {
                n_mad_ = Min((l1_n_size_ - n_offset), 128);
                n0_ = RoundUp<C0_SIZE>(n_mad_);

                LocalTensor<ElementB>* l0_b_tensor = pingpongFlagL0B ? &l0_b_pong_tensor : &l0_b_ping_tensor;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + pingpongFlagL0B + 2);
                LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(k0_, n_mad_);
                LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(k0_, n_mad_);
                copyL1ToL0B(*l0_b_tensor, (*l1_b_tensor)[n_offset * SIZE_16], layoutBInL0, layoutBInL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);

                for (int m_offset = 0; m_offset < l1_m_size_; m_offset += SIZE_128) {
                    m_mad_ = Min((l1_m_size_ - m_offset), 128);
                    m0_ = RoundUp<C0_SIZE>(m_mad_);

                    bool l0_skip_flag = (upper_right_flag && m_offset == 0);
                    LocalTensor<ElementA>* l0_a_tensor = pingpongFlagL0A ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                    LocalTensor<float>* l0_c_tensor = pingpongFlagC ? &l0_c_pong_tensor : &l0_c_ping_tensor;

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + pingpongFlagL0A);
                    if (!l0_skip_flag) {
                        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1_m_size_align_, k0_);
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(m_mad_, k0_);
                        copyL1ToL0A(*l0_a_tensor, (*l1_a_tensor)[m_offset * SIZE_16], layoutAInL0, layoutAInL1);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                    if (!l0_skip_flag) {
                        AscendC::MmadParams commonMadParams {
                            BASE_BLOCK_LENGTH,
                            BASE_BLOCK_LENGTH,
                            BASE_BLOCK_LENGTH,
                            3,
                            false,
                            true
                        };

                        uint16_t m_modify = (m_mad_ == 1) ? 2 : m_mad_;
                        tileMmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, m_modify, n_mad_, k_mad_, true, 3);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3 + pingpongFlagL0A);
                    auto out_offset = cube1Cnt * SIZE_LONG_BLOCK;

                    if (!l0_skip_flag) {
                        auto blockShape = MakeCoord(m_mad_, n_mad_);
                        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
                        LayoutC layoutC(m_mad_, n_mad_, 128);
                        copyL0CToGm((*gOut)[out_offset], *l0_c_tensor, layoutC, layoutInL0C, 3);
                        cube1Cnt++;
                    }

                    pingpongFlagC = 1 - pingpongFlagC;
                    pingpongFlagL0A = 1 - pingpongFlagL0A;
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3 +  pingpongFlagL0B + 2);
                pingpongFlagL0B = 1 - pingpongFlagL0B;
            }
        }

        CATLASS_DEVICE
        void operator()(const CubeAddrInfo &addrs, __gm__ ElementA *left, __gm__ ElementB *right, __gm__ float *out,
                        uint32_t &pingpongFlagL1A, uint32_t &pingpongFlagL0A, uint32_t &pingpongFlagL1B,
                        uint32_t &pingpongFlagL0B, uint32_t &pingpongFlagC)
        {
            pingPongIdx = addrs.taskId % 2;
            globalBlockOffset =  GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;
            for (uint32_t i = 0; i < addrs.blockLength; ++i) {
                cube1Cnt = 0;
                auto &shapeInfo = addrs.addrInfo[i];
                uint32_t km = shapeInfo.ky;
                uint32_t kn = shapeInfo.kx;
                int32_t lineStride = shapeInfo.lineStride;

                uint32_t src_k_size_ = headdim;

                uint32_t l1_m_size_ = km;
                uint32_t l1_n_size_ = kn;
                uint32_t l1_k_size_ = src_k_size_;
                uint32_t l1_m_size_align_ = RoundUp<C0_SIZE>(l1_m_size_);
                uint32_t l1_n_size_align_ = RoundUp<C0_SIZE>(l1_n_size_);

                uint32_t m_loop = CeilDiv<SIZE_256>(km);
                uint32_t n_loop = CeilDiv<SIZE_128>(kn);

                auto gm_a = left + shapeInfo.left;
                auto gm_b = right + shapeInfo.right;
                auto gm_c = out + shapeInfo.out + globalBlockOffset;

                AscendC::GlobalTensor<ElementA> gLeft;
                gLeft.SetGlobalBuffer((__gm__ ElementA *)gm_a);

                AscendC::GlobalTensor<ElementB> gRight;
                gRight.SetGlobalBuffer((__gm__ ElementB *)gm_b);

                AscendC::GlobalTensor<ElementC> gOut;
                gOut.SetGlobalBuffer((__gm__ ElementC *)gm_c);

                bool upperRight = !shapeInfo.upperRight;
                bool lowerLeft = !shapeInfo.lowerLeft;

                LayoutA layoutA(km, headdim, nheads * headdim);
                LayoutB layoutB(headdim, kn, nheads_k * headdim);
                LayoutC layoutC(km, kn, 128);
                for (int m_index = 0; m_index < m_loop; m_index++) {
                    LocalTensor<ElementA>* l1_a_tensor = pingpongFlagL1A ? &l1_a_pong_tensor : &l1_a_ping_tensor;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(km, static_cast<uint32_t>(headdim)));
                    LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1_m_size_, headdim);                                        
                    copyGmToL1A(*l1_a_tensor, gLeft, layoutAInL1, layoutTileA);

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);

                    for (int n_index = 0; n_index < n_loop; n_index++) {
                        l1_n_size_ = (n_index == n_loop - 1) ? (kn - n_index * 128) : 128;
                        l1_n_size_align_ = RoundUp<C0_SIZE>(l1_n_size_);
                        bool upper_right_flag = (upperRight && n_index == n_loop - 1);

                        LocalTensor<ElementB>* l1_b_tensor = pingpongFlagL1B ? &l1_b_pong_tensor : &l1_b_ping_tensor;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);
                        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(l1_k_size_, l1_n_size_));
                        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(l1_k_size_, l1_n_size_);
                        copyGmToL1B(*l1_b_tensor, gRight[n_index * 128 * src_k_size_ * nheads_k], layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);

                        cube1_base_matmul(l1_a_tensor, l1_b_tensor, &gOut, pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B,
                            pingpongFlagC, l1_m_size_, l1_n_size_, upper_right_flag);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);
                        pingpongFlagL1B = 1 - pingpongFlagL1B;
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);
                    pingpongFlagL1A = 1 - pingpongFlagL1A;
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

        uint32_t cube1Cnt = 0;
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

        uint64_t nheads;
        uint64_t nheads_k;
        uint64_t headdim;

        uint32_t pingPongIdx = 0;
        uint64_t globalBlockOffset = 0;
    };

    ////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif  // ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE1_HPP```
