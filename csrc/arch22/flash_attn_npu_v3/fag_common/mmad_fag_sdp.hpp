#ifndef FAG_COMMON_GEMM_BLOCK_MMAD_FAG_SDP_HPP
#define FAG_COMMON_GEMM_BLOCK_MMAD_FAG_SDP_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "../../flash_attn_npu_v2/fag_block.h"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Gemm::Block {

template <    
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    class BlockTileShape_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadFagSdp <
    MmadAtlasA2FagSdp<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>,
    BlockTileShape_,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2FagSdp<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using BlockTileShape = BlockTileShape_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout; // rowMajor
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout; // columnMajor
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout; // rowMajor
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc; // zN
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc; // nZ
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst; // zZ
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst; // nZ
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0_STAGES = 2;

    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / L0_STAGES;

    Arch::Resource<ArchTag> resource;

    // Check BlockTileShape
    // ==================Todo: 待白盒校验基本块的Tiling组合=========================================
    static_assert(BlockTileShape::M == 512, "BlockM must be 512!");
    static_assert(BlockTileShape::N == 512, "BlockN must be 512!");
    static_assert((BlockTileShape::K == 64) || \
                  (BlockTileShape::K == 128) || \
                  (BlockTileShape::K == 192) || \
                  (BlockTileShape::K == 256), "BlockK must be in (64, 128, 192, 256)!");

    // Specify FAG templates
    static constexpr bool ENABLE_A_B_FullLoad = std::is_same_v<BlockTileShape, L1TileShape> && \
        (L1A_STAGES == 1) && \
        (L1B_STAGES == 1);
    static constexpr bool ENABLE_A_TILED_B_FullLoad = (BlockTileShape::M > L1TileShape::M) && \
        (BlockTileShape::N == L1TileShape::N) && \
        (BlockTileShape::K == L1TileShape::K) && \
        (L1A_STAGES == 2) && \
        (L1B_STAGES == 1);
    static constexpr bool ENABLE_A_TILED_B_TILED = (BlockTileShape::M > L1TileShape::M) && \
        (BlockTileShape::N > L1TileShape::N) && \
        (BlockTileShape::K > L1TileShape::K) && \
        (L1A_STAGES == 2) && \
        (L1B_STAGES == 2);

    // Check Layout
    static_assert(std::is_same_v<LayoutA, layout::RowMajor>, "LayoutA only support RowMajor yet!"); // A矩阵仅支持[M, K]格式
    static_assert(std::is_same_v<LayoutB, layout::ColumnMajor>, "LayoutB only support ColumnMajor yet!"); // B矩阵仅支持[N, K]格式，需要在L1到L0B进行transpose
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!"); // C矩阵仅支持[M, N]格式，输出需要NZ2ND

    static_assert(std::is_same_v<LayoutAInL1, layout::zN>, "LayoutAInL1 only support zN yet!");
    static_assert(std::is_same_v<LayoutBInL1, layout::nZ>, "LayoutBInL1 only support nZ yet!");
    static_assert(std::is_same_v<LayoutAInL0, layout::zZ>, "LayoutAInL0 only support zZ yet!");
    static_assert(std::is_same_v<LayoutBInL0, layout::nZ>, "LayoutBInL0 only support nZ yet!");

    // Check L1TileShape
    static_assert((L1A_SIZE * L1A_STAGES + L1B_SIZE * L1B_STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
    static_assert((L0A_TILE_SIZE * L0_STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * L0_STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= L0C_SIZE, "L0TileShape exceeding the L0C space!");
    
    /// Construct
    CATLASS_DEVICE
    BlockMmadFagSdp(Arch::Resource<ArchTag> &ArchResource, uint32_t l1BufAddrStart = 0)
    {
        resource = ArchResource;
    }

    CATLASS_DEVICE
    void SetFlag(uint32_t l1BufAddrStart = 0)
    {
        uint32_t l1AAddrStart = l1BufAddrStart;
        uint32_t l1BAddrStart = l1BufAddrStart + L1A_SIZE * L1A_STAGES;
        /*
        L1A_STAGES L1B_STAGES L1AEVENT L1BEVENT
           1          1         [0]      [1]
           1          2         [0]      [1, 2]
           2          1         [0, 1]   [2]
           2          2         [0, 1]   [2, 3]
        */        
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AAddrStart + L1A_SIZE * i);
            l1AEventList[i] = i;
        }

        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BAddrStart + L1B_SIZE * i);
            l1BEventList[i] = i + L1A_STAGES;
        }

        if constexpr(ENABLE_A_TILED_B_FullLoad || ENABLE_A_TILED_B_TILED) {
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
        }

        if constexpr(ENABLE_A_TILED_B_TILED) {
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
        }
        /*
        L0_STAGES L0AEVENT L0BEVENT
           2       [0, 1]   [2, 3]
        */
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = i;
            l0BEventList[i] = i + L0_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }

        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadFagSdp()
    {
    }

    CATLASS_DEVICE
    void WaitFlag()
    {
        if constexpr(ENABLE_A_TILED_B_FullLoad || ENABLE_A_TILED_B_TILED) {
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
        }

        if constexpr(ENABLE_A_TILED_B_TILED) {
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
        }

        for (uint32_t i = 0; i < L0_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        GemmCoord const &actualShape) // actual shape是实际每个block分到的数据量，可能是512*512，也有可能是尾块
    {
        SetFlag();
        if constexpr(ENABLE_A_B_FullLoad) {
            GemmABFullLoad(gmBlockA, layoutA, gmBlockB, layoutB, gmBlockC, layoutC, actualShape);
        } else if constexpr(ENABLE_A_TILED_B_FullLoad) {
            GemmATiledBFullLoad(gmBlockA, layoutA, gmBlockB, layoutB, gmBlockC, layoutC, actualShape);
        } else if constexpr(ENABLE_A_TILED_B_TILED) {
            GemmATiledBTiled(gmBlockA, layoutA, gmBlockB, layoutB, gmBlockC, layoutC, actualShape);
        }
        WaitFlag();
    }

    CATLASS_DEVICE
    void GemmABFullLoad(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        GemmCoord const &actualShape
    )
    // L1A和L1B全载
    {
        // Specify single block MM size
        uint32_t actualM = actualShape.m();
        uint32_t actualN = actualShape.n();
        uint32_t actualK = actualShape.k();

        // 基本块都是block对齐的，取数据取到尾块一定是最后一次
        uint32_t MAlign = RoundUp<L1AAlignHelper::M_ALIGNED>(actualM);
        uint32_t NAlign = RoundUp<L1BAlignHelper::N_ALIGNED>(actualN);
        uint32_t KAlign = RoundUp<L1BAlignHelper::K_ALIGNED>(actualK);

        // Specify the outmost loop size
        uint32_t mL0TileCount = CeilDiv<L0TileShape::M>(MAlign);
        uint32_t nL0TileCount = CeilDiv<L0TileShape::N>(NAlign);
        uint32_t kL0TileCount = CeilDiv<L0TileShape::K>(KAlign);

        // 实际被使用的只有stride
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        // Fullload L1A
        auto l1ATensor = l1ATensorList[0];
        auto layoutTileAInGM = layoutA.GetTileLayout(MakeCoord(actualM, actualK));

        copyGmToL1A(l1ATensor, gmBlockA, layoutAInL1, layoutTileAInGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[0]); // Notify L1A MTE2 done

        // Fullload L1B
        auto l1BTensor = l1BTensorList[0];
        auto layoutTileBInGM = layoutB.GetTileLayout(MakeCoord(actualK, actualN));

        copyGmToL1B(l1BTensor, gmBlockB, layoutBInL1, layoutTileBInGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[0]); // Notify L1B MTE2 done

        // Load AL0&BL0
        for (uint32_t nL0TileIdx = 0; nL0TileIdx < nL0TileCount; nL0TileIdx++) { // DDR2L0LoopN
            uint32_t nL0 = (nL0TileIdx < nL0TileCount - 1) ?
                L0TileShape::N : (NAlign - (nL0TileCount - 1)*L0TileShape::N); // nL0 for mmad
            uint32_t actualNL0 = (nL0TileIdx < nL0TileCount - 1) ?
                L0TileShape::N : (actualN - (nL0TileCount - 1)*L0TileShape::N); // Actual nL0 for fixpipe
            for (uint32_t mL0TileIdx = 0; mL0TileIdx < mL0TileCount; mL0TileIdx++) { // DDR2L0LoopM
                uint32_t mL0 = (mL0TileIdx < mL0TileCount - 1) ?
                    L0TileShape::M : (MAlign - (mL0TileCount - 1)*L0TileShape::M); // mL0 for mmad
                uint32_t actualML0 = (mL0TileIdx < mL0TileCount - 1) ?
                    L0TileShape::M : (actualM - (mL0TileCount - 1)*L0TileShape::M); // Actual mL0 for fixpipe
                for (uint32_t kL0TileIdx = 0; kL0TileIdx < kL0TileCount; kL0TileIdx++) { // DDR2L0LoopK
                    uint32_t kL0 = (kL0TileIdx < kL0TileCount - 1) ?
                        L0TileShape::K : (KAlign - (kL0TileCount - 1)*L0TileShape::K);

                    //====================================L0A MTE1=====================================
                    // L0A Tensor
                    auto l0ATensor = l0ATensorList[l0AListId];

                    // L1A Tile Tensor
                    MatrixCoord l1ATileOffset{mL0TileIdx * L0TileShape::M, kL0TileIdx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileOffset)];

                    // L0A Layout
                    auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0, kL0);

                    // Wait for L0A MTE1 dst buffer free
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

                    // Wait for L1A MTE2 done for the first time L0A MTE1
                    if ((mL0TileIdx == 0) && (nL0TileIdx == 0) && (kL0TileIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[0]);
                    }

                    // Load the current tile from L1 to L0A
                    copyL1ToL0A(l0ATensor, l1ATile, layoutAInL0, layoutAInL1);

                    //====================================L0B MTE1=====================================
                    // L0B Tensor
                    auto l0BTensor = l0BTensorList[l0BListId];

                    // L1B Tile Tensor
                    MatrixCoord l1BTileOffset{kL0TileIdx * L0TileShape::K, nL0TileIdx * L0TileShape::N};
                    auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileOffset)];

                    // L0B Layout
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0, nL0);

                    // Wait for L0B MTE1 dst buffer free
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                    // Wait for L1B MTE2 done for the first time L1B MTE1
                    if ((mL0TileIdx == 0) && (nL0TileIdx == 0) && (kL0TileIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[0]);
                    }

                    // Load the current tile from L1 to L0B
                    copyL1ToL0B(l0BTensor, l1BTile, layoutBInL0, layoutBInL1);

                    // Notify L0B MTE1 done, mmad can be started
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    //====================================MMAD=========================================
                    // Wait for L0B MTE1 done
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (kL0TileIdx == 0);

                    // Wait for MMAD dst buffer free
                    if constexpr (!ENABLE_UNIT_FLAG) {
                        if (initC) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                        }
                    }

                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (kL0TileIdx == kL0TileCount - 1) { // L0C ready for fixpipe
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    tileMmad(l0CTensor, l0ATensor, l0BTensor, mL0, nL0, kL0, initC, unitFlag);

                    // Notify mmad done, the L0A MTE1 after next can be started
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < L0_STAGES) ? (l0AListId + 1) : 0;

                    // Notify mmad done, the L0B MTE1 after next can be started
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0_STAGES) ? (l0BListId + 1) : 0;

                }
                //====================================Fixpipe======================================
                // CGM Tile Tensor
                MatrixCoord CGMTileOffset{mL0TileIdx * L0TileShape::M, nL0TileIdx * L0TileShape::N};
                auto gmCTile = gmBlockC[layoutC.GetOffset(CGMTileOffset)];

                // CGM Tile Layout
                auto layoutTileCInGM = layoutC.GetTileLayout(MakeCoord(actualML0, actualNL0));

                // L0C Layout
                auto layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0, nL0));

                if constexpr (!ENABLE_UNIT_FLAG) {
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                    copyL0CToGm(gmCTile, l0CTensor, layoutTileCInGM, layoutCInL0);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                } else {
                    copyL0CToGm(gmCTile, l0CTensor, layoutTileCInGM, layoutCInL0, 0b11);
                }
            }
        }
    }

    CATLASS_DEVICE
    void GemmATiledBFullLoad(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        GemmCoord const &actualShape
    )
    // L1A切块 L1B全载
    {
        // Specify single block MM size
        // actual means original shape in GM which may not be block aligned
        uint32_t actualM = actualShape.m();
        uint32_t actualN = actualShape.n();
        uint32_t actualK = actualShape.k();

        // 基本块都是block对齐的，取数据取到尾块一定是最后一次
        uint32_t MAlign = RoundUp<L1AAlignHelper::M_ALIGNED>(actualM);
        uint32_t NAlign = RoundUp<L1BAlignHelper::N_ALIGNED>(actualN);
        uint32_t KAlign = RoundUp<L1BAlignHelper::K_ALIGNED>(actualK);

        // Specify the outmost loop size
        uint32_t mL0TileCount = CeilDiv<L0TileShape::M>(MAlign);
        uint32_t nL0TileCount = CeilDiv<L0TileShape::N>(NAlign);
        uint32_t kL0TileCount = CeilDiv<L0TileShape::K>(KAlign);

        // 实际被使用的只有stride
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        // Fullload L1B
        auto l1BTensor = l1BTensorList[0];
        auto layoutTileBInGM = layoutB.GetTileLayout(MakeCoord(actualK, actualN));

        copyGmToL1B(l1BTensor, gmBlockB, layoutBInL1, layoutTileBInGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[0]); // notify L1B MTE2 done

        for (uint32_t mL0TileIdx = 0; mL0TileIdx < mL0TileCount; mL0TileIdx++) { // DDR2L1LoopM
            uint32_t mL0 = (mL0TileIdx < mL0TileCount - 1) ?
                L0TileShape::M : (MAlign - (mL0TileCount - 1)*L0TileShape::M); // mL0 for MTE2 dst
            uint32_t actualML0 = (mL0TileIdx < mL0TileCount - 1) ?
                L0TileShape::M : (actualM - (mL0TileCount - 1)*L0TileShape::M); // actual mL0 for MTE2 src
            //====================================L1A MTE2=====================================
            // L1A Tensor
            auto l1ATensor = l1ATensorList[l1AListId];

            // AGM Tile Tensor
            MatrixCoord AGMTileOffset{mL0TileIdx * L0TileShape::M, 0};
            auto gmATile = gmBlockA[layoutA.GetOffset(AGMTileOffset)];

            // AGM Tile Layout
            auto layoutTileAInGM = layoutA.GetTileLayout(MakeCoord(actualML0, actualK));

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]); // Wait for L1A MTE2 dst buffer free
            copyGmToL1A(l1ATensor, gmATile, layoutAInL1, layoutTileAInGM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]); // Notify L1A MTE2 done

            for (uint32_t nL0TileIdx = 0; nL0TileIdx < nL0TileCount; nL0TileIdx++) {
                uint32_t nL0 = (nL0TileIdx < nL0TileCount - 1) ?
                    L0TileShape::N : (NAlign - (nL0TileCount - 1)*L0TileShape::N); // nL0 for mmad
                uint32_t actualNL0 = (nL0TileIdx < nL0TileCount - 1) ?
                    L0TileShape::N : (actualN - (nL0TileCount - 1)*L0TileShape::N); // Actual nL0 for fixpipe
                for (uint32_t kL0TileIdx = 0; kL0TileIdx < kL0TileCount; kL0TileIdx++) {
                    uint32_t kL0 = (kL0TileIdx < kL0TileCount - 1) ?
                        L0TileShape::K : (KAlign - (kL0TileCount - 1)*L0TileShape::K); // kL0 for mmad
                    //====================================L0A MTE1=====================================
                    // L0A Tensor
                    auto l0ATensor = l0ATensorList[l0AListId];

                    // L1A Tile Tensor
                    MatrixCoord l1ATileOffset{0, kL0TileIdx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileOffset)];

                    // L0A Layout
                    auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0, kL0);

                    // Wait for L0A MTE1 dst buffer free
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

                    // Wait for L1A MTE2 done for the first time L0A MTE1
                    if ((nL0TileIdx == 0) && (kL0TileIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                    }

                    // Load the current tile from L1 to L0A
                    copyL1ToL0A(l0ATensor, l1ATile, layoutAInL0, layoutAInL1);

                    // Notify L0A MTE1 src buffer free, the L1A MTE2 after next can be started
                    if ((nL0TileIdx == nL0TileCount - 1) && (kL0TileIdx == kL0TileCount - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                    }

                    //====================================L0B MTE1=====================================
                    // L0B Tensor
                    auto l0BTensor = l0BTensorList[l0BListId];

                    // L1B Tile Tensor
                    MatrixCoord l1BTileOffset{kL0TileIdx * L0TileShape::K, nL0TileIdx * L0TileShape::N};
                    auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileOffset)];

                    // L0B Layout
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0, nL0);

                    // Wait for L0B MTE1 dst buffer free
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                    // Wait for L1B MTE2 done for the first time L1B MTE1
                    if ((mL0TileIdx == 0) && (nL0TileIdx == 0) && (kL0TileIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[0]);
                    }

                    // Load the current tile from L1 to L0B
                    copyL1ToL0B(l0BTensor, l1BTile, layoutBInL0, layoutBInL1);

                    // Notify L0B MTE1 done, mmad can be started
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    //====================================MMAD=========================================
                    // Wait for L0B MTE1 done
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (kL0TileIdx == 0);

                    // Wait for MMAD dst buffer free
                    if constexpr (!ENABLE_UNIT_FLAG) {
                        if (initC) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                        }
                    }

                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (kL0TileIdx == kL0TileCount - 1) { // L0C ready for fixpipe
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    tileMmad(l0CTensor, l0ATensor, l0BTensor, mL0, nL0, kL0, initC, unitFlag);

                    // Notify mmad done, the L0A MTE1 after next can be started
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < L0_STAGES) ? (l0AListId + 1) : 0;

                    // Notify mmad done, the L0B MTE1 after next can be started
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0_STAGES) ? (l0BListId + 1) : 0;
                }

                //====================================Fixpipe======================================
                // CGM Tile Tensor
                MatrixCoord CGMTileOffset{mL0TileIdx * L0TileShape::M, nL0TileIdx * L0TileShape::N};
                auto gmCTile = gmBlockC[layoutC.GetOffset(CGMTileOffset)];

                // CGM Tile Layout
                auto layoutTileCInGM = layoutC.GetTileLayout(MakeCoord(actualML0, actualNL0));

                // L0C Layout
                auto layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0, nL0));

                if constexpr (!ENABLE_UNIT_FLAG) {
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                    copyL0CToGm(gmCTile, l0CTensor, layoutTileCInGM, layoutCInL0);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                } else {
                    copyL0CToGm(gmCTile, l0CTensor, layoutTileCInGM, layoutCInL0, 0b11);
                }
            }
            l1AListId = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
        }
    }

    CATLASS_DEVICE
    void GemmATiledBTiled(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        GemmCoord const &actualShape
    )
    // L1A切块 L1B切块
    {
        // Specify single block MM size
        uint32_t actualM = actualShape.m();
        uint32_t actualN = actualShape.n();
        uint32_t actualK = actualShape.k();

        // 基本块都是block对齐的，取数据取到尾块一定是最后一次
        uint32_t MAlign = RoundUp<L1AAlignHelper::M_ALIGNED>(actualM);
        uint32_t NAlign = RoundUp<L1BAlignHelper::N_ALIGNED>(actualN);
        uint32_t KAlign = RoundUp<L1BAlignHelper::K_ALIGNED>(actualK);

        // Specify the outmost loop size
        uint32_t mL0TileCount = CeilDiv<L0TileShape::M>(MAlign);
        uint32_t nL0TileCount = CeilDiv<L0TileShape::N>(NAlign);
        uint32_t kL0TileCount = CeilDiv<L0TileShape::K>(KAlign);

        // 实际被使用的只有stride
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        for (uint32_t nL0TileIdx = 0; nL0TileIdx < nL0TileCount; nL0TileIdx++) { // DDR2L1LoopN
            uint32_t nL0 = (nL0TileIdx < nL0TileCount - 1) ?
                L0TileShape::N : (NAlign - (nL0TileCount - 1)*L0TileShape::N); // nL0 for MTE2 dst
            uint32_t actualNL0 = (nL0TileIdx < nL0TileCount - 1) ?
                L0TileShape::N : (actualN - (nL0TileCount - 1)*L0TileShape::N); // actual nL0 for MTE2 src
            for (uint32_t mL0TileIdx = 0; mL0TileIdx < mL0TileCount; mL0TileIdx++) { // DDR2L1LoopM
                uint32_t mL0 = (mL0TileIdx < mL0TileCount - 1) ?
                    L0TileShape::M : (MAlign - (mL0TileCount - 1)*L0TileShape::M); // mL0 for MTE2 dst
                uint32_t actualML0 = (mL0TileIdx < mL0TileCount - 1) ?
                    L0TileShape::M : (actualM - (mL0TileCount - 1)*L0TileShape::M); // actual mL0 for MTE2 src
                for (uint32_t kL0TileIdx = 0; kL0TileIdx < kL0TileCount; kL0TileIdx++) {
                    uint32_t kL0 = (kL0TileIdx < kL0TileCount - 1) ?
                        L0TileShape::K : (KAlign - (kL0TileCount - 1)*L0TileShape::K); // kL0 for MTE2 dst
                    uint32_t actualKL0 = (kL0TileIdx < kL0TileCount - 1) ?
                        L0TileShape::K : (actualK - (kL0TileCount - 1)*L0TileShape::K); // actual kL0 for MTE2 src

                    //====================================L1A MTE2=====================================
                    // L1A Tensor
                    auto l1ATensor = l1ATensorList[l1AListId];

                    // AGM Tile Tensor
                    MatrixCoord AGMTileOffset{mL0TileIdx * L0TileShape::M, kL0TileIdx * L0TileShape::K};
                    auto gmATile = gmBlockA[layoutA.GetOffset(AGMTileOffset)];

                    // AGM Tile Layout
                    auto layoutTileAInGM = layoutA.GetTileLayout(MakeCoord(actualML0, actualKL0));

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]); // Wait for L1A MTE2 dst buffer free
                    copyGmToL1A(l1ATensor, gmATile, layoutAInL1, layoutTileAInGM);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]); // Notify L1A MTE2 done

                    //====================================L1B MTE2=====================================
                    // L1B Tensor
                    auto l1BTensor = l1BTensorList[l1BListId];

                    // BGM Tile Tensor
                    MatrixCoord BGMTileOffset{kL0TileIdx * L0TileShape::K, nL0TileIdx * L0TileShape::N};
                    auto gmBTile = gmBlockB[layoutB.GetOffset(BGMTileOffset)];

                    // BGM Tile Layout
                    auto layoutTileBInGM = layoutB.GetTileLayout(MakeCoord(actualKL0, actualNL0));

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]); // Wait for L1B MTE2 dst buffer free
                    copyGmToL1B(l1BTensor, gmBTile, layoutBInL1, layoutTileBInGM);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]); // Notify L1B MTE2 done

                    //====================================L0A MTE1=====================================
                    // L0A Tensor
                    auto l0ATensor = l0ATensorList[l0AListId];

                    // L0A Layout
                    auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0, kL0);

                    // Wait for L0A MTE1 dst buffer free
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

                    // Wait for L1A MTE2 done
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

                    // Load the current tile from L1 to L0A
                    copyL1ToL0A(l0ATensor, l1ATensor, layoutAInL0, layoutAInL1);

                    // Notify L0A MTE1 src buffer free
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);

                    //====================================L0B MTE1=====================================
                    // L0B Layout
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0, nL0);
                    // L0B Tensor
                    auto l0BTensor = l0BTensorList[l0BListId];

                    // Wait for L0B MTE1 dst buffer free
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                    // Wait for L1B MTE2 done
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

                    // Load the current tile from L1 to L0B
                    copyL1ToL0B(l0BTensor, l1BTensor, layoutBInL0, layoutBInL1);

                    // Notify L0B MTE1 src buffer free
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);

                    // Notify L0B MTE1 done, mmad can be started
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    //====================================MMAD=========================================
                    // L0C Layout
                    auto layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0, nL0));

                    // Wait for L0B MTE1 done
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (kL0TileIdx == 0);

                    // Wait for MMAD dst buffer free
                    if constexpr (!ENABLE_UNIT_FLAG) {
                        if (initC) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                        }
                    }

                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (kL0TileIdx == kL0TileCount - 1) { // L0C ready for fixpipe
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    tileMmad(l0CTensor, l0ATensor, l0BTensor, mL0, nL0, kL0, initC, unitFlag);

                    // Notify mmad done, the L0A MTE1 after next can be started
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

                    // Notify mmad done, the L0B MTE1 after next can be started
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                    l0AListId = (l0AListId + 1 < L0_STAGES) ? (l0AListId + 1) : 0;
                    l0BListId = (l0BListId + 1 < L0_STAGES) ? (l0BListId + 1) : 0;
                    l1AListId = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
                    l1BListId = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
                }

                //====================================Fixpipe======================================
                // CGM Tile Tensor
                MatrixCoord CGMTileOffset{mL0TileIdx * L0TileShape::M, nL0TileIdx * L0TileShape::N};
                auto gmCTile = gmBlockC[layoutC.GetOffset(CGMTileOffset)];

                // CGM Tile Layout
                auto layoutTileCInGM = layoutC.GetTileLayout(MakeCoord(actualML0, actualNL0));

                // L0C Layout
                auto layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0, nL0));

                if constexpr (!ENABLE_UNIT_FLAG) {
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                    copyL0CToGm(gmCTile, l0CTensor, layoutTileCInGM, layoutCInL0);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                } else {
                    copyL0CToGm(gmCTile, l0CTensor, layoutTileCInGM, layoutCInL0, 0b11);
                }
            }
        }
    }


protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    int32_t l1AEventList[L1A_STAGES]; // set/wait for (MTE2, MTE1) and (MTE1, MTE2)
    int32_t l1BEventList[L1B_STAGES]; // set/wait for (MTE2, MTE1) and (MTE1, MTE2)
    int32_t l0AEventList[L0_STAGES];  // set/wait for (M, MTE1)
    int32_t l0BEventList[L0_STAGES];  // set/wait for (M, MTE1)

    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace Catlass::Gemm::Block

#endif // FAG_COMMON_GEMM_BLOCK_MMAD_FAG_SDP_HPP
