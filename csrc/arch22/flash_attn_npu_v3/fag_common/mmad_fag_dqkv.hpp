#ifndef FAG_COMMON_GEMM_BLOCK_MMAD_FAG_DQKV_HPP
#define FAG_COMMON_GEMM_BLOCK_MMAD_FAG_DQKV_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Gemm::Block {

template <
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    class BlockTileShape_,
    class L1ATileShape_,
    class L1BTileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadFAG <
    MmadAtlasA2FagdQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>,
    BlockTileShape_,
    L1ATileShape_,
    L1BTileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    using DispatchPolicy = MmadAtlasA2FagdQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using BlockTileShape = BlockTileShape_;
    using L1ATileShape = L1ATileShape_;
    using L1BTileShape = L1BTileShape_;
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

    // static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;

    static constexpr bool ENABLE_UNIT_FLAG = true;
    static constexpr uint32_t L1A_STAGES = 2;
    static constexpr uint32_t L1B_STAGES = 1;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;

    static constexpr uint32_t L1A_SIZE = L1ATileShape::M * L1ATileShape::K * sizeof(ElementA);  // head_dim * seq_len
    static constexpr uint32_t L1B_SIZE = L1BTileShape::N * L1BTileShape::K * sizeof(ElementB);  // seq_len * seq_len
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / L0C_STAGES;

    Arch::Resource<ArchTag> resource;
    uint32_t l1BufAddrStart;
    uint32_t eventIdStart;

    // block M, N, K
    static_assert(BlockTileShape::M == 512, "Only support BlockM==512!");
    static_assert(BlockTileShape::K == 512, "Only support BlockK==512!");
    static_assert((BlockTileShape::N == 64) || \
                  (BlockTileShape::N == 128) || \
                  (BlockTileShape::N == 192) || \
                  (BlockTileShape::N == 256), "BlockN must be in (64, 128, 192, 256)!");

    // Check Layout
    static_assert(std::is_same_v<LayoutA, layout::RowMajor> || std::is_same_v<LayoutA, layout::ColumnMajor>, "LayoutA can only be in RowMajor, or ColumnMajor yet!");
    static_assert(std::is_same_v<LayoutB, layout::RowMajor>, "LayoutB only support RowMajor yet!");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    static_assert(std::is_same_v<LayoutAInL1, layout::zN> || std::is_same_v<LayoutAInL1, layout::nZ>, "LayoutAInL1 only support zN and nZ!");
    static_assert(std::is_same_v<LayoutBInL1, layout::zN>, "LayoutBInL1 only support zN yet!");
    static_assert(std::is_same_v<LayoutAInL0, layout::zZ>, "LayoutAInL0 only support zZ yet!");
    static_assert(std::is_same_v<LayoutBInL0, layout::nZ>, "LayoutBInL0 only support nZ yet!");

    // Check L1TileShape
    static_assert((L1A_SIZE * L1A_STAGES + L1B_SIZE * L1B_STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
    static_assert((L0A_TILE_SIZE * L0AB_STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * L0AB_STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= L0C_SIZE, "L0TileShape exceeding the L0C space!");
    
    CATLASS_DEVICE
    BlockMmadFAG(Arch::Resource<ArchTag> &ArchResource, uint32_t l1BufAddrStart_ = 0, uint32_t eventIdStart_ = 0) // eventIdStart给0或4
    {
        resource = ArchResource;
        l1BufAddrStart = l1BufAddrStart_;
        eventIdStart = eventIdStart_;
    }

    CATLASS_DEVICE
    void SetFlag()
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * L1A_STAGES;
        // Init buffers
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
            l1AEventList[i] = i + eventIdStart; // 0 1  or  4 5
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
            l1BEventList[i] = i + L1A_STAGES + eventIdStart; // 2    or  6
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        
        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = i + eventIdStart;               // 0 1    or    4 5
            l0BEventList[i] = i + L0AB_STAGES + eventIdStart; // 2 3    or    6 7
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }

        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
            l0CEventList[i] = i + eventIdStart;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
        mmadSyncEventId = (eventIdStart + 7) % 8;
    }

    CATLASS_DEVICE
    ~BlockMmadFAG()
    {
    }

    CATLASS_DEVICE
    void WaitFlag()
    {
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        
        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

// tile AL1; full-load B
CATLASS_DEVICE
void operator()(
    AscendC::GlobalTensor<ElementA> gmBlockA,  // Q^T: (D, M)
    AscendC::GlobalTensor<ElementB> gmBlockB,  // dO: (M, N)
    AscendC::GlobalTensor<ElementC> gmBlockC,  // dk: (D, N)
    LayoutA layoutA, LayoutB layoutB, LayoutC layoutC,
    GemmCoord actualShape, const bool enAtomic)
{
    SetFlag();
    uint32_t actualM = actualShape.m();
    uint32_t actualN = actualShape.n();
    uint32_t actualK = actualShape.k();

    uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualM);
    uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualN);
    uint32_t kRound = RoundUp<L1BAlignHelper::K_ALIGNED>(actualK);

    LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1ATileShape::M, L1ATileShape::K);
    // B在L1中是满载的，所以其布局是实际K x actualN
    LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kRound, nRound);

    // load matrix B tile from GM to L1 (Full load)
    auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualK, actualN));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[0]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[0]);
    copyGmToL1B(l1BTensorList[0], gmBlockB, layoutBInL1, layoutTileB);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[0]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[0]);
    // 【关键】由于B是满载，这里Wait完成后，L1B数据将一直安全可用，循环内不要再有L1B的事件操作。

    AscendC::SetAtomicNone();

    uint32_t mL1TileCount = CeilDiv<L1ATileShape::M>(mRound); 
    uint32_t kL1TileCount = CeilDiv<L1ATileShape::K>(kRound); 
    
    for (uint32_t mL1TileIdx = 0; mL1TileIdx < mL1TileCount; mL1TileIdx++) { 
        uint32_t mAL1Actual = (mL1TileIdx < mL1TileCount - 1) ? L1ATileShape::M : (actualM - mL1TileIdx * L1ATileShape::M);
        uint32_t mAL1Round = RoundUp<L1AAlignHelper::M_ALIGNED>(mAL1Actual);
        for (uint32_t kAL1TileIdx = 0; kAL1TileIdx < kL1TileCount; kAL1TileIdx++) {
            uint32_t kAL1Actual = (kAL1TileIdx < kL1TileCount - 1) ? L1ATileShape::K : (actualK - kAL1TileIdx * L1ATileShape::K);

            // ================= L1A MTE2 (GM -> L1A PingPong) =================
            auto l1ATensor = l1ATensorList[l1AListId];
            MatrixCoord AGMTileOffset{mL1TileIdx * L1ATileShape::M, kAL1TileIdx * L1ATileShape::K};
            auto gmATile = gmBlockA[layoutA.GetOffset(AGMTileOffset)];
            auto layoutTileAInGM = layoutA.GetTileLayout(MakeCoord(mAL1Actual, kAL1Actual));

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]); 
            copyGmToL1A(l1ATensor, gmATile, layoutAInL1, layoutTileAInGM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]); 

            uint32_t kL0TileCount = CeilDiv<L0TileShape::K>(kAL1Actual);

            for (uint32_t kL0TileIdx = 0; kL0TileIdx < kL0TileCount; kL0TileIdx++) {
                uint32_t kL0Actual = (kL0TileIdx < kL0TileCount - 1) ? L0TileShape::K : (kAL1Actual - kL0TileIdx * L0TileShape::K); 
                uint32_t kL0Round = RoundUp<L1AAlignHelper::K_ALIGNED>(kL0Actual);
                // ================= L0A MTE1 =================
                auto l0ATile = l0ATensorList[l0AListId];  // 修正：统一使用 l0AListId
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mAL1Round, kL0Round);
                MatrixCoord l1ACoord{0, kL0TileIdx * L0TileShape::K};
                auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ACoord)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                // 第一个 L0 块需要等待当前的 L1A 搬运完成
                if (kL0TileIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                }
                copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);
                // 最后一个 L0 块搬完后，释放当前的 L1A 给后续 MTE2 覆盖
                if (kL0TileIdx == kL0TileCount - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                }

                // ================= L0B MTE1 =================
                auto l0BTile = l0BTensorList[l0BListId]; // 修正：统一使用 l0BListId
                LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Round, actualN);
                
                // 【核心修复】L1B 坐标必须包含全局的 kAL1 偏移！
                MatrixCoord l1BCoord{kAL1TileIdx * L1ATileShape::K + kL0TileIdx * L0TileShape::K, 0}; 
                auto l1BTile = l1BTensorList[0][layoutBInL1.GetOffset(l1BCoord)]; // L1B没有pingpong，恒取[0]

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                // ================= MMAD =================
                // 通知 Vector 核心可以开始算
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mmadSyncEventId);

                auto l0CTile = l0CTensorList[0];
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mmadSyncEventId);

                bool initC = (kAL1TileIdx == 0) && (kL0TileIdx == 0);
                
                // make sure fixp already finished write, GM then issue mmad to write L0C
                if (initC) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[0]);
                }

                uint8_t unitFlag = 0b00;
                if constexpr (ENABLE_UNIT_FLAG) {
                    if ((kAL1TileIdx == kL1TileCount - 1) && (kL0TileIdx == kL0TileCount - 1)) {
                        unitFlag = 0b11;
                    } else {
                        unitFlag = 0b10;
                    }
                }
                tileMmad(l0CTile, l0ATile, l0BTile, mAL1Round, actualN, kL0Actual, initC, unitFlag);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
                
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
            } // end of kL0 loop

            // 【核心修复】L1A 也是沿着 K 轴交替流水线的，所以游标更新必须在 K 循环里！
            l1AListId = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
        } // end of kL1 loop

        // ================= COPY OUT =================
        // Layout 修改：只 Copy 当前的实际 m 大小，而不是全 Block 大小
        LayoutC layoutBlock = layoutC.GetTileLayout(MakeCoord(mAL1Actual, actualN));
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mAL1Round, actualN));
        MatrixCoord gmCoordC{mL1TileIdx * L1ATileShape::M, 0};
        auto gmTileC = gmBlockC[layoutC.GetOffset(gmCoordC)];

        // 【核心修复】增加 L0C 到 GM 的安全同步
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[0]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[0]);

        if (enAtomic) {
            AscendC::SetAtomicType<float>();
            copyL0CToGm(gmTileC, l0CTensorList[0], layoutBlock, layoutInL0C, 0b11);
            AscendC::SetAtomicNone();
        } else {
            copyL0CToGm(gmTileC, l0CTensorList[0], layoutBlock, layoutInL0C, 0b11);
        }

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[0]);
    } // end of mL1 loop

    WaitFlag();
}

protected:
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];

    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0AB_STAGES];
    int32_t l0BEventList[L0AB_STAGES];
    int32_t l0CEventList[L0C_STAGES];
    int32_t mmadSyncEventId{0};

    uint32_t l1AListId{0};
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

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_FA_DK_HPP
