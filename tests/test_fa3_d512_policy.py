"""CPU checks for the shared host/AICPU D512 layout and scheduling policy."""

import pathlib
import shutil
import subprocess


def test_d512_layout_and_task_partition(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    source = tmp_path / "policy.cpp"
    source.write_text(r"""
#include "csrc/ascend950/flash_attn_npu_3/fwd_policy.hpp"
#include "csrc/ascend950/flash_attn_npu_3/tilingdata.h"
#include "csrc/ascend950/flash_attn_npu_3/fa_metadata_args.h"
#include <cassert>
using namespace flash_attn::d512;
int main() {
    FAInferTilingData t{};
    t.embeddingSize = 512;
    t.embeddingSizeV = 512;
    ApplyTiling(t);
    assert(t.qBaseTile == 128 && t.kvBaseTile == 128);
    assert(t.qkL1TileKRight == 128 && t.qkL1TileKLeft == 512);
    assert(t.pL1BufNum == 3 && t.vL1BufNum == 1 && t.kL1BufNum == 2);
    // The live GO state sits after QK/P/PV workspace, for every launched Cube block.
    for (unsigned cores = 1; cores <= 64; ++cores) {
        const auto goEnd = fa_metadata::Mm1OutSize(cores) + fa_metadata::SmOnlineOutSize(cores) +
            fa_metadata::Mm2OutSize(cores) + uint64_t(cores) * GO_CORE_BYTES;
        assert(goEnd <= fa_metadata::WorkSpaceSize(cores));
    }
    auto b = MakeUbLayout();
    assert(b.loRows == 16 && b.endOffset == 209.75 * 1024);
    assert(MakeUbLayout(8192, 0).loRows == 16);
    assert(MakeUbLayout(8192, UB_BYTES).loRows == 0);
    assert(b.pOffset == 65536 && b.goOffset == 98304);
    assert(b.goOffset + GO_ROWS * 512 * 4 == b.lmOffset);
    for (unsigned core = 0; core < 56; ++core) {
        for (unsigned seq = 0; seq < 17; ++seq) {
            auto off = LoByteOffset(core, seq);
            assert(off >= uint64_t(core) * LO_CORE_BYTES);
            assert(off + LO_TILE_BYTES <= uint64_t(core + 1) * LO_CORE_BYTES);
            assert(LoByteOffset(core, seq) == LoByteOffset(core, seq + 2));
            assert(LoByteOffset(core, seq) != LoByteOffset(core, seq + 1));
        }
    }
    // Every GQA head is covered once, including the last short group. The
    // actual Q tile is capped at 128 for sequences longer than one task.
    for (unsigned q = 1; q < 260; ++q) {
        for (unsigned group = 1; group <= 128; ++group) {
            auto heads = QHeadTile(q, group);
            assert(heads >= 1 && heads <= group);
            unsigned covered = 0;
            for (unsigned first = 0; first < group; first += heads) {
                auto actual = group - first < heads ? group - first : heads;
                auto rows = (q < Q_TILE ? q : Q_TILE) * actual;
                assert(rows <= Q_TILE);
                covered += actual;
            }
            assert(covered == group);
        }
    }
}
""")
    compiler = shutil.which("g++")
    assert compiler, "g++ is required for the host/AICPU policy test"
    binary = tmp_path / "policy"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_d512_flash_decode_workspace(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    source = tmp_path / "fd_workspace.cpp"
    source.write_text(r'''
#include "csrc/ascend950/flash_attn_npu_3/tiling.cpp"
#include <cassert>
int main() {
    const int shapes[][3] = {{1,16,2},{1,128,1},{4,32,2},{16,16,2}};
    for (int dim : {256,512}) for (auto& shape : shapes) {
        int32_t qs[] = {0,shape[0]}, ks[] = {4096};
        optiling::FAInferContext ctx{};
        ctx.batch=1; ctx.numTokens=shape[0]; ctx.numHeads=shape[1]; ctx.kvHeads=shape[2];
        ctx.embeddingSize=dim; ctx.embeddingSizeV=dim; ctx.blockSize=128; ctx.numBlocks=32;
        ctx.maxNumBlocksPerBatch=32; ctx.maxQSeqlen=shape[0]; ctx.maxKvSeqlen=4096;
        ctx.qSeqlenList=qs; ctx.kvSeqlenList=ks; ctx.layout=FwdLayout::TND;
        ctx.maskType=optiling::MaskType::NO_MASK;
        FAInferTilingData base{};
        optiling::FAInferTiling plain(ctx); plain.SetCoreNum(28); plain.DoTiling(base);
        for (unsigned splits : {2U,4U}) {
            ctx.flashDecodeFlag=true; ctx.numSplits=splits;
            FAInferTilingData fd{};
            optiling::FAInferTiling split(ctx); split.SetCoreNum(28); split.DoTiling(fd);
            assert(fd.flashDecodeFlag != 0 && fd.fdPartialTaskNum > fd.fdCombineTaskNum);
            assert(fd.fdWorkspaceEnd == fd.workSpaceSize);
            assert(fd.fdPartialOOffset > fd.fdPartialLseOffset);
            if (dim == 512 && shape[0] == 1 && shape[1] == 128 && splits == 2) {
                assert((fd.fdCombineUseUniform & FD_COMBINE_UNIFORM_NO_LSE) != 0);
            }
            if (dim == 512) {
                assert(fd.fdPartialLseOffset >= base.workSpaceSize);
                const uint64_t goEnd=fd.qkOutSize+fd.smOnlineOutSize+fd.pvOutSize+
                    uint64_t(28)*128*512*sizeof(float);
                assert(goEnd <= fd.fdPartialLseOffset);
                assert(fd.qkOutSize == base.qkOutSize && fd.UpdateSize == base.UpdateSize);
            } else {
                assert(fd.fdPartialLseOffset == 0 && fd.UpdateSize == 0);
            }
        }
    }
}
''')
    compiler = shutil.which("g++") or shutil.which("clang++")
    assert compiler is not None, "A C++ compiler is required"
    binary = tmp_path / "fd_workspace"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(root), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
