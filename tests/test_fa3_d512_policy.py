"""CPU checks for the shared host/AICPU D512 layout and scheduling policy."""

import pathlib
import shutil
import subprocess


def test_d512_layout_and_task_partition(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    source = tmp_path / "policy.cpp"
    source.write_text(r"""
#include "csrc/ascend950/flash_attn_npu_3/d512/config.hpp"
#include "csrc/ascend950/flash_attn_npu_3/tilingdata.h"
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
