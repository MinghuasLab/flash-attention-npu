"""Grouped-task coverage, capacity, and NPU output/LSE regressions for v2."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_fused_task_geometry(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the shared scheduler test")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "geometry.cpp"
    source.write_text(r'''
#include "fused_task.h"
#include <algorithm>
#include <cassert>
#include <vector>

int main() {
    // Full two-vector capacity, odd GQA, group balancing, and tail dispatch.
    assert(fa_fused::HeadTile(64, 4, 8, 8) == 2);
    assert(fa_fused::HeadTile(32, 4, 8, 8) == 4);
    assert(fa_fused::HeadTile(40, 3, 9, 9) == 1);
    assert(fa_fused::HeadTile(1, 1, 9, 9) == 5);
    assert(fa_fused::MakeTaskShape(144, 16, 16, 32, 24, true).tailHeadTile == 2);
    assert(fa_fused::MakeTaskShape(272, 16, 16, 32, 24, true).tailHeadTile == 8);
    for (unsigned sq : {0, 1, 3, 7, 15, 16, 17, 31, 32, 33, 40, 63, 64, 65,
                       127, 128, 129, 144, 145, 255, 256, 272, 288, 320, 513}) {
        for (unsigned group : {1, 2, 3, 4, 5, 7, 8, 16}) {
            for (unsigned kv : {1, 2, 3, 9, 16}) {
                for (unsigned batch : {1, 23, 24, 25, 32}) {
                    for (bool fixed : {false, true}) {
                        unsigned heads = group * kv;
                        auto s = fa_fused::MakeTaskShape(sq, heads, kv, batch, 24, fixed);
                        std::vector<unsigned char> visits(sq * heads, 0);
                        for (unsigned t = 0; t < s.totalTasks; ++t) {
                            bool tail = t >= s.normTasks;
                            unsigned qb = tail ? s.normQBlocks : t / s.normHeadBlocks;
                            unsigned hb = tail ? t - s.normTasks : t % s.normHeadBlocks;
                            unsigned tile = tail ? s.tailHeadTile : s.normHeadTile;
                            unsigned rows = tail ? s.tailRows : fa_fused::Q_ROWS;
                            unsigned h0 = hb * tile;
                            assert(h0 < heads && rows > 0);
                            unsigned hn = std::min(tile, heads - h0);
                            unsigned segN = std::min(hn, group);
                            assert(hn % segN == 0);
                            unsigned l1Rows = (hn / segN) * ((rows * segN + 15) / 16 * 16);
                            assert(l1Rows <= fa_fused::Q_ROWS);
                            unsigned subRows = hn == 1 ? (rows + 1) / 2 : ((hn + 1) / 2) * rows;
                            assert(subRows <= fa_fused::SUB_CORE_ROWS);
                            for (unsigned h = h0; h < h0 + hn; ++h) {
                                // Every segment maps to exactly one KV head.
                                assert(h / group == h0 / group + (h - h0) / segN);
                                for (unsigned q = qb * 128; q < qb * 128 + rows; ++q) {
                                    assert(q < sq);
                                    assert(++visits[h * sq + q] == 1);
                                }
                            }
                        }
                        for (auto count : visits) assert(count == 1);
                    }
                }
            }
        }
    }
}
''')
    executable = tmp_path / "geometry"
    subprocess.run([compiler, "-O2", "-std=c++17", "-I",
                    str(root / "csrc/ascend910/flash_attn_npu"), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


@pytest.fixture
def npu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend 910 NPU")
    if "Ascend950" in torch.npu.get_device_name():
        pytest.skip("v2 requires Ascend 910")
    return torch


def _reference(torch, q, k, v, causal):
    group = q.shape[-2] // k.shape[-2]
    q = q.float().transpose(-3, -2)
    k = k.float().repeat_interleave(group, dim=-2).transpose(-3, -2)
    v = v.float().repeat_interleave(group, dim=-2).transpose(-3, -2)
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    if causal:
        keep = torch.arange(k.shape[-2])[None, :] <= torch.arange(q.shape[-2])[:, None] + k.shape[-2] - q.shape[-2]
        scores.masked_fill_(~keep, float("-inf"))
    out = (scores.softmax(-1).nan_to_num() @ v).transpose(-3, -2)
    lse = torch.logsumexp(scores, -1)
    lse.masked_fill_(torch.isneginf(lse), float("inf"))
    return out, lse


def _check(torch, out, lse, expected, dtype):
    tolerance = 0.003 if dtype == torch.float16 else 0.025
    torch.testing.assert_close(out.cpu().float(), expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse.cpu(), expected[1], atol=4e-5, rtol=4e-5)


def test_fused_metadata_reduces_task_count(npu):
    torch = npu
    from flash_attn_npu import get_scheduler_metadata
    from tests.test_flash_attn_npu_v2_metadata import _tiling_from_metadata

    batch = torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num
    lengths = torch.full((batch,), 577, dtype=torch.int32, device="npu")
    metadata = get_scheduler_metadata(batch, 5, 577, 18, 6, 128, lengths, qkv_dtype=torch.float16)
    tiling = _tiling_from_metadata(metadata)
    assert tiling.fusedTask == 1
    assert tiling.firstBatchTaskNum == 3  # Two KV groups per task; formerly six tasks.
    assert tiling.totalTaskNum == batch * 3
    paged = get_scheduler_metadata(batch, 5, 577, 18, 6, 128, lengths,
                                  qkv_dtype=torch.float16, page_size=128)
    assert _tiling_from_metadata(paged).fusedTask == 0


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sq,sk,hq,hk,dim", [
    (3, 577, 9, 9, 64),       # MHA across groups, odd final head, multiple KV stacks
    (5, 577, 18, 6, 128),     # GQA, 15-row segments with separate padded L1 slices
    (7, 257, 27, 9, 80),      # odd group count and a non-128 head dimension
    (17, 257, 9, 3, 64),      # odd head split between vectors
    (64, 257, 8, 2, 64),      # use both vectors' full state capacity
    (145, 577, 8, 2, 128),    # normal/tail tasks have different head counts
    (145, 0, 8, 2, 64),       # empty KV initializes each head at its actual LSE stride
    (161, 1, 8, 2, 256),      # invalid causal rows across a tail's vector row loops
    (144, 577, 16, 16, 128),  # tail dispatch heuristic
    (272, 257, 16, 16, 256),  # multiwave case keeps a larger fused tail
])
def test_fused_fixed_forward(npu, dtype_name, causal, sq, sk, hq, hk, dim):
    torch = npu
    from flash_attn_npu import flash_attn_func

    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(4951)
    cpu = [torch.randn(32, s, h, dim, generator=generator).to(dtype)
           for s, h in ((sq, hq), (sk, hk), (sk, hk))]
    out, lse, _ = flash_attn_func(*(x.npu() for x in cpu), causal=causal, return_attn_probs=True)
    _check(torch, out, lse, _reference(torch, *cpu, causal), dtype)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True])
def test_fused_bnsd_and_host_metadata(npu, dtype_name, causal):
    torch = npu
    from flash_attn_npu import flash_attn_func
    from flash_attn_npu.flash_attn_npu_interface import _flash_attn_forward

    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(1881)
    cpu = [torch.randn(32, s, h, 128, generator=generator).to(dtype)
           for s, h in ((145, 18), (257, 6), (257, 6))]
    physical = [x.transpose(1, 2).contiguous().npu() for x in cpu]
    # The public API detects BNSD storage from BSND-shaped transpose views.
    out, lse, _ = flash_attn_func(*(x.transpose(1, 2) for x in physical), causal=causal, return_attn_probs=True)
    expected = _reference(torch, *cpu, causal)
    _check(torch, out, lse, expected, dtype)
    # Omit scheduler_metadata to exercise the Host tiler instead of AICPU.
    host = _flash_attn_forward(*physical, 0.0, 128 ** -0.5, causal, -1, -1, 0.0, None, False, None, 2)
    _check(torch, host[0], host[1], expected, dtype)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True])
def test_fused_varlen_empty_and_changing_tiles(npu, dtype_name, causal):
    torch = npu
    from flash_attn_npu import flash_attn_varlen_func

    dtype = getattr(torch, dtype_name)
    qlens = [0, 1, 3, 17, 64, 128, 145, 272] * 4
    klens = [33, 577, 0, 257, 129, 128, 513, 577] * 4
    generator = torch.Generator().manual_seed(967)
    cpu = [torch.randn(total, heads, 80, generator=generator).to(dtype)
           for total, heads in ((sum(qlens), 18), (sum(klens), 6), (sum(klens), 6))]
    cuq = torch.tensor([0, *torch.tensor(qlens).cumsum(0).tolist()], dtype=torch.int32).npu()
    cuk = torch.tensor([0, *torch.tensor(klens).cumsum(0).tolist()], dtype=torch.int32).npu()
    out, lse, _ = flash_attn_varlen_func(*(x.npu() for x in cpu), cuq, cuk, max(qlens), max(klens),
                                        causal=causal, return_attn_probs=True)
    qi = ki = 0
    outputs, lses = [], []
    for nq, nk in zip(qlens, klens):
        expected = _reference(torch, cpu[0][qi:qi+nq], cpu[1][ki:ki+nk], cpu[2][ki:ki+nk], causal)
        outputs.append(expected[0])
        lses.append(expected[1])
        qi += nq
        ki += nk
    _check(torch, out, lse, (torch.cat(outputs), torch.cat(lses, dim=1)), dtype)
