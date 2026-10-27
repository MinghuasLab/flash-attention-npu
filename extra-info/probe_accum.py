"""Probe the fp16 rounding error of the combine kernel's split accumulation.

Compares, on identical random data:
  * CPU golden (torch fp32 sum over splits), i.e. what ATK compares against
  * exact float64 sum (best achievable pre-cast value)
  * sequential fp32 accumulation (old kernel behaviour)
  * two-way fp32 accumulation (current kernel behaviour)
  * the actual NPU kernel output
"""

import torch
import torch_npu  # noqa: F401

import flash_attn_npu_3


def reference(out_partial, lse_partial):
    lse_max = lse_partial.max(0, keepdim=True).values
    lse = lse_max + (lse_partial - lse_max).exp().sum(0, keepdim=True).log()
    weight = (lse_partial - lse).exp()
    out = (weight.unsqueeze(-1) * out_partial).sum(0)
    return out, lse.squeeze(0)


def rmse(a, b):
    d = (a.float() - b.float()).reshape(-1)
    return d.pow(2).mean().sqrt().item()


def make_inputs(splits, rows, heads, dim, seed):
    g = torch.Generator().manual_seed(seed)
    mean = float(torch.empty(1).uniform_(-100.0, 100.0, generator=g))
    std = float(torch.empty(1).uniform_(1.0, 25.0, generator=g))
    op = torch.randn(splits, rows, heads, dim, generator=g) * std + mean
    lse = torch.rand(splits, rows, heads, generator=g) * 10.0 - 5.0
    return op, lse


def main():
    splits, rows, heads, dim = 16, 4, 32, 128
    for seed in range(4):
        op, lse = make_inputs(splits, rows, heads, dim, seed)
        golden32, _ = reference(op, lse)
        golden16 = golden32.to(torch.float16)

        exact = (lse.max(0, keepdim=True).values.new_zeros(1))
        lse_max = lse.max(0, keepdim=True).values
        lse_out = lse_max + (lse - lse_max).exp().sum(0, keepdim=True).log()
        w = (lse - lse_out).exp().double()
        exact32 = (w.unsqueeze(-1) * op.double()).sum(0)
        exact16 = exact32.float().to(torch.float16)

        seq = torch.zeros(rows, heads, dim)
        wf = (lse - lse_out).exp()
        for s in range(splits):
            seq = seq + wf[s].unsqueeze(-1) * op[s]
        seq16 = seq.to(torch.float16)

        two = torch.zeros(rows, heads, dim)
        two2 = torch.zeros(rows, heads, dim)
        for s in range(splits):
            t = wf[s].unsqueeze(-1) * op[s]
            if s % 2 == 0:
                two = two + t
            else:
                two2 = two2 + t
        two16 = (two + two2).to(torch.float16)

        # The operator requires lse to be contiguous along the seqlen dim.
        lse_kernel = lse.transpose(1, 2).contiguous().transpose(1, 2)
        npu_out, npu_lse = flash_attn_npu_3.flash_attn_combine(
            op.npu(), lse_kernel.npu(), out_dtype=torch.float16
        )
        npu16 = npu_out.cpu()
        npu_lse = npu_lse.cpu()
        npu32, _ = flash_attn_npu_3.flash_attn_combine(
            op.npu(), lse_kernel.npu(), out_dtype=torch.float32
        )
        npu32 = npu32.cpu()
        # Same accumulation as the reference, but with the kernel's own LSE:
        # isolates "LSE/weight path" from "split accumulation order".
        w_npu = (lse - npu_lse.unsqueeze(0)).exp()
        seq_npuw = torch.zeros(rows, heads, dim)
        for s in range(splits):
            seq_npuw = seq_npuw + w_npu[s].unsqueeze(-1) * op[s]
        npu_lse_diff = float((npu_lse - reference(op, lse)[1]).abs().max())

        print(
            f"seed={seed} "
            f"exact={rmse(exact16, golden16):.4e} "
            f"seq={rmse(seq16, golden16):.4e} "
            f"two={rmse(two16, golden16):.4e} "
            f"npu={rmse(npu16, golden16):.4e} "
            f"seq-with-npu-lse={rmse(seq_npuw.to(torch.float16), golden16):.4e} "
            f"lse-diff={npu_lse_diff:.2e} "
            f"npu32-vs-seq={float((npu32-seq).abs().max()):.2e} "
            f"npu32-vs-two={float((npu32-(two+two2)).abs().max()):.2e} "
            f"npu32-rel-vs-two={float(((npu32-(two+two2)).abs()/(two+two2).abs()).max()):.1e} "
            f"npu32-rel-vs-golden32={float(((npu32-golden32).abs()/golden32.abs()).max()):.1e} "
            f"npu32-vs-golden32={float((npu32-golden32).abs().max()):.2e} "
            f"cast(npu32)-vs-npu16={rmse(npu32.to(torch.float16), npu16):.2e} "
            f"(fp32 pre-cast: exact-seq={float((exact32.float()-seq).abs().max()):.2e} "
            f"bit-identical npu==seq:{bool(torch.equal(npu16, seq16))} "
            f"npu==two:{bool(torch.equal(npu16, two16))})"
        )


if __name__ == "__main__":
    main()
