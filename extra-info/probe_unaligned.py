"""Check flash_attn_combine for non-aligned head_dim on the NPU.

Shapes mirror the new ATK cases:
  * TND fp16 [17, 513, 5, 113]
  * BSND bf16 [5, 12, 7, 3, 200]
"""

import torch
import torch_npu  # noqa: F401

import flash_attn_npu_3


def reference(op, lse):
    m = lse.max(0, keepdim=True).values
    lse_out = m + (lse - m).exp().sum(0, keepdim=True).log()
    w = (lse - lse_out).exp()
    return (w.unsqueeze(-1) * op).sum(0), lse_out.squeeze(0)


def run(shape, dtype, uniform=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    if uniform:
        op = (torch.rand(*shape, generator=g) * 10.0 - 5.0)
    else:
        mean = float(torch.empty(1).uniform_(-100.0, 100.0, generator=g))
        std = float(torch.empty(1).uniform_(1.0, 25.0, generator=g))
        op = torch.randn(*shape, generator=g) * std + mean
    lse = torch.rand(*shape[:-1], generator=g) * 10.0 - 5.0

    ref_out, ref_lse = reference(op, lse)
    lse_kernel = lse.transpose(-1, -2).contiguous().transpose(-1, -2) if len(shape) == 4 else \
        lse.transpose(2, 3).contiguous().transpose(2, 3)
    out, out_lse = flash_attn_npu_3.flash_attn_combine(
        op.npu(), lse_kernel.npu(), out_dtype=dtype
    )
    out = out.cpu().float()
    ref = ref_out.to(dtype).float()
    d = (out - ref).abs()
    rel = d / ref.abs().clamp_min(1e-3)
    print(f"shape={tuple(shape)} {dtype} uniform={uniform}")
    print(f"  out: max_abs={d.max():.3e} max_rel={rel.max():.3e} bad(rel>1e-2)={int((rel > 1e-2).sum())}")
    if d.max() > 0:
        idx = torch.nonzero(d == d.max())[0].tolist()
        print(f"  worst idx={idx} npu={out[tuple(idx)].item():.6f} ref={ref[tuple(idx)].item():.6f}")
    lse_d = (out_lse.cpu().float() - ref_lse.float()).abs()
    print(f"  lse: max_abs={lse_d.max():.3e}")


def main():
    run([17, 513, 5, 113], torch.float16)
    run([5, 12, 7, 3, 200], torch.bfloat16)
    run([17, 513, 5, 113], torch.float16, uniform=True)
    run([5, 12, 7, 3, 200], torch.bfloat16, uniform=True)
    # control: aligned head_dim on the same pattern
    run([17, 513, 5, 112], torch.float16)
    run([5, 12, 7, 3, 192], torch.bfloat16)


def lane_profile(shape, dtype, seed=0):
    """Per-head_dim-lane error profile: shows whether the tail lanes are wrong."""
    g = torch.Generator().manual_seed(seed)
    mean = float(torch.empty(1).uniform_(-100.0, 100.0, generator=g))
    std = float(torch.empty(1).uniform_(1.0, 25.0, generator=g))
    op = torch.randn(*shape, generator=g) * std + mean
    lse = torch.rand(*shape[:-1], generator=g) * 10.0 - 5.0
    ref_out, _ = reference(op, lse)
    lse_kernel = lse.transpose(-1, -2).contiguous().transpose(-1, -2) if len(shape) == 4 else \
        lse.transpose(2, 3).contiguous().transpose(2, 3)
    out, _ = flash_attn_npu_3.flash_attn_combine(op.npu(), lse_kernel.npu(), out_dtype=dtype)
    d = (out.cpu().float() - ref_out.to(dtype).float()).abs()
    dr = d.reshape(-1, shape[-1])
    rr = ref_out.to(dtype).float().abs().reshape(-1, shape[-1])
    prof = dr.pow(2).mean(0).sqrt() / rr.pow(2).mean(0).sqrt().clamp_min(1e-9)
    bad = torch.nonzero(prof > 1e-2).flatten().tolist()
    print(f"lane profile {tuple(shape)} {dtype}: bad lanes={bad[:40]}{'...' if len(bad) > 40 else ''}"
          f" count={len(bad)}/{shape[-1]}")
    print("   first lanes:", " ".join(f"{v:.1e}" for v in prof[:8].tolist()),
          "| last lanes:", " ".join(f"{v:.1e}" for v in prof[-8:].tolist()))


if __name__ == "__main__":
    main()
    lane_profile([17, 513, 5, 113], torch.float16)
    lane_profile([5, 12, 7, 3, 200], torch.bfloat16)
    lane_profile([17, 513, 5, 112], torch.float16)
