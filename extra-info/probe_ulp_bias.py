"""ULP-level analysis of the two failing ATK shapes.

  id 56: TND bf16 [32, 16, 2, 256]
  id 90: BSND fp16 [8, 8, 1, 32, 128]

For each case compare:
  * CPU golden (fp32 sum -> cast), i.e. what ATK compares against
  * float64 exact -> cast (best achievable)
  * NPU output
and report the fraction of elements that land on a different float, with the
sign of the difference in ULP.
"""

import torch
import torch_npu  # noqa: F401

import flash_attn_npu_3


def reference(op, lse):
    m = lse.max(0, keepdim=True).values
    lse_out = m + (lse - m).exp().sum(0, keepdim=True).log()
    w = (lse - lse_out).exp()
    return (w.unsqueeze(-1) * op).sum(0), lse_out.squeeze(0), w


def ulp_stats(name, ref16, other16, exact16=None):
    diff = other16.float() - ref16.float()
    ref32 = ref16.float()
    # ULP of the reference value in its own dtype
    nxt = torch.nextafter(ref16, torch.full_like(ref16, float("inf")))
    ulp = (nxt.float() - ref32).abs().clamp_min(1e-30)
    steps = diff / ulp
    changed = (diff != 0).float().mean().item()
    up = (steps > 0.5).float().mean().item()
    down = (steps < -0.5).float().mean().item()
    print(f"  {name}: 不同元素占比={changed*100:.2f}% | +1ULP={up*100:.2f}% | -1ULP={down*100:.2f}% "
          f"| RMS误差={diff.pow(2).mean().sqrt().item():.3e}")
    if exact16 is not None:
        de = (exact16.float() - ref16.float())
        print(f"     exact vs golden: 不同元素占比={(de != 0).float().mean().item()*100:.2f}% "
              f"| RMS={de.pow(2).mean().sqrt().item():.3e}")


def run(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    mean = float(torch.empty(1).uniform_(-100.0, 100.0, generator=g))
    std = float(torch.empty(1).uniform_(1.0, 25.0, generator=g))
    op = torch.randn(*shape, generator=g) * std + mean
    lse = torch.rand(*shape[:-1], generator=g) * 10.0 - 5.0
    ref32, ref_lse, w = reference(op, lse)
    ref16 = ref32.to(dtype)
    exact16 = (w.double().unsqueeze(-1) * op.double()).sum(0).float().to(dtype)
    if len(shape) == 4:
        t = lse.transpose(-1, -2)
        buf = torch.empty(t.shape, dtype=t.dtype)
        buf.copy_(t)
        lk = buf.transpose(-1, -2)
    else:
        t = lse.transpose(2, 3)
        buf = torch.empty(t.shape, dtype=t.dtype)
        buf.copy_(t)
        lk = buf.transpose(2, 3)
    out, out_lse = flash_attn_npu_3.flash_attn_combine(op.npu(), lk.npu(), out_dtype=dtype)
    print(f"shape={tuple(shape)} {dtype}")
    ulp_stats("npu vs golden", ref16, out.cpu(), exact16)
    lse_d = (out_lse.cpu().float() - ref_lse.float()).abs().max().item()
    print(f"     lse max abs diff={lse_d:.3e}")


def main():
    for seed in range(3):
        run([32, 16, 2, 256], torch.bfloat16, seed)
        run([8, 8, 1, 32, 128], torch.float16, seed)


if __name__ == "__main__":
    main()
