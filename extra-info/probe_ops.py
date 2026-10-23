"""Measure the accuracy of the device side exp / div on fp32.

The combine kernel is limited by element-wise op accuracy once the split
accumulation is exact (Kahan), so measure what the NPU does for the same
element-wise operations torch performs on the CPU.
"""

import torch
import torch_npu  # noqa: F401


def rel_err(a, b):
    d = (a - b).abs() / b.abs().clamp_min(1e-30)
    return float(d.max()), float(d.median())


def main():
    g = torch.Generator().manual_seed(0)
    # exp over the range the combine kernel sees: lse_s - max in [-40, 0]
    x = -torch.rand(1 << 20, generator=g) * 40.0
    e_cpu = torch.exp(x)
    e_npu = torch.exp(x.npu()).cpu()
    mx, med = rel_err(e_npu, e_cpu)
    print(f"exp  max rel {mx:.3e}  median rel {med:.3e}")

    a = torch.randn(1 << 20, generator=g) * 100.0
    b = torch.rand(1 << 20, generator=g) * 16.0 + 1.0
    q_cpu = a / b
    q_npu = (a.npu() / b.npu()).cpu()
    mx, med = rel_err(q_npu, q_cpu)
    print(f"div  max rel {mx:.3e}  median rel {med:.3e}")

    # scalar-vector divide (what a Div with a broadcast den costs)
    d_cpu = a * (1.0 / 3.0)
    d_npu = (a.npu() * (1.0 / 3.0)).cpu()
    mx, med = rel_err(d_npu, d_cpu)
    print(f"mul  max rel {mx:.3e}  median rel {med:.3e}")


if __name__ == "__main__":
    main()
