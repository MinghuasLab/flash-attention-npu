"""CPU-only study: how far can k-way fp32 split accumulation get us?

For each seeded random case we compute the fp16 output of
  * the golden reference (torch fp32 sum over splits, cast to fp16)
  * float64 exact sum (best achievable pre-cast value)
  * k-way fp32 accumulation for k = 1, 2, 4, 8
and report the RMSE of each against the golden, in units of the golden RMSE.
This isolates the split-accumulation contribution from exp/log/exactness.
"""

import torch


def reference(op, lse):
    m = lse.max(0, keepdim=True).values
    lse_out = m + (lse - m).exp().sum(0, keepdim=True).log()
    w = (lse - lse_out).exp()
    return (w.unsqueeze(-1) * op).sum(0), lse_out.squeeze(0), w


def rmse(a, b):
    d = (a.float() - b.float()).reshape(-1)
    return d.pow(2).mean().sqrt().item()


def kway(op, w, k):
    accs = [torch.zeros_like(op[0]) for _ in range(k)]
    for s in range(op.shape[0]):
        accs[s % k] = accs[s % k] + w[s].unsqueeze(-1) * op[s]
    # pairwise combine of the k partial sums
    while len(accs) > 1:
        nxt = []
        for i in range(0, len(accs) - 1, 2):
            nxt.append(accs[i] + accs[i + 1])
        if len(accs) % 2:
            nxt.append(accs[-1])
        accs = nxt
    return accs[0]


def main():
    splits, rows, heads, dim, seeds = 16, 4, 32, 128, 20
    ks = [1, 2, 4, 8]
    tot = {k: 0.0 for k in ks}
    tot["exact"] = 0.0
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        mean = float(torch.empty(1).uniform_(-100.0, 100.0, generator=g))
        std = float(torch.empty(1).uniform_(1.0, 25.0, generator=g))
        op = torch.randn(splits, rows, heads, dim, generator=g) * std + mean
        lse = torch.rand(splits, rows, heads, generator=g) * 10.0 - 5.0

        golden32, _, w = reference(op, lse)
        golden16 = golden32.to(torch.float16)
        base = rmse(golden16, golden16)
        # error floor: RMSE of the golden vs a fp32 reference of the same value
        ref_rmse = rmse(golden32, golden32)
        exact16 = (w.double().unsqueeze(-1) * op.double()).sum(0).float().to(torch.float16)
        exact_err = rmse(exact16, golden16)
        for k in ks:
            tot[k] += rmse(kway(op, w, k).to(torch.float16), golden16)
        tot["exact"] += exact_err
    print(f"mean RMSE vs golden16 over {seeds} seeds "
          f"(golden self-RMSE={base:.1e}, fp32={ref_rmse:.1e})")
    for k in ks:
        print(f"  k={k:<2d} {tot[k] / seeds:.4e}")
    print(f"  float64 exact {tot['exact'] / seeds:.4e}")


if __name__ == "__main__":
    main()
