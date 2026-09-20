"""Ascend 950：自定义右对齐 causal mask、分数缩放及可选块稀疏。"""

import argparse

import catlass.tla as tla

from . import compute_block_sparsity, flash_attn_func, simd


def causal_mask(b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return kv_idx <= q_idx + seqlen_info.seqlen_k - seqlen_info.seqlen_q


@simd
def vector_causal_mask(b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return tla.cmp(kv_idx, q_idx + seqlen_info.seqlen_k - seqlen_info.seqlen_q, "le")


def scale_score(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return score * aux_scalars[0]


@simd
def vector_scale_score(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars):
    return score * aux_scalars[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--head-dim", type=int, choices=(64, 96, 128), default=128)
    parser.add_argument("--q-len", type=int, default=129)
    parser.add_argument("--kv-len", type=int, default=257)
    parser.add_argument(
        "--simd", action="store_true", help="mask 和 score 均使用显式 64-lane SIMD 回调"
    )
    parser.add_argument("--simd-mask", action="store_true", help="mask 使用显式 64-lane SIMD 回调")
    parser.add_argument(
        "--simd-score", action="store_true", help="score 使用显式 64-lane SIMD 回调"
    )
    parser.add_argument("--sparse", action="store_true", help="先分类，再执行块稀疏 attention")
    parser.add_argument("--no-lse", action="store_true", help="仅返回 O，LSE 为 None")
    args = parser.parse_args()
    if args.q_len <= 0 or args.kv_len <= 0:
        parser.error("q-len 和 kv-len 必须为正数")

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    sq, sk, dim = args.q_len, args.kv_len, args.head_dim
    q = torch.randn(1, sq, 2, dim, device=f"npu:{args.device}", dtype=dtype)
    k = torch.randn(1, sk, 1, dim, device=q.device, dtype=dtype)
    v = torch.randn_like(k)
    mask_mod = vector_causal_mask if args.simd or args.simd_mask else causal_mask
    score_mod = vector_scale_score if args.simd or args.simd_score else scale_score
    aux_scalars = (0.75,)
    blocks = None
    if args.sparse:
        blocks = compute_block_sparsity(
            128, 128, 1, 2, sq, sk, mask_mod, None, q.device, aux_scalars=aux_scalars
        )
    out, lse = flash_attn_func(
        q,
        k,
        v,
        mask_mod=mask_mod,
        score_mod=score_mod,
        aux_scalars=aux_scalars,
        block_sparse_tensors=blocks,
        return_lse=not args.no_lse,
    )
    torch.npu.synchronize()

    # 小规模 CPU FP32 对照：GQA 展开、右对齐 mask、缩放、自然对数 LSE。
    q_ref, k_ref, v_ref = (x.detach().float().cpu() for x in (q, k, v))
    k_ref, v_ref = (x.repeat_interleave(2, dim=2) for x in (k_ref, v_ref))
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref, k_ref) / dim**0.5 * aux_scalars[0]
    keep = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
    scores = scores.masked_fill(~keep, -torch.inf)
    expected_lse = torch.logsumexp(scores, dim=-1)
    empty = torch.isneginf(expected_lse)
    safe_lse = torch.where(empty, 0, expected_lse)
    probabilities = torch.exp(scores - safe_lse[..., None])
    expected_out = torch.einsum("bhqk,bkhd->bqhd", probabilities, v_ref)
    actual_out = out.detach().float().cpu()
    torch.testing.assert_close(actual_out, expected_out, atol=0.05, rtol=0)
    assert bool((actual_out.permute(0, 2, 1, 3)[empty] == 0).all())
    assert (lse is None) == args.no_lse
    if lse is not None:
        actual_lse = lse.detach().cpu()
        assert torch.equal(torch.isneginf(actual_lse), empty)
        torch.testing.assert_close(actual_lse[~empty], expected_lse[~empty], atol=0.05, rtol=0)
    print(f"通过：{args.dtype} D={dim}，O{' only' if args.no_lse else '+LSE'}，绝对误差 ≤ 0.05")


if __name__ == "__main__":
    main()
