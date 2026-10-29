import torch
import torch.nn.functional as F
import torch_npu

from uniattention.utils.attention_params import unsupported_capability


# Save and cache the masks on each device
ATTN_MASK_NPU_CACHE = {}


def get_attn_mask_npu(device):
    """Get or create attention mask for the specified device."""
    if device not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[device] = torch.triu(torch.ones([2048, 2048], device=device), diagonal=1).bool()
    return ATTN_MASK_NPU_CACHE[device]


def get_attn_mask_npu_compile(device):
    return torch.triu(torch.ones([2048, 2048], device=device), diagonal=1).bool()


def flash_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    softmax_scaling: float = 1.0,
    dropout_ratio: float = 0.0,
    window_size: tuple | None = None,
    softcap: float = 0.0,
    deterministic: bool = False,
    alibi_slopes: torch.Tensor | None = None,
    return_lse: bool = False,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    attention_mask: str | torch.Tensor = None,
    query_pe_states: torch.Tensor = None,
    key_pe_states: torch.Tensor = None,
    **kwargs,
):
    cu_seqlens_q = cu_seqlens_q if cu_seqlens_q is not None else cu_seqlens
    max_seqlen_q = max_seqlen_q if max_seqlen_q is not None else max_seqlen
    cu_seqlens_k = cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q
    max_seqlen_k = max_seqlen_k if max_seqlen_k is not None else max_seqlen_q

    if cu_seqlens_q is None:
        raise ValueError("cu_seqlens_q is required for the NPU TND layout")

    # Capability checks — NPU npu_fusion_attention supports only dropout (via keep_prob).
    if window_size is not None:
        raise unsupported_capability("npu npu_fusion_attention", "window_size", "no NPU kernel support yet")
    if softcap != 0.0:
        raise unsupported_capability("npu npu_fusion_attention", "softcap", "no NPU kernel support yet")
    if deterministic:
        raise unsupported_capability("npu npu_fusion_attention", "deterministic", "no NPU kernel support yet")
    if alibi_slopes is not None:
        raise unsupported_capability(
            "npu npu_fusion_attention", "alibi_slopes", "no NPU kernel support yet (pse exists; future work)"
        )
    if return_lse:
        raise unsupported_capability("npu npu_fusion_attention", "return_lse", "no NPU kernel support yet")
    if q_descale is not None or k_descale is not None or v_descale is not None:
        raise unsupported_capability("npu npu_fusion_attention", "q/k/v_descale", "no NPU kernel support yet")

    keep_prob = 1.0 - dropout_ratio

    # MLA Concat
    if query_pe_states is not None and key_pe_states is not None:
        query_states = torch.cat((query_states, query_pe_states), dim=-1)
        key_states = torch.cat((key_states, key_pe_states), dim=-1)

    if query_states.shape[-1] != key_states.shape[-1]:
        raise ValueError("query dim is not equal to key dim.")

    # MLA Padding
    qk_head_dim = query_states.shape[-1]
    vo_head_dim = value_states.shape[-1]
    if qk_head_dim > vo_head_dim:
        value_states = F.pad(value_states, [0, qk_head_dim - vo_head_dim])
    elif qk_head_dim < vo_head_dim:
        query_states = F.pad(query_states, [0, vo_head_dim - qk_head_dim])
        key_states = F.pad(key_states, [0, vo_head_dim - qk_head_dim])

    if attention_mask == "causal":
        # If the sparse_mode is 3, when generate atten_mask_npu, no matter what shape is, it will always be 2048
        if torch.compiler.is_compiling():
            atten_mask_npu = get_attn_mask_npu_compile(query_states.device)
            from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_stream import inductor_npu_fusion_attention

            output = inductor_npu_fusion_attention(
                query_states,
                key_states,
                value_states,
                query_states.shape[1],
                atten_mask=atten_mask_npu,
                scale=softmax_scaling,
                keep_prob=keep_prob,
                input_layout="TND",
                actual_seq_qlen=cu_seqlens_q[1:],
                actual_seq_kvlen=cu_seqlens_k[1:],
                sparse_mode=3,
            )
        else:
            atten_mask_npu = get_attn_mask_npu(query_states.device)
            output = torch_npu.npu_fusion_attention(
                query_states,
                key_states,
                value_states,
                query_states.shape[1],
                atten_mask=atten_mask_npu,
                scale=softmax_scaling,
                keep_prob=keep_prob,
                input_layout="TND",
                actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                sparse_mode=3,
            )
    else:
        # Full mask
        if torch.compiler.is_compiling():
            from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_stream import inductor_npu_fusion_attention

            output = inductor_npu_fusion_attention(
                query_states,
                key_states,
                value_states,
                query_states.shape[1],
                atten_mask=None,
                scale=softmax_scaling,
                keep_prob=keep_prob,
                input_layout="TND",
                actual_seq_qlen=cu_seqlens_q[1:],
                actual_seq_kvlen=cu_seqlens_k[1:],
            )
        else:
            output = torch_npu.npu_fusion_attention(
                query_states,
                key_states,
                value_states,
                query_states.shape[1],
                atten_mask=None,
                scale=softmax_scaling,
                keep_prob=keep_prob,
                input_layout="TND",
                actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            )

    output = (output[0][..., :vo_head_dim], *output[1:])

    return output
