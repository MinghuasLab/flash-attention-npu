"""Packed NPU attention using the FlashAttention v2 (910) varlen API."""

import torch
import torch.nn.functional as F
from flash_attn_npu.flash_attn_npu_interface import flash_attn_varlen_func

from uniattention.utils.attention_params import unsupported_capability


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
    """Return a one-element output tuple, preserving the original V head dim.

    Cumulative lengths must include the initial zero and final token count.
    Their values and maximum lengths are caller-provided metadata: no device
    tensor is copied to the host or read as a Python scalar in this wrapper.
    Causal attention is bottom-right aligned for unequal Q/K sequence lengths.
    Arbitrary tensor masks must continue to use another attention backend.
    """
    backend = "npu flash_attn v2"
    for name, enabled in (
        ("window_size", window_size is not None),
        ("softcap", softcap != 0.0),
        ("deterministic", deterministic),
        ("alibi_slopes", alibi_slopes is not None),
        ("return_lse", return_lse),
        ("q/k/v_descale", q_descale is not None or k_descale is not None or v_descale is not None),
    ):
        if enabled:
            raise unsupported_capability(backend, name, "not supported by this wrapper")
    if isinstance(attention_mask, torch.Tensor):
        raise unsupported_capability(backend, "attention_mask", "use the tensor-mask backend")
    if attention_mask not in (None, "full", "causal"):
        raise ValueError("attention_mask must be None, 'full', or 'causal'")
    if not 0.0 <= dropout_ratio < 1.0:
        raise ValueError("dropout_ratio must be in [0, 1)")

    cu_seqlens_q = cu_seqlens_q if cu_seqlens_q is not None else cu_seqlens
    cu_seqlens_k = cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q
    max_seqlen_q = max_seqlen_q if max_seqlen_q is not None else max_seqlen
    max_seqlen_k = max_seqlen_k if max_seqlen_k is not None else max_seqlen_q
    if cu_seqlens_q is None:
        raise ValueError("cu_seqlens_q is required for the NPU TND layout")
    if max_seqlen_q is None or max_seqlen_k is None:
        raise ValueError("max_seqlen_q and max_seqlen_k are required")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("maximum sequence lengths must be positive")
    for lengths in (cu_seqlens_q, cu_seqlens_k):
        if lengths.ndim != 1 or lengths.numel() < 2:
            raise ValueError("cu_seqlens must be 1D and include initial zero and final token count")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must have int32 or int64 dtype")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q and K cumulative lengths must describe the same batch size")

    if any(t.ndim != 3 for t in (query_states, key_states, value_states)):
        raise ValueError("Q/K/V must use the three-dimensional TND layout")
    if key_states.shape[:2] != value_states.shape[:2]:
        raise ValueError("K/V token counts and head counts must match")
    if key_states.shape[1] == 0 or query_states.shape[1] == 0:
        raise ValueError("Q/K head counts must be positive")
    if query_states.shape[1] % key_states.shape[1] != 0:
        raise ValueError("KV head count must divide Q head count")
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashAttention v2 requires float16 or bfloat16 Q/K/V")
    if any(t.dtype != query_states.dtype or t.device != query_states.device for t in (key_states, value_states)):
        raise ValueError("Q/K/V must have the same dtype and device")
    if (query_pe_states is None) != (key_pe_states is None):
        raise ValueError("query_pe_states and key_pe_states must be supplied together")

    if query_pe_states is not None:
        query_states = torch.cat((query_states, query_pe_states), dim=-1)
        key_states = torch.cat((key_states, key_pe_states), dim=-1)
    if query_states.shape[-1] != key_states.shape[-1]:
        raise ValueError("query dim is not equal to key dim")
    qk_head_dim = query_states.shape[-1]
    vo_head_dim = value_states.shape[-1]
    if min(qk_head_dim, vo_head_dim) < 1 or max(qk_head_dim, vo_head_dim) > 256:
        raise ValueError("FlashAttention v2 supports head dimensions in [1, 256]")
    if qk_head_dim > vo_head_dim:
        value_states = F.pad(value_states, (0, qk_head_dim - vo_head_dim))
    elif qk_head_dim < vo_head_dim:
        query_states = F.pad(query_states, (0, vo_head_dim - qk_head_dim))
        key_states = F.pad(key_states, (0, vo_head_dim - qk_head_dim))

    # Keep complete cumulative-length tensors (including zero) on the Q device.
    #cu_seqlens_q = cu_seqlens_q.to(device=query_states.device, dtype=torch.int32)
    #cu_seqlens_k = cu_seqlens_k.to(device=query_states.device, dtype=torch.int32)
    output = flash_attn_varlen_func(
        q=query_states,
        k=key_states,
        v=value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_ratio,
        softmax_scale=softmax_scaling,
        causal=attention_mask == "causal",
    )
    # The varlen API returns a Tensor, not the legacy fusion-attention tuple.
    if output.shape[-1] != vo_head_dim:
        output = output[..., :vo_head_dim]
    return (output,)
