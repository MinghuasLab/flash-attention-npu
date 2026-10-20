"""Opaque inference calls for torch.compile; device compilation remains in TLA."""

import torch

from .block_sparsity import BlockSparseTensorsTorch

# Graphs may outlive the bounded executable cache. Keep immutable Python call
# specifications alive for their dispatcher handles; never retain input tensors.
_specifications = []


@torch._dynamo.assume_constant_result
def _bind_specification(kind, options, has_auxiliary, block_size, has_blocks):
    spec = (kind, options, has_auxiliary, block_size, has_blocks)
    for index, existing in enumerate(_specifications):
        if existing == spec:
            return index
    _specifications.append(spec)
    return len(_specifications) - 1


@torch.library.custom_op(
    "flash_attn_npu_dsl::forward",
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor?[] metadata, Tensor[] auxiliary, "
    "Scalar[] scalars, int specification) -> (Tensor, Tensor?)",
)
def _forward(q, k, v, metadata, auxiliary, scalars, specification):
    from .interface import flash_attn_func, flash_attn_varlen_func

    kind, options, has_auxiliary, block_size, has_blocks = _specifications[specification]
    (
        scale,
        causal,
        window,
        softcap,
        splits,
        pack_gqa,
        deterministic,
        score,
        mask,
        return_lse,
        max_q,
        max_k,
    ) = options
    kwargs = dict(
        softmax_scale=scale,
        causal=causal,
        window_size=window,
        softcap=softcap,
        num_splits=splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        score_mod=score,
        mask_mod=mask,
        return_lse=return_lse,
        aux_tensors=list(auxiliary) if has_auxiliary else None,
        aux_scalars=tuple(scalars) if scalars else None,
    )
    if has_blocks:
        kwargs["block_sparse_tensors"] = BlockSparseTensorsTorch(*metadata[5:], block_size)
    if kind == "varlen":
        cuq, cuk, used_q, used_k, pages = metadata[:5]
        kwargs.update(
            cu_seqlens_q=cuq,
            cu_seqlens_k=cuk,
            seqused_q=used_q,
            seqused_k=used_k,
            page_table=pages,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
        )
        return flash_attn_varlen_func(q, k, v, **kwargs)
    return flash_attn_func(q, k, v, **kwargs)


@_forward.register_fake
def _fake(q, k, v, metadata, auxiliary, scalars, specification):
    options = _specifications[specification][1]
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))
    lse = None
    if options[9]:
        shape = (q.shape[-2], q.shape[0]) if q.ndim == 3 else (q.shape[0], q.shape[2], q.shape[1])
        lse = torch.empty(shape, dtype=torch.float32, device=q.device)
    return out, lse


def call(kind, q, k, v, options, sequence_tensors, aux_tensors, aux_scalars, blocks):
    """Keep callback/static options outside the graph and tensor values inside it."""
    block_size = None
    block_tensors = (None,) * 6
    if blocks is not None:
        if not isinstance(blocks, BlockSparseTensorsTorch):
            raise TypeError("block_sparse_tensors must be BlockSparseTensorsTorch")
        if any(
            x is not None for x in (blocks.dq_write_order, blocks.dq_write_order_full, blocks.spt)
        ):
            raise NotImplementedError("backward sparse metadata is not supported")
        block_size = tuple(blocks.block_size) if blocks.block_size is not None else None
        block_tensors = (
            blocks.mask_block_cnt,
            blocks.mask_block_idx,
            blocks.full_block_cnt,
            blocks.full_block_idx,
            blocks.cu_total_m_blocks,
            blocks.cu_block_idx_offsets,
        )
    if aux_scalars is not None and any(type(x) not in (bool, int, float) for x in aux_scalars):
        raise NotImplementedError("torch.compile requires Python bool/int/float auxiliary scalars")
    specification = _bind_specification(
        kind, options, aux_tensors is not None, block_size, blocks is not None
    )
    return _forward(
        q,
        k,
        v,
        [*sequence_tensors, *block_tensors],
        list(aux_tensors) if aux_tensors is not None else [],
        list(aux_scalars) if aux_scalars else [],
        specification,
    )
