"""Mask cases: native implementation paired with a Torch oracle.

Native callbacks come from tests/flash_attn/examples.py. The reference expressions
are a separate, attributed Torch-only snapshot of the upstream tests.
"""

from functools import lru_cache

from . import mask_reference as ref
import torch
from .test_utils import aux_for

from . import examples
from flash_attn_npu_4.catlass_dsl import builtin
from flash_attn_npu_4.catlass_dsl.mods import MaskMod

STATIC_MASK_NAMES = (
    "identity",
    "block_diagonal",
    "mini_causal",
    "prefix_lm",
    "dilated_sliding_window",
    "document",
    "ima",
    "packed_aux",
)
PARAMETERIZED_MASK_NAMES = ("causal", "block_causal", "sliding_window")
GLOBAL_MASK_NAMES = ("global_packed_doc", "global_ima", "global_causal_window")
MASK_NAMES = STATIC_MASK_NAMES + PARAMETERIZED_MASK_NAMES + GLOBAL_MASK_NAMES


def make_mask_aux(name, *, batch=2, heads=4, length=257):
    if name == "global_packed_doc":
        positions = torch.arange(batch * length, dtype=torch.int32)
        return positions // 5, positions // 5
    if name in ("ima", "global_ima"):
        return (torch.arange(batch * length, dtype=torch.int32) % 13,)
    if name == "global_causal_window":
        return (torch.arange(batch * length, dtype=torch.int32) % 7 + 1,)
    return aux_for(name, batch=batch, heads=heads, length=length)


@lru_cache(maxsize=128)
def get_mask_pair(name, *, window_size=128, offset=0):
    """Return a MaskMod with independent native/reference callables.

    causal/window use each sequence's bottom-right alignment; block_causal
    uses an explicit offset, as in the upstream parameterized factory.
    """
    if name not in MASK_NAMES:
        raise ValueError(f"Unknown mask: {name}")
    native = (
        builtin.all_visible
        if name == "identity"
        else examples.mask(name, left=window_size, right=window_size, offset=offset)
    )

    def reference(b, h, q, k, info, aux, scalars):
        sq, sk, oq, ok = info
        if name == "identity":
            return torch.ones_like(q + k, dtype=torch.bool)
        if name in (
            "block_diagonal",
            "mini_causal",
            "prefix_lm",
            "dilated_sliding_window",
        ):
            return getattr(ref, "flex_" + name + "_mask")(b, h, q, k)
        if name == "document":
            return ref.flex_document_mask(b, h, q, k, aux[0])
        if name == "ima":
            return ref.flex_ima_mask(b, h, q, k, aux[0])
        if name == "packed_aux":
            # Upstream packs bool[b,q,k] into little-endian UInt32 words.
            words = aux[0].to(torch.int64)[b, q, k // 32]
            return (words.bitwise_right_shift(k % 32).bitwise_and(1)) != 0
        if name in GLOBAL_MASK_NAMES:
            cq = torch.tensor([oq, oq + sq], dtype=torch.int32)
            ck = torch.tensor([ok, ok + sk], dtype=torch.int32)
            factory = getattr(ref, name + "_flex_factory")
            if name == "global_packed_doc":
                local = factory(*aux, cq, ck)
            else:
                local = factory(aux[0], ck if name == "global_ima" else cq)
            return local(0, sq, sk)(b, h, q, k)
        if name == "sliding_window":
            local = ref.get_flex_sliding_window_mask(window_size, window_size, sk - sq)
        else:
            local = ref.get_flex_causal_mask(sk - sq if name == "causal" else offset)
        return local(b, h, q, k)

    return MaskMod(
        native.simt,
        native.simd,
        reference,
        native.broadcast_batch,
        native.broadcast_head,
    )
