"""Native score/VJP pairs with independent upstream Torch forward oracles."""

from functools import lru_cache

from . import score_reference as ref
import torch
from .test_utils import aux_for

from . import examples
from flash_attn_npu_4.catlass_dsl.mods import Mod


def make_score_aux(name, **kwargs):
    return aux_for(name, **kwargs)


SCORE_NAMES = examples.SCORE_NAMES
AUX_SCORE_NAMES = ("batch_bias", "dual_buffer")
GLOBAL_SCORE_NAMES = (
    "scaled_squared",
    "global_kv_bias",
    "global_q_bias",
    "global_rel_plus_kv_bias",
    "global_q_and_kv_bias",
    "global_logical_rel_plus_kv_bias",
    "stress_complex_arithmetic",
    "stress_conditional_mask",
    "stress_multi_buffer",
    "stress_global_offset",
    "stress_xor_pattern",
    "debug_global_idx",
)
LOCAL_SCORE_NAMES = tuple(n for n in SCORE_NAMES if n not in GLOBAL_SCORE_NAMES)


def eager_reference(name, aux, cq, ck):
    if hasattr(ref, name + "_eager"):
        return getattr(ref, name + "_eager")
    if name in AUX_SCORE_NAMES:
        return getattr(ref, name + "_factory")(*aux)
    if name == "global_q_and_kv_bias":
        return ref.packed_q_and_kv_bias_factory(*aux, cq, ck)
    if name in ("stress_multi_buffer", "stress_conditional_mask", "stress_xor_pattern"):
        return getattr(ref, name + "_factory")(*aux, cq, ck)
    if name == "debug_global_idx":
        return ref.debug_global_idx_factory(None, ck)
    factory = {
        "global_kv_bias": "packed_kv_bias_factory",
        "global_q_bias": "packed_q_bias_factory",
        "global_rel_plus_kv_bias": "packed_rel_plus_kv_bias_factory",
        "global_logical_rel_plus_kv_bias": "packed_logical_rel_plus_kv_bias_factory",
    }.get(name, name + "_factory")
    return getattr(ref, factory)(
        *aux, cq if name in ("global_q_bias", "stress_complex_arithmetic") else ck
    )


@lru_cache(maxsize=128)
def get_score_pair(name):
    forward, backward = examples.score(name)

    def reference(s, b, h, q, k, info, aux, scalars):
        sq, sk, oq, ok = info
        # Factories index cumulative lengths by batch. Only this batch's
        # interval is evaluated; BNSD's global offset remains zero.
        cq = torch.tensor([oq] * (int(b) + 1) + [oq + sq], dtype=torch.int32)
        ck = torch.tensor([ok] * (int(b) + 1) + [ok + sk], dtype=torch.int32)
        return eager_reference(name, aux, cq, ck)(s, b, h, q, k)

    # Gradients of the independent oracle are computed by Torch autograd;
    # the device VJP is never used to manufacture expected gradients.
    return Mod(forward.simt, forward.simd, reference), backward
