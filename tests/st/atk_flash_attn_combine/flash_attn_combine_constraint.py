# Constraint / scenario generator for flash_attn_combine.
#
# Design goals (L1/L2 case rules):
#   1. lse_partial.shape == out_partial.shape[:-1]; both fp32.
#   2. Deterministic scenario assignment that is independent of the ATK
#      generation order (ATK may generate with a process pool, so a shared
#      counter is not reliable). A stable CRC of ATK's own pre-override case
#      content selects the scenario.
#   3. 2:1 generalization : business-shape split.
#   4. Coverage of layout (BSND/TND), out_dtype (fp16/bf16 + a few fp32),
#      num_splits, head_dim, seqlen and the all-masked (-inf) short path.
#
# Value-distribution mix (10% narrow uniform / 30% wide uniform / 40% normal /
# 20% outlier) is declared in the YAML via random_types + custom_dist_ratio.
import zlib

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


HEAD_DIMS = [64, 96, 128, 192, 256]
NS_LIST = [1, 2, 3, 5, 8, 17, 32]
DTYPES = ["fp16", "bf16"]
LAYOUTS = [5, 4]
SEQ_LIST = [1, 16, 33, 113, 257, 512]
BATCH_LIST = [1, 2, 4]
HEADS_LIST = [1, 2, 4, 8]
MAX_ELEMS = 1 << 22  # keep each partial tensor <= 16 MiB in fp32

# Business-shaped profiles: (rank, num_splits, batch/total_q, seqlen, heads, dim)
BUSINESS_PROFILES = [
    (5, 8, 8, 1, 32, 128),       # decode, BSND
    (5, 16, 16, 1, 32, 128),     # decode, larger batch
    (5, 4, 32, 4, 64, 128),      # decode, many heads
    (4, 8, 1, 1, 32, 128),       # decode, TND
    (4, 16, 8, 4, 32, 128),      # decode, TND multi-token
    (5, 4, 1, 512, 32, 128),     # prefill, BSND
    (5, 2, 2, 2048, 32, 128),    # long prefill
    (4, 8, 1, 1024, 64, 128),    # long prefill, TND
]


def _prod(shape) -> int:
    total = 1
    for d in shape:
        total *= d
    return total


def _fit_budget(shape, token_axis):
    shape = list(shape)
    while _prod(shape) > MAX_ELEMS and shape[token_axis] > 1:
        shape[token_axis] = max(1, shape[token_axis] // 2)
    while _prod(shape) > MAX_ELEMS and shape[0] > 1:
        shape[0] = max(1, shape[0] // 2)
    return shape


def _choose_out_dtype(seed: int) -> str:
    """fp16/bf16 only: the requirement scopes output dtype to bf16/fp16."""
    return DTYPES[(seed // 2) % len(DTYPES)]


@GENERATOR_REGISTRY.register("generate_flash_attn_combine")
class FlashAttnCombineConstraint(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tensors = {inp.name: inp for inp in case_config.inputs if inp.type == "tensor"}
        attrs = {inp.name: inp for inp in case_config.inputs if inp.type == "attr"}
        op = tensors.get("out_partial")
        if op is None:
            return case_config
        shape = [int(d) for d in (op.shape or [])]
        is_boundary = bool(getattr(case_config, "is_boundary", False))
        out_attr = attrs.get("out_dtype")

        if is_boundary:
            # Keep ATK's boundary intent (empty / minimal) but make it runnable:
            # rank 4/5, num_splits >= 1, head_dim from the tested set.
            if len(shape) not in (4, 5):
                shape = [1, 1, 1, 1, HEAD_DIMS[0]]
            shape[0] = max(1, shape[0])
            shape[-1] = max(HEAD_DIMS[0], min(shape[-1], HEAD_DIMS[-1]))
            if len(shape) == 5:
                shape[3] = max(1, shape[3])
            else:
                shape[2] = max(1, shape[2])
            seed = 0
        else:
            orig_dtype = str(out_attr.range_values) if out_attr is not None else ""
            seed = zlib.crc32(
                f"{len(shape)}|{tuple(shape)}|{orig_dtype}".encode()
            )
            if seed % 3 == 0:
                # Business-shaped case (1/3 of the regular cases).
                rank, ns, batch, seqlen, heads, dim = BUSINESS_PROFILES[
                    (seed // 3) % len(BUSINESS_PROFILES)
                ]
                shape = (
                    [ns, batch, seqlen, heads, dim]
                    if rank == 5
                    else [ns, seqlen, heads, dim]
                )
            else:
                # Generalization case (2/3): orthogonal axis coverage.
                rank = LAYOUTS[seed % len(LAYOUTS)]
                dim = HEAD_DIMS[(seed // 4) % len(HEAD_DIMS)]
                ns = NS_LIST[(seed // 13) % len(NS_LIST)]
                seqlen = SEQ_LIST[(seed * 5) % len(SEQ_LIST)]
                batch = BATCH_LIST[(seed * 7) % len(BATCH_LIST)]
                heads = HEADS_LIST[(seed * 3) % len(HEADS_LIST)]
                shape = (
                    [ns, batch, seqlen, heads, dim]
                    if rank == 5
                    else [ns, seqlen, heads, dim]
                )
            if out_attr is not None:
                out_attr.range_values = _choose_out_dtype(seed)

        shape = _fit_budget(shape, token_axis=2 if len(shape) == 5 else 1)
        op.shape = shape
        op.dtype = "fp32"

        lse = tensors.get("lse_partial")
        if lse is not None:
            lse.shape = list(shape[:-1])
            lse.dtype = "fp32"
            # ~6% all-masked (-inf) short-path cases.
            if not is_boundary and seed % 17 == 0:
                lse.range_values = "-inf"

        return case_config
