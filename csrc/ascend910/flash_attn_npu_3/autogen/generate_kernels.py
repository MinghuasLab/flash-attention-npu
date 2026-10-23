# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by Minghua Shen, 2026.

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Mirrors flash-attention/csrc/flash_attn/src/generate_kernels.py, adapted to the
# v3 NPU dispatch layout. The GPU script's generation axis is
# <dtype, head_dim, is_causal> — one explicit instantiation per .cu so nvcc
# compiles them in parallel. On the NPU side head_dim is NOT a template axis:
# the forward (FAInfer) does not bin by head_dim at all (it is a runtime tiling
# value), and the backward (FAGGeneral) selects head_dim at runtime via a
# switch() inside the impl (bwd_dispatch_common.hpp). So head_dim is deliberately
# dropped here.
#
# The real compile-time axes are dtype and input layout. The forward supports
# the three input layouts (TND / BSND / BNSD); the backward only two (TND /
# BSND — BNSD never enters the backward). Each generated .cpp holds exactly ONE
# explicit instantiation of the per-family dispatch template, so the heavy
# kernel templates land in separate, parallel-compiled translation units. The
# impl bodies live in the shared fwd_dispatch_impl.hpp / bwd_dispatch_common.hpp
# headers (one directory up); this script only emits the instantiation stubs.

# dtype key -> C++ type token
DTYPE_MAP = {
    "fp16": "half",
    "bf16": "bfloat16_t",
}

# Forward layout key -> (display name, FaiKenel::inputLayout token).
FWD_LAYOUTS = [
    ("bsnd", "BSND", "FaiKenel::inputLayout::BSND"),
    ("tnd", "TND", "FaiKenel::inputLayout::TND"),
    ("bnsd", "BNSD", "FaiKenel::inputLayout::BNSD"),
]

# Backward layout key -> (display name, FAG layout constant token). The backward
# uses the FAG-side BSND/TND constants (see fag_kernel_common.hpp), not the
# kernel inputLayout enum.
BWD_LAYOUTS = [
    ("bsnd", "BSND", "BSND"),
    ("tnd", "TND", "TND"),
]

PRELUDE = (
    "/**\n"
    " * Copyright (c) 2026 Huawei Technologies Co., Ltd.\n"
    " * Modified by Minghua Shen, 2026.\n"
    " */\n"
)


def _header(family_desc: str, layout_display: str) -> str:
    return (
        PRELUDE + f"// v3 {family_desc}, {layout_display} variant. One explicit instantiation per\n"
        "// translation unit so the kernel templates compile in parallel across\n"
        "// cores; head_dim is a runtime axis (switch/tiling inside the impl), not a\n"
        "// template parameter, so it is not a generation axis.\n\n"
    )


def fwd_kernel(dtype_key: str, layout: tuple) -> "Kernel":
    ctype = DTYPE_MAP[dtype_key]
    layout_key, display, layout_token = layout
    body = (
        _header("forward FAInfer dispatch", display)
        + '#include "../fwd_dispatch_impl.hpp"\n\n'
        + f"template void launch_fwd_dtype<{ctype}, {layout_token}>(const FwdLaunchArgs &);"
        f"  // {display}\n"
    )
    return Kernel(
        family="fwd",
        dtype=dtype_key,
        layout=layout_key,
        filename=f"fwd_dispatch_{dtype_key}_{layout_key}.cpp",
        content=body,
    )


def bwd_kernel(dtype_key: str, layout: tuple) -> "Kernel":
    ctype = DTYPE_MAP[dtype_key]
    layout_key, display, enum = layout
    body = (
        _header("backward FAGGeneral dispatch", display)
        + '#include "../bwd_dispatch_common.hpp"\n\n'
        + f"template void bwd_dispatch_run<{ctype}, {enum}>(const BwdLaunchArgs &);"
        f"  // {display}\n"
    )
    return Kernel(
        family="bwd",
        dtype=dtype_key,
        layout=layout_key,
        filename=f"bwd_dispatch_{dtype_key}_{layout_key}.cpp",
        content=body,
    )


@dataclass
class Kernel:
    family: str
    dtype: str
    layout: str
    filename: str
    content: str


# MLA groups layout instantiations by implementation family and dtype. This
# keeps parallel compilation without a separate macro-configured file per layout.
MLA_FAMILIES = {
    "auto_split": ("mla_auto_split_dispatch_impl.hpp", [("LaunchCompressed", "MlaLaunchArgs", "")]),
    "prefill": ("mla_prefill_dispatch_impl.hpp", [("LaunchPrefill", "MlaLaunchArgs", "")]),
    "gqa": ("mla_gqa_dispatch_impl.hpp", [("LaunchGrouped", "MlaLaunchArgs", "")]),
}


def mla_kernel(dtype: str, family: str, header: str, entries: list) -> Kernel:
    ctype = DTYPE_MAP[dtype]
    body = PRELUDE + (
        "// Generated MLA template instantiations.\n"
        f'#include "../mla/{header}"\n\n'
        "namespace flash_attn::mla {\n"
    )
    for launch, arguments, options in entries:
        body += f"template aclError {launch}<{ctype}{options}>(const {arguments}&);\n"
    body += "} // namespace flash_attn::mla\n"
    return Kernel("mla", dtype, family, f"mla_dispatch_{dtype}_{family}.cpp", body)


def get_all_kernels() -> List[Kernel]:
    kernels: List[Kernel] = []
    for dtype in DTYPE_MAP:
        for layout in FWD_LAYOUTS:
            kernels.append(fwd_kernel(dtype, layout))
        for layout in BWD_LAYOUTS:
            kernels.append(bwd_kernel(dtype, layout))
        for family, (header, entries) in MLA_FAMILIES.items():
            kernels.append(mla_kernel(dtype, family, header, entries))
    return kernels


def write_kernel(kernel: Kernel, output_dir: Path) -> None:
    (output_dir / kernel.filename).write_text(kernel.content)


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)
    print(f"Generated {len(get_all_kernels())} dispatch TUs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the v3 flash_attn_npu dispatch template instantiations "
        "(per dtype x layout; head_dim is a runtime axis and is not generated).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the dispatch .cpp stubs; "
        "defaults to this script's directory (csrc/flash_attn_npu_3/autogen).",
    )
    args = parser.parse_args()
    main(args.output_dir)
