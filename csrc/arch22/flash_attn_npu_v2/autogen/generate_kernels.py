# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by Minghua Shen, 2026.

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Mirrors flash-attention/csrc/flash_attn/src/generate_kernels.py, but adapted to
# the v2 NPU dispatch layout. The GPU script's generation axis is
# <dtype, head_dim, is_causal> — one explicit instantiation per .cu so nvcc
# compiles them in parallel. On the NPU side head_dim is NOT a template axis:
# the forward (FAInfer) does not bin by head_dim at all, and the backward
# (FAGGeneral) selects head_dim at runtime via a switch() inside the impl
# (fag_general_dispatch_impl.hpp). So head_dim is deliberately dropped here.
#
# The real compile-time axes are dtype and input layout (TND / BSND). Each
# generated .cpp holds exactly ONE explicit instantiation of the per-family
# launch wrapper, so the heavy kernel templates land in separate, parallel-
# compiled translation units. The impl bodies live in the shared *_impl.hpp
# headers (one directory up); this script only emits the instantiation stubs.

# dtype key -> (C++ type token, filename suffix)
DTYPE_MAP = {
    "fp16": ("half", "fp16"),
    "bf16": ("bfloat16_t", "bf16"),
}

# layout key -> (display name, fwd IS_TND bool token, fag layout enum token)
LAYOUTS = [
    ("bsnd", "BSND", "false", "BSND"),
    ("tnd",  "TND",  "true",  "TND"),
]

PRELUDE = (
    "/**\n"
    " * Copyright (c) 2026 Huawei Technologies Co., Ltd.\n"
    " * Modified by Minghua Shen, 2026.\n"
    " */\n"
)


def _header(family_desc: str, dtype_key: str, layout_display: Optional[str]) -> str:
    layout_note = f", {layout_display} variant" if layout_display else " variant (always TND)"
    return (
        PRELUDE
        + f"// v2 {family_desc}{layout_note}. One explicit instantiation per\n"
        "// translation unit so the kernel templates compile in parallel across\n"
        "// cores; head_dim is a runtime axis (switch inside the impl), not a\n"
        "// template parameter, so it is not a generation axis.\n\n"
    )


def fwd_kernel(dtype_key: str, layout: tuple) -> "Kernel":
    ctype, _ = DTYPE_MAP[dtype_key]
    layout_key, display, is_tnd, _ = layout
    body = (
        _header("forward FAInfer dispatch", dtype_key, display)
        + '#include "../fwd_dispatch_impl.hpp"\n\n'
        + f"template void launch_fwd_impl<{ctype}, {is_tnd}>(const FwdLaunchArgs &);"
        f"  // {display}\n"
    )
    return Kernel(
        family="fwd",
        dtype=dtype_key,
        layout=layout_key,
        filename=f"fwd_dispatch_{dtype_key}_{layout_key}.cpp",
        content=body,
    )


def fag_kernel(dtype_key: str, layout: tuple) -> "Kernel":
    _, suffix = DTYPE_MAP[dtype_key]
    layout_key, display, _, enum = layout
    body = (
        _header("FAGGeneral backward dispatch", dtype_key, display)
        + '#include "../fag_general_dispatch_impl.hpp"\n\n'
        + f"template void launch_fag_general_dispatch_{suffix}<{enum}>"
        f"(const FagGeneralLaunchArgs &);  // {display}\n"
    )
    return Kernel(
        family="fag_general_bwd",
        dtype=dtype_key,
        layout=layout_key,
        filename=f"fag_general_dispatch_{dtype_key}_{layout_key}.cpp",
        content=body,
    )


def varlen_bwd_kernel(dtype_key: str) -> "Kernel":
    ctype, _ = DTYPE_MAP[dtype_key]
    body = (
        _header("varlen backward FAGVarlenOpt dispatch", dtype_key, None)
        + '#include "../varlen_bwd_dispatch_impl.hpp"\n\n'
        + f"template void launch_varlen_bwd_impl<{ctype}>(const VarlenBwdLaunchArgs &);\n"
    )
    return Kernel(
        family="varlen_bwd",
        dtype=dtype_key,
        layout=None,
        filename=f"varlen_bwd_dispatch_{dtype_key}.cpp",
        content=body,
    )


@dataclass
class Kernel:
    family: str
    dtype: str
    layout: Optional[str]
    filename: str
    content: str


def get_all_kernels() -> List[Kernel]:
    kernels: List[Kernel] = []
    for dtype in DTYPE_MAP:
        for layout in LAYOUTS:
            kernels.append(fwd_kernel(dtype, layout))
            kernels.append(fag_kernel(dtype, layout))
        kernels.append(varlen_bwd_kernel(dtype))
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
        description="Generate the v2 flash_attn_npu dispatch template instantiations "
        "(per dtype x layout; head_dim is a runtime axis and is not generated).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the dispatch .cpp stubs; "
        "defaults to this script's directory (csrc/flash_attn_npu/autogen).",
    )
    args = parser.parse_args()
    main(args.output_dir)
