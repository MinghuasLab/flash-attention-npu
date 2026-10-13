# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by Minghua Shen, 2026.

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Splits the MXFP4 dispatch into one translation unit per
# (sparsity, layout, quant mode, output dtype) so the kernel templates compile
# in parallel. Each generated .cpp holds exactly ONE explicit instantiation of
# launch_mxfp4_impl<OType, Layout, QuantMode, Sparse>. The kernel lives in
# mxfp4_kernel.hpp and is launched from mxfp4_kernel_dispatch_impl.hpp; the
# runtime if-tree lives in mxfp4_kernel_dispatch.hpp (one directory up).

# dtype key -> C++ type token
DTYPE_MAP = {
    "fp16": "half",
    "bf16": "bfloat16_t",
}

# layout key -> (display name, Mxfp4Layout enumerator)
LAYOUTS = [
    ("tnd", "TND", "TND"),
    ("bnsd", "BNSD", "BNSD"),
    ("bsnd", "BSND", "BSND"),
]

# quant key -> (display name, Mxfp4QuantMode enumerator)
QUANTS = [
    ("ocp", "OCP", "OCP"),
    ("cx", "CX", "CX"),
]

# policy key -> Sparse bool token
POLICIES = [
    ("dense", "false"),
    ("sparse", "true"),
]

PRELUDE = (
    "/**\n"
    " * Copyright (c) 2026 Huawei Technologies Co., Ltd.\n"
    " * Modified by Minghua Shen, 2026.\n"
    " */\n"
)


@dataclass
class Kernel:
    filename: str
    content: str


def mxfp4_kernel(policy: tuple, layout: tuple, quant: tuple, dtype_key: str) -> Kernel:
    policy_key, sparse = policy
    layout_key, layout_display, layout_enum = layout
    quant_key, quant_display, quant_enum = quant
    ctype = DTYPE_MAP[dtype_key]
    content = (
        PRELUDE
        + f"// MXFP4 dispatch, {policy_key} x {dtype_key} x {layout_display} x {quant_display} variant.\n"
        "// One explicit instantiation per translation unit so the MXFP4 kernel\n"
        "// templates compile in parallel across cores.\n\n"
        + '#include "../mxfp4_kernel_dispatch_impl.hpp"\n\n'
        + "template void launch_mxfp4_impl<"
        + f"{ctype}, Mxfp4Layout::{layout_enum}, Mxfp4QuantMode::{quant_enum}, {sparse}>(\n"
        "    const Mxfp4LaunchArgs &a);\n"
    )
    return Kernel(
        filename=f"mxfp4_dispatch_{policy_key}_{layout_key}_{quant_key}_{dtype_key}.cpp",
        content=content,
    )


def get_all_kernels() -> List[Kernel]:
    return [
        mxfp4_kernel(policy, layout, quant, dtype)
        for policy in POLICIES
        for layout in LAYOUTS
        for quant in QUANTS
        for dtype in DTYPE_MAP
    ]


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = get_all_kernels()
    expected = set()
    for kernel in kernels:
        (output_dir / kernel.filename).write_text(kernel.content)
        expected.add(kernel.filename)
    for path in output_dir.glob("mxfp4_*.cpp"):
        if path.name not in expected:
            path.unlink()
    print(f"Generated {len(kernels)} dispatch TUs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the MXFP4 dispatch template instantiations "
        "(per sparsity x layout x quant mode x output dtype).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the dispatch .cpp stubs; defaults to this script's directory.",
    )
    args = parser.parse_args()
    main(args.output_dir)
