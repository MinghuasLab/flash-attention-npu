# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by Minghua Shen, 2026.

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Splits the v3 (950) forward FAInfer dispatch into four per-(dtype, layout)
# translation units so the FAInfer / FAInferDn kernel templates compile in
# parallel. head_dim is a runtime tiling axis (not a template parameter), so it
# is not a generation axis. Each generated .cpp holds exactly ONE explicit
# instantiation of launch_fai_dispatch<DType, IS_TND>; the generic dispatch body
# (which selects the per-(dtype, layout) case set with if constexpr) lives in
# the shared fai_host_api_impl.hpp header (one directory up).

# dtype key -> C++ type token
DTYPE_MAP = {
    "fp16": "half",
    "bf16": "bfloat16_t",
}

# layout key -> (display name, IS_TND bool token)
LAYOUTS = [
    ("bsnd", "BSND", "false"),
    ("tnd",  "TND",  "true"),
]

PRELUDE = (
    "/**\n"
    " * Copyright (c) 2026 Huawei Technologies Co., Ltd.\n"
    " * Modified by Minghua Shen, 2026.\n"
    " */\n"
)


@dataclass
class Kernel:
    dtype: str
    layout: str
    filename: str
    content: str


def fai_kernel(dtype_key: str, layout: tuple) -> Kernel:
    ctype = DTYPE_MAP[dtype_key]
    layout_key, display, is_tnd = layout
    content = (
        PRELUDE
        + f"// v3 (950) forward FAInfer dispatch, {dtype_key} x {display} variant.\n"
        "// One explicit instantiation per translation unit so the FAInfer / FAInferDn\n"
        "// kernel templates compile in parallel across cores; head_dim is a runtime\n"
        "// tiling axis (not a template parameter), so it is not a generation axis.\n\n"
        + '#include "../fai_host_api_impl.hpp"\n\n'
        + "namespace fai_host {\n"
        + f"template aclError launch_fai_dispatch<{ctype}, {is_tnd}>(\n"
        "    uint32_t kernelKey, bool enableDN,\n"
        "    uint32_t blockDim, aclrtStream stream,\n"
        "    uint8_t *qDevice, uint8_t *kDevice, uint8_t *vDevice,\n"
        "    uint8_t *maskDevice, uint8_t *blockTableDevice,\n"
        "    uint8_t *oDevice, uint8_t *lseDevice,\n"
        "    uint8_t *qSeqDevice, uint8_t *kvSeqDevice,\n"
        "    uint8_t *workspaceDevice, uint8_t *tilingDevice);\n"
        + "}  // namespace fai_host\n"
    )
    return Kernel(
        dtype=dtype_key,
        layout=layout_key,
        filename=f"fai_dispatch_{dtype_key}_{layout_key}.cpp",
        content=content,
    )


def get_all_kernels() -> List[Kernel]:
    return [fai_kernel(dtype, layout) for dtype in DTYPE_MAP for layout in LAYOUTS]


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in get_all_kernels():
        (output_dir / kernel.filename).write_text(kernel.content)
    print(f"Generated {len(get_all_kernels())} dispatch TUs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the v3 (950) forward dispatch template instantiations "
        "(per dtype x layout; head_dim is a runtime axis and is not generated).",
    )
    parser.add_argument(
        "-o", "--output_dir", required=False,
        help="Where to generate the dispatch .cpp stubs; "
             "defaults to this script's directory.",
    )
    args = parser.parse_args()
    main(args.output_dir)
