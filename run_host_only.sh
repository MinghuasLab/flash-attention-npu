#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/tmp/fa-v2-test-deps:/tmp/onerec_ops_deps:/home/hxb/uni-attn/UniAttention
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Each invocation requires a fresh output directory.
python benchmark_host_only.py --output "${1:?Usage: bash run_host_only.sh NEW_OUTPUT_DIRECTORY}" --device 2
