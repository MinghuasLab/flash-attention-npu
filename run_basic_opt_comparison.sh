#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/tmp/fa-v2-test-deps:/tmp/onerec_ops_deps:/home/hxb/uni-attn/UniAttention
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
result_root="${1:?Usage: bash run_basic_opt_comparison.sh NEW_RESULT_DIRECTORY}"
mkdir "$result_root"
npu-smi info > "$result_root/npu_before.txt"
python ../host_original_before_basic_opt/benchmark_host_only.py --output "$result_root/host_baseline" --device 2 > "$result_root/host_baseline.log" 2>&1
python benchmark_host_only.py --output "$result_root/ported" --device 2 > "$result_root/ported.log" 2>&1
npu-smi info > "$result_root/npu_after.txt"
python summarize_basic_opt.py --baseline "$result_root/host_baseline" --ported "$result_root/ported"
