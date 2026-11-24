# Ascend 950DT BF16 window pipeline experiment

These benchmark overlays apply to source commit `6202fd9`. The case uses
`B=1, H=64, S=4096, D=256`, physical BNSD storage, a 511-token left window,
a zero-token right window, BF16 PSE, and a learnable FP32 sink. Performance runs
use a zero-initialized sink. Accuracy checks also exercise nonzero sink values.

| Main kernel | CANN 9.2.1 | Experiment | Reduction |
| --- | ---: | ---: | ---: |
| FA4 forward | 730.1 µs | 479.8 µs | 34.3% |
| FA3 backward | 2069.3 µs | 1352.2 µs | 34.7% |

Each row summarizes two groups of seven retained samples on one device.
Forward ran on device 3 and backward ran on device 2. The busiest pipes were
MTE2 at about 87% in forward and Fixpipe at about 76% in backward.

## Build

Use an isolated checkout, Python 3.11, PyTorch 2.14.0+cpu, torch_npu 2.14.0,
and the CANN 9.2.1 environment. Run the following commands from the repository
root. The original builds used the workspace's CANN compatibility headers via
`FLASH_ATTN_COMPAT_INCLUDE`.

```sh
git submodule update --init --recursive
bench=benchmarks/ascend950_window_pipeline
git apply "$bench/fa4-forward.patch"
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v4 \
  python setup.py build_ext --build-lib build/window-fa4 --build-temp build/window-fa4-obj

git apply "$bench/fa3-backward.patch"
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v3 \
  python setup.py build_ext --build-lib build/window-fa3 --build-temp build/window-fa3-obj
```

Build FA4 before applying the FA3 overlay because the FA3 overlay changes
shared backward headers and narrows FA3 backward dispatch to the benchmark
case. The profiled backward path is the FA3 extension, using the output and
LSE produced by FA4. FA4 backward remains outside the scope of this benchmark.

## Verify and profile

Run from outside the repository so in-place extensions cannot shadow the
separately built binaries. Replace `repo` with the absolute checkout path.

```sh
repo=/path/to/flash-attention-npu
bench="$repo/benchmarks/ascend950_window_pipeline"
export PYTHONPATH="$repo/build/window-fa3:$repo/build/window-fa4:$repo"
export FAG_TEST_DEVICE=0
cd /tmp

python "$bench/forward.py" accuracy -1 30208
for seed in 0 394 766; do
  python "$bench/backward_accuracy.py" "$seed" 269551368
done
msprof --ai-core=on --aic-metrics=PipeUtilization --output=window-forward \
  python "$bench/forward.py" 30208 30208
msprof --ai-core=on --aic-metrics=PipeUtilization --output=window-backward \
  python "$bench/backward.py"
```

Forward mode `30208` uses the fixed-shape QK path and PV buffer reuse. It
leaves the first-V prefetch experiment disabled. Backward mode `269551368`
uses UB accumulation for dV, L0A reuse for dQ and dK, and a revised issue order.
Forward validates bitwise equality of output and LSE. Backward compares
tolerance failures against CANN and reports elementwise differences from the
unoptimized schedule.
