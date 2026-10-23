# `flash_attn_combine` 服务器自验证指南

> 仓库：`/data/w00896777/code/flash-attention-npu`
> 推荐环境：`source /data/w00896777/ops/0930/ascend-toolkit/set_env.sh` + `conda activate wk`
> 预计耗时：编译 5~10 min（首次）；pytest ~9 min（wk 环境）；ATK 200 条约 3 min。

## 0. 一次性环境准备

```bash
source /data/w00896777/ops/0930/ascend-toolkit/set_env.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wk

cd /data/w00896777/code/flash-attention-npu
git submodule update --init --recursive csrc/catlass   # 仅首次

python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
npu-smi info | head -6
```

期望：`ASCEND_HOME_PATH` 指向 `.../cann-9.3.0`，`torch 2.7.1 / torch_npu 2.7.1.post2`，能看到 910B3。

> ⚠️ `wk` 是 Python 3.10；仓库里若已有 cp312 的 `.so`，必须在 `wk` 里重新编译（见第 1 步），否则导入会失败。

## 1. 编译与安装

```bash
cd /data/w00896777/code/flash-attention-npu

# A2/A3（910）v3/v4
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python setup.py build_ext --inplace
FLASH_ATTN_BUILD_VERSION=v4 FLASH_ATTN_BUILD_NPU=910 python setup.py build_ext --inplace

# 检查产物（应为 cp310）
ls flash_attn_npu_3/*.so flash_attn_npu_4/*.so
```

可选：编译 A5（950，仅编译验证，需要工具链支持 `dav-3510`）

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=950 python setup.py build_ext --inplace
FLASH_ATTN_BUILD_VERSION=v4 FLASH_ATTN_BUILD_NPU=950 python setup.py build_ext --inplace
```

## 2. 单例冒烟（先确认能跑）

```bash
cd /data/w00896777/code/flash-attention-npu
python - <<'EOF'
import torch, torch_npu
from flash_attn_npu_3 import flash_attn_combine

op = torch.randn(4, 2, 64, 4, 128, device="npu")          # (NS,B,S,H,D) fp32
lp = torch.randn(4, 2, 4, 64, device="npu").transpose(-1, -2)  # (NS,B,S,H)，seqlen 连续
out, lse = flash_attn_combine(op, lp, out_dtype=torch.bfloat16)
torch.npu.synchronize()
print("out", out.shape, out.dtype, "| lse", lse.shape)
EOF
```

期望输出：`out torch.Size([2, 64, 4, 128]) torch.bfloat16 | lse torch.Size([2, 64, 4])`。

## 3. 跑 pytest（op 级，127 例）

```bash
cd /data/w00896777/code/flash-attention-npu
python -m pytest tests/test_flash_attn_npu_v3_combine.py \
                 tests/test_flash_attn_npu_v4_combine.py -q
```

期望：`127 passed`。

## 4. 跑 ATK（接口级，每后端 200 条；其中 182 条当前有 GPU 基线）

用例文件构成：182 条与旧 GPU 基线逐条对齐 + 18 条新增（id 182..199）。
新增 18 条要进双标杆，需先在 GPU 机器上用同一份 200 条 JSON 重跑一次基线；
刷新前它们只能跑单标杆（见 §4.2）。

双标杆需要 GPU 基线做分母，所以用 `atk node` 把 NPU 与 GPU 基线挂在同一任务里。
旧基线只覆盖 182 条，所以**立刻可跑**的是 182 条那个子集（用 `.182.bak` 复制成一份 json）：

```bash
source /data/w00896777/ops/0930/ascend-toolkit/set_env.sh
source /data/miniconda3/etc/profile.d/conda.sh && conda activate wk
export ASCEND_RT_VISIBLE_DEVICES=3 FLASH_ATTN_NPU_ROOT=/data/w00896777/code/flash-attention-npu
cd /data/w00896777/ATK/flash_attn_combine

cp all_test_flash_attn_combine_bf16fp16.json.182.bak verify_182_fa3.json
cp all_test_flash_attn_combine_v4_bf16fp16.json.182.bak verify_182_fa4.json

# FA3
atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./all_test_flash_attn_combine_reports_2026-10-20-07-10-25.xlsx \
  task -c ./verify_182_fa3.json \
  -p ./execute_flash_attn_combine.py --task accuracy

# FA4
atk node --backend npu --devices 0 \
  node --backend gpu --devices 0 --name gpu_base \
  --bm_file ./all_test_flash_attn_combine_v4_reports_2026-10-20-07-08-39.xlsx \
  task -c ./verify_182_fa4.json \
  -p ./execute_flash_attn_combine.py --task accuracy
```

期望输出（每次约 1 min）：`总用例数 182 / 通过 182 / 通过率 100.0 / 精度是否达标 Pass`。

全量 200 条跑双标杆：先按 `flash_attn_combine_atk_result.md` §5.1 在 GPU 机器上刷新基线，
再直接用 `all_test_flash_attn_combine{,_v4}_bf16fp16.json` 配新报告。

### 4.1 新增 18 条单标杆（无 GPU 基线时的快速验证）

```bash
cd /data/w00896777/ATK/flash_attn_combine
PYTHONPATH=/data/w00896777/code/flash-attention-npu \
  atk pytorch ./all_test_flash_attn_combine_bf16fp16.json \
  -p ./execute_flash_attn_combine.py --task accuracy --devices 3 -s 182 -e 200
# 期望：总用例数 18 / 通过 18 / 通过率 100.0 / Pass
```

FA4 把 `-c` 换成 `all_test_flash_attn_combine_v4_bf16fp16.json`。

### 4.2 yaml 一键脚本（仅冒烟，不含 GPU 分母）

```bash
cd /data/w00896777/code/flash-attention-npu/tests/st/atk_flash_attn_combine
bash run_atk.sh 0 100 3      # 参数：start end device
```

脚本依次执行：

1. `atk case -f test_flash_attn_combine.yaml`（FA3，按 yaml 抽样）
2. `atk case -f test_flash_attn_combine_v4.yaml`（FA4，按 yaml 抽样）
3. `atk pytorch ... --task accuracy`（FA3，npu 被测 + cpu 标杆）
4. `atk pytorch ... --task accuracy`（FA4）

> 该路径只有 CPU 标杆，`cv_fused_double_benchmark` 的比值失去分母，不能作为 L2 验收结论。

### 4.3 手动分步（便于排查）

```bash
cd /data/w00896777/code/flash-attention-npu/tests/st/atk_flash_attn_combine

atk case -f test_flash_attn_combine.yaml -p flash_attn_combine_constraint.py
atk case -f test_flash_attn_combine_v4.yaml -p flash_attn_combine_constraint.py

python - <<'EOF'
import json
for t in ("test_flash_attn_combine", "test_flash_attn_combine_v4"):
    d = json.load(open(f"result/{t}/json/all_{t}.json"))
    print(t, len(d), d[0]["standard"]["acc"])
EOF
# 期望：acc = {'cv_fused_double_benchmark': {'max_re_ratio': 2, 'avg_re_ratio': 1.2, 'root_mean_squared_ratio': 1.2}}

PYTHONPATH=$PWD:/data/w00896777/code/flash-attention-npu \
  atk pytorch result/test_flash_attn_combine/json/all_test_flash_attn_combine.json \
  -p execute_flash_attn_combine.py --task accuracy --devices 3 -s 0 -e 100

PYTHONPATH=$PWD:/data/w00896777/code/flash-attention-npu \
  atk pytorch result/test_flash_attn_combine_v4/json/all_test_flash_attn_combine_v4.json \
  -p execute_flash_attn_combine.py --task accuracy --devices 3 -s 0 -e 100
```

## 5. 结果判读

### 5.1 控制台（通过标志）

```
Total Task: 100, success 100, failed 0
Summary info: acc_pass_result:Pass ...
```

以及汇总表：

```
|  名称 | 总用例数 | 执行成功用例个数 | 执行失败用例个数 | 通过用例个数 | ... | 通过率 | 精度是否达标 |
| cpu_0 |   100    |       100        |        0         |     100      | ... | 100.0  |     Pass     |
```

> `npu_0` 列显示 `-` 是正常的：NPU 是被测端，CPU 是标杆端。

### 5.2 报告（xlsx）

```bash
ls -t atk_output/*/report/*.xlsx | head -2
```

`statistic` 页的 `精度详情` 列包含三项比值：
`最大相对误差比例` / `平均相对误差比例` / `均方根误差比例`，对应 L2 阈值 2 / 1.2 / 1.2。

快速统计（可选）：

```bash
python - <<'EOF'
import glob, ast, pandas as pd
f = sorted(glob.glob("atk_output/all_test_flash_attn_combine_2*/report/*.xlsx"))[-1]
df = pd.read_excel(f, sheet_name="statistic")
mx, av, rm, bad = [], [], [], 0
for s in df["精度详情"]:
    if not isinstance(s, str):
        continue
    for lst in ast.literal_eval(s).values():
        for item in lst:
            ind = item.get("new_benchmark_indicate") or {}
            mx.append(float(ind.get("最大相对误差比例", 0)))
            av.append(float(ind.get("平均相对误差比例", 0)))
            rm.append(float(ind.get("均方根误差比例", 0)))
            bad += 0 if item.get("result", True) else 1
print(f"outputs={len(mx)} false={bad} max_re={max(mx):.4f} avg_re={max(av):.4f} rmse={max(rm):.4f}")
EOF
```

参考结果（双标杆，每后端 182 条）：`false=0 max_re=1.7311 avg_re=0.0023 rmse=0.8047`。

## 6. 常见问题排查

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: No module named 'flash_attn_npu_3'` | 不在仓库根目录，或 `PYTHONPATH` 未包含仓库；从仓库根执行，或 `export PYTHONPATH=/data/w00896777/code/flash-attention-npu` |
| 导入时报 `cannot import name 'flash_attn_npu_3' from partially initialized module` | `.so` 与当前 Python 版本不匹配（cp310 vs cp312）；在 `wk` 里重编 |
| `ImportError: libhccl.so` | 没有 `source set_env.sh` |
| `KeyError 'cv_fused_double_benchmark'` | ATK 版本过旧；需要 26.9.24（`wk` 环境已具备） |
| `RuntimeError: LSE tensor must be contiguous in the seqlen dimension` | 自写执行器时需把 LSE 重排成 seqlen 连续；参考 `execute_flash_attn_combine.py::_to_seqlen_contiguous` |
| `aicore exception` | 先确认设备空闲（`npu-smi info`），并用 `--devices <空闲卡>`；kernel 已处理 MTE2 <32B / rightPad>7 的硬件约束 |
| ATK 用例数不是 100 | 检查 `dtype_numbers`/`extra_numbers`（92 + 8），并确认 `atk case` 的 `save case json` 路径 |
| 跑错 JSON | `atk case` 输出在**当前目录**的 `result/<yaml 文件名去后缀>/json/all_*.json` |
| 设备被占用 | `--devices N` 用可见设备索引，必要时 `ASCEND_RT_VISIBLE_DEVICES=<物理卡>` |

## 7. 已验证结果（参考）

| 项 | 结果 |
|---|---|
| pytest | 127 passed |
| ATK FA3 | 100/100，100%，Pass |
| ATK FA4 | 100/100，100%，Pass |
| L2 比值 | max_re 2.0000（≤2）、avg_re 0.0020（≤1.2）、rmse 0.0503（≤1.2） |
