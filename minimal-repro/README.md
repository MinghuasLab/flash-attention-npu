# 最小源码复现：同尺寸自定义 AICPU kernel 部署折叠 bug

独立于 flash-attention-npu 的自包含复现：两个 10 行内核 + 一份胶水 + 一个主程序，
不依赖 torch/catlass/仓库其它代码。结构与 FA 完全同构（每内核一个 DSO、同进程先后
dlopen、首次 <<<>>> 触发运行时惰性部署）。

## 环境
- 驱动 25.5.0（910B3，固件 7.8.0.2.212）：复现；25.5.2：不复现
- CANN ≥ 9.1（bisheng + aicpu 工具链 + libascendcl）

## 结构
```
kernels/min_repro_a.aicpu   内核 A：MinReproKernelA
kernels/min_repro_b.aicpu   内核 B：MinReproKernelB（与 A 仅符号名差、等长
                            → 两者 .aicpu_binary payload 字节数完全相等）
build.sh                    bisheng -x aicpu 编译两个内核，抽取 .aicpu_binary，
                            断言字节数相等（bug 触发条件）
build_host.sh               每内核编译 glue 并与 aicpu 对象链接成 DSO
                            （liblaunch_a.so / liblaunch_b.so），再编译主程序
glue.cpp                    每内核的宿主胶水：<<<>>> 启动 + 同步（bisheng asc 模式）
host.cpp                    主程序：同进程 dlopen A、dlopen B、先后启动
run.sh                      一键：build → build_host → 运行 → 打印设备侧软链
```

## 运行
```bash
bash run.sh
```

## 判读
| 输出 | 含义 |
|---|---|
| glue A sync=0 (OK)，**glue B sync≠0 (FAIL)**（507018 / errcode 11003 get kernel failed）| **bug 复现** |
| A B 均 sync=0 (OK)，设备日志两条软链指向**不同**文件 | 环境健康 |
| build.sh 报 sizes differ | 内核源被改动导致 payload 不再等长，勿继续 |

设备日志软链（run.sh 末尾自动打印）：
```
.../7032015585752036025_9536.so  ->  lib/7032015585752036025_9536.so...   ← A(hash_A)
.../10483255507175688311_9536.so ->  lib/10483255507175688311_9536.so...  ← B(hash_B)
```
健康时两条 target 不同；bug 复现时 B 的 linkPath(宿主 hash) 指向 **A 的 target**(设备读到的
内容 hash)——即"折叠"的直接证据。

## 原理
同进程两次部署**字节数完全相等**的自定义 AICPU kernel 时，设备侧 store 调度器对第二次
load 读到的仍是第一次的旧字节（读到旧 buffer；与 free/unmap 无关），内容 hash 命中第一次
的缓存项，跳过写文件、把第二个 soName 软链到第一个的 store 文件。第二个内核启动时在该
文件里找不到自己的符号 → errcode 11003 get kernel failed。字节数不等则永不触发。
