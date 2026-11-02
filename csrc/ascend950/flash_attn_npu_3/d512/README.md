# FA3 / Ascend950 D512 development

This directory is an independent implementation under development. The D512 public path is under development. D<=256 continues to use the existing implementation.

Status at the initial `26837f3` checkpoint (historical):

- `config.hpp`: 128x128 task policy, 416 KiB L1 layout, per-AIV UB offsets,
  capacity-derived LO row chunks, and per-AIC two-slot LO GM addressing.
- Host `tiling.cpp` and `fa_metadata.aicpu` select the shared D512 policy for
  Q/K/V dimensions equal to 512. No metadata ABI fields were added.
- `qk_matmul.hpp`: independent QK specialization, adapted from this version's
  existing implementation to load K in four 128-dimensional slices while
  retaining the full Q layout and accumulating S across slices.
- `pv_matmul.hpp`: independent PV specialization retaining the existing
  per-column-block compute/Fixpipe/release order. GM destination integration
  is pending in the D512 kernel.

Validation performed at that checkpoint:

- `python -m pytest -q tests/test_fa3_d512_policy.py`: passes. Compiles and runs
  C++ checks against the real tiling structure, including GQA row limits, UB
  capacity and disjoint per-core/slot GM addresses.
- Header syntax checked with the Ascend950 compiler and this repository's
  pinned `csrc/catlass` commit `769cd40a`. This parses templates; it is not a
  full kernel instantiation, link, or device correctness test.

Historical remaining-work list at that checkpoint:

1. Adapt online softmax/P conversion and mask lifetimes to the new UB layout.
   The existing P padding and two mask work areas must not silently overlap GO.
2. Implement GM LO row-chunk rescale, keeping GO resident and preserving row
   statistics, fully masked rows, LSE and window semantics.
3. Connect kernel/dispatch and per-core workspace; synchronize host/AICPU task
   packing and kernel scheduling. Verify cache/page source strides in sliced K.
4. Instantiate/build the device kernels following the repository README, then
   compare single/two/five KV-block outputs against the existing attention
   reference. Include Q/KV tails and cross-task slot reuse.
5. Cover paged cache, new KV, layouts, masks, FD and old-dimension regression;
   only then adapt FA4 and compare performance against A3.

Use `README.zh.md` / `README.md` in this repository for build instructions.
The compiler dependency is the pinned `csrc/catlass`, not a neighboring checkout.

## Isolated device path and rollback verification (2026-10-28)

The uncommitted D512 kernel now connects sliced QK, packed P, PV-to-GM LO,
and row-chunk rescale. LO uses two one-row UB buffers per AIV because the
existing two mask work areas and P layout leave less space than the original
GBSA estimate. The public API and dispatch remain at checkpoint `26837f3`.

Concurrent changes to `fwd_dispatch.hpp`, `fwd_dispatch_impl.hpp`, `mha_fwd.cpp`
and `tests/test_flash_attn_npu_v3.py` were reverted to that checkpoint. Concurrent
kernel/helper renames and the replacement PV GM copy were also removed while
preserving this isolated implementation. Pre-rollback copies are stored in
`/tmp/fa3-d512-concurrent-backup-osjl56vt` on the server.

After rollback, the standalone FP16/TND/nonpaged/no-mask smoke passed for
Q=128 and KV=128 (twice), 256, 640 and 129. Each checks all 65536 output elements
for Q=K=0 and V=1. The policy test passed (1 test). Device build/run artifacts:
`/tmp/fa3-d512-smoke.wrrnohla`; aggregate log:
`/tmp/fa3_d512_rollback_matrix.log`.

The earlier stream synchronization timeout was not reproduced in these runs.
Its cause is not established. These uniform-input smokes do not validate random
attention values, all synchronization schedules, multiple Q tasks, GQA, BF16,
paged cache, new KV, masks, LSE or FD. No public D512 interface is enabled and
no extension install/build completion is claimed.

## Random FP16 smoke

Run `python tests/run_fa3_d512_smoke.py --random 128 129 256 640`
after sourcing CANN. Each case uses a fixed seed, FP16 TND inputs, one Q
tile of 128 rows, one head, no mask and normal KV cache. The repository
`ref_flash_attention_pair` produces the two reference outputs and
`assert_fa_close` applies the same dual-reference comparison used by FA3 tests.
This is a maximum-error rule; equivalence to the requirement's L2 dual-reference
acceptance must be confirmed. Historical measurements below used an exploratory
FP32 reference with atol=0.002 and rtol=0.01, not the repository rule.

On 2026-10-28, all 65536 elements per case passed. Maximum absolute errors
for KV=128, 129, 256 and 640 were 0.000245631, 0.000259042, 0.000194609
and 0.000122160 respectively. This does not cover multiple Q tasks, BF16,
masks, paged cache, new KV, LSE, FD, or the complete public interface.

## Multiple Q tasks

The isolated smoke can take `--q-length`; it launches one AIC, so Q lengths
256 and 384 exercise two and three sequential 128-row tasks on the same core.
On 2026-10-28, FP16 TND random tests with KV=128, 129, 256 and 640 passed
for both Q lengths using the historical exploratory tolerances above. Q=257 with KV=129
and 640 also passed, including a one-row final Q task. Each task's rows were
compared against the FP32 reference. Q=384/KV=640 was repeated three more
times with the same seed and completed successfully. These checks do not
establish race freedom, official precision acceptance or multicore behavior.

Example: `python tests/run_fa3_d512_smoke.py --random --q-length 384 128 129 256 640`

## Repository comparison update

The isolated random smoke now directly calls the FA3 test suite's
`ref_flash_attention_pair` and `assert_fa_close` on FP16 output tensors.
The reference runs without the golden cache. This verifies the repository's
current comparison path, not the as-yet unconfirmed official L2 criterion.

On 2026-10-28, repository comparison passed for Q=256 with KV=128, 129,
256, 640, and for Q=128, 257, 384 with KV=129 and 640 (10 cases total).
All cases used FP16 TND, one head, no mask and normal KV cache.

## BF16 and BSND development checks

The standalone smoke accepts `--dtype fp16|bf16` and `--layout tnd|bsnd`.
TND uses batch 1; BSND uses batch 2 with equal allocated Q/KV lengths and one
head. Random tests compare against the repository dual-reference rule. This
is isolated kernel testing, not public API or official L2 acceptance.

A FP16 BSND case with Q=257, KV=128 and batch=2 initially produced a large
error in part of the second batch. The preceding one-row Q tail leaves one
Vector subcore without work. Its PV-ready acknowledgement uses V, while the
next task waits on MTE2; the current change orders these pipes in that empty
path. The case passed afterward, including repeated runs. This is evidence for
the synchronization hypothesis, not proof that all races are resolved.

Example: `python tests/run_fa3_d512_smoke.py --random --dtype bf16 --layout bsnd --q-length 257 128 129 640`

For the exact failing FP16 BSND Q=257/KV=128 input, the pre-change binary
reproduced max absolute error 1.1993 in three further runs. The adjusted
binary passed the repository comparison in repeated runs on the same seed.
BF16 TND and BSND were exercised for Q=128, 256, 257 with KV=128, 129, 640;
FP16 TND and BSND were also checked. BSND uses two batches, TND one batch,
and both use one head. The official L2 criterion, multicore, GQA, masks,
paged cache, append KV, LSE and FD remain to be validated.

## Reduced independent-kernel scenarios

The smoke supports `--cores`, `--heads` and `--kv-heads`. Workspace is sized
per launched AIC and task counts follow the D512 head packing policy.
Representative repository dual-reference checks passed for BSND batch 2,
Q=257/KV=129 on two AICs; TND GQA 16/2, Q=16/KV=129 on two AICs; BSND
MHA 4/4, Q=129/KV=128 on two AICs; and one task on two AICs.

`--paged` tests FP16 TND with one head, a 128-token block size, and reversed
physical block order; Q=16/KV=129 passed. `--causal` tests the 2048-square
triu(1) device mask and bottom-right aligned reference; Q=16/KV=129 passed.
These are representative cases, not exhaustive paged/masked API coverage.
No claim is made for the official L2 precision requirement.

## Packed TND variable-length smoke

`--varlen` creates two packed batches with Q lengths `[Q, Q-1]` and KV
lengths `[KV, KV-1]`; it currently covers dense TND without causal mask.
The device receives cumulative lengths, and each batch is compared separately
using the repository dual-reference rule. FP16 GQA 16/2 with Q=[129,128],
KV=[129,128] and two AICs passed. BF16 GQA 16/2 with Q=[17,16] and
KV=[129,128] passed; these Q lengths select different head tile sizes.
This is not validation of arbitrary sequence distributions or public APIs.

## Initial public kvcache integration (in progress)

`flash_attn_with_kvcache` routes matching Q/K/V dimensions of 512 to the
independent kernel using host tiling. The non-Flash-Decode workspace is now
allocated for the per-core GM LO buffers. Automatic split-KV selection falls
back to ordinary tiling for D512. Explicit split-KV, scheduler metadata, LSE,
softcap, SWA, BNSD and paged blocks other than 128 are not yet supported by
this D512 integration. These limitations do not apply to the existing D<=256
path. The Python wrapper uses host tiling when D512 metadata is not supplied.

The `tests/test_fa3_d512_integration.py` cases run the public kvcache path
against repository dual references: D256 and D512, both dtypes, TND/BSND,
causal and non-causal, reverse-table paged KV, TND variable lengths, and a
paged decode with `num_splits=0`. Explicit unsupported split-KV and LSE paths
are checked for errors. On 2026-10-28, 14 cases passed with
`ASCEND_RT_VISIBLE_DEVICES=0`; official L2 precision and other features remain
unvalidated.

## 2026-10-29: shared task kernel (uncommitted)

The public D512 dispatch now uses `FAInfer` with the D512 compile-time flag and
shares `FAIKernel950` task scheduling with the original path. The D512 compute
components and memory layout remain separate. `kernel.hpp` is only the standalone
smoke adapter to that shared task kernel. No change to the D512 feature gates or
workspace policy is implied by this refactor.

Validation: v3/950 build/install, policy test (1 passed), integration tests
(14 passed), and BF16 BSND B=2/Q=257/two-core random smoke with KV=128/129/640
passed the repository dual-reference comparison. Performance is not validated.

## 2026-10-29: accuracy policy

The temporary mentor comparison adapter has been removed. Development tests use
`tests/common/compare.py::assert_fa_close` again. ATK fused dual-benchmark accuracy
validation will be planned separately; its FA4 adapter cannot be used unchanged
for the current FA3 D512 path.


## 2026-10-29：LSE接通（未提交）

公开kvcache的D512路径开放 `return_softmax_lse=True`。移除Host的LSE拒绝项，
dispatch允许实例化已有的 `LseMode=true` D512 kernel；计算组件与空间布局不变。
LSE沿用 `WriteGroupedLse`：最后一个KV块完成O写回后，等待MTE3读取GO完成，
复用GO区域扩展LSE，再写入GM。没有新增UB、L1或GM LO buffer。
BSND返回FP32 `[B,H,Sq]`，TND返回FP32 `[H,total_Q]`。

接口测试扩到D64/128/192/256/512，并新增两种dtype下的LSE测试，覆盖单KV块、
跨KV块、Q257尾块、分页、decode、不等长多batch及causal全mask行。
全mask行沿用生态仓约定：O=0、LSE=+inf。显式split-KV拒绝测试仍保留。
ATK增加独立LSE配置，核实每项同时比较 `output_0.pt`（O）与 `output_1.pt`（LSE）。
SWA、softcap、外部metadata、BNSD和其他分页block大小仍未开放；本次不改变FA4。


LSE空任务补充：Q257/KV1 causal发现原InitOutputs950固定O区域
[192,256) KiB与LSE区域[226,228) KiB重叠（D256仅需[192,224) KiB）。
D512现在复用GO内部的[96,160) KiB存放FP16/BF16零O，
[160,162) KiB存放展开的+inf LSE。两段均在既有GO [96,224) KiB内；
D<=256继续使用原默认地址。回归固定BSND/TND、两种dtype的Q257/KV1场景。

最终修复后验证：v3/950构建安装成功，接口45项通过；原kvcache/BNSD相关集
227项通过、20项跳过；ATK LSE16项及O-only 100+32项通过。阈值未改变。
源码仍未提交，正式性能和内存验收尚未完成。

## Shared policy checkpoint (2026-11-03)

The layout and tiling policy now lives in `../fwd_policy.hpp`, shared by host, AICPU and device code. The `flash_attn::d512` namespace and all layout values remain unchanged. The QK/PV/Softmax/Rescale implementations below are still separate components pending convergence. No tiling-data or metadata ABI fields were added.
