# FA3 Ascend950 migration regressions

Source: `origin/fa4-pipeline-paged-strides-rebase` at `c9a7b26`, relative to
`be95ec4`. These tests port the branch's **910 FA3** changes to the existing
950 FA3 implementation.

Covered behavior:

- Logical `[page, token, head, dim]` caches backed by BBND or BNBD storage;
  independent K/V page strides and storage offsets, including split-KV.
- Contiguous fallback for unsupported read-only Q/K/V views; append requests
  cannot silently replace a non-contiguous cache with a temporary copy.
- Zero-length sequences and the `total_q = max_seqlen_q = 0` return path.
- Packed tensors longer than `cu_seqlens[-1]`, correct LSE head pitch in
  ordinary and split-KV execution, and rejection of undersized packed tensors.
- Zero gradients for unused tails, including deterministic MHA/GQA backward
  and independently padded K/V tensors.
- Bottom-right SWA when Q is longer than K, including `seqused_q/k`.
- Reusing scheduler metadata across different storage geometries and NPU graph
  capture after warmup validation.

Storage geometry is passed per kernel launch; the scheduler metadata ABI is
unchanged. Existing 950 limits still apply: head dimensions up to 256, page
sizes 128/256/512/1024, no append-KV execution, and no SWA backward. FA4-only
pipeline and attention-sink changes are outside this FA3 migration.

Build the 950 target (no 950 device is needed for compilation):

```bash
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v3 \
FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace
```

The CATLASS submodule must already be initialized and the CANN environment
loaded. The extension uses the local Python ABI and host CPU architecture;
build again on a target with a different host architecture.

Run checks that do not launch NPU kernels:

```bash
python -m pytest -q tests/test_flash_attn_npu_v3_950_stride_policy.py
```

Run the migrated numerical and graph regressions on Ascend950:

```bash
python -m pytest -q tests/test_flash_attn_npu_v3_950_migration.py
```

The latter suite explicitly skips on other devices. A successful cross-build
or CPU test run does not establish 950 numerical correctness.
