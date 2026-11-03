"""Build and run the standalone device smoke using this repo's pinned CATLASS."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    repo = Path(__file__).resolve().parents[1]
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    build = Path(tempfile.mkdtemp(prefix="fa3-d512-smoke."))
    print(f"Smoke artifacts: {build}", flush=True)
    flags = [
        "-x",
        "asc",
        "--npu-arch=dav-3510",
        "--cce-auto-infer-kernel-type=false",
        "-DCATLASS_ARCH=3510",
        "-std=c++17",
        "-O2",
        f"-I{repo}/csrc/ascend950/flash_attn_npu_3",
        f"-I{repo}/csrc/catlass/include",
        f"-I{cann}/compiler/tikcpp/include",
        f"-I{cann}/include",
    ]
    commands = [
        (
            "kernel-build.log",
            [
                "bisheng",
                *flags,
                "-c",
                str(repo / "tests/cpp/fa3_d512_smoke_kernel.cpp"),
                "-o",
                str(build / "kernel.o"),
            ],
        ),
        (
            "host-build.log",
            [
                "bisheng",
                *flags,
                str(repo / "tests/cpp/fa3_d512_smoke.cpp"),
                "-x",
                "none",
                str(build / "kernel.o"),
                f"-L{cann}/lib64",
                "-lascendcl",
                "-o",
                str(build / "smoke"),
            ],
        ),
    ]
    for log_name, command in commands:
        with (build / log_name).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--paged", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--layout", choices=["tnd", "bsnd"], default="tnd")
    parser.add_argument("--q-length", type=int, default=128)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("lengths", nargs="*", type=int, default=[128])
    args = parser.parse_args()
    if args.q_length <= 0 or any(length <= 0 for length in args.lengths):
        parser.error("Q and KV lengths must be positive")
    if args.cores < 1 or args.heads < 1 or args.kv_heads < 1 or args.heads % args.kv_heads:
        parser.error("cores and heads must be positive and heads must be divisible by kv-heads")
    if (args.paged or args.causal) and (
        args.dtype != "fp16"
        or args.layout != "tnd"
        or args.heads != 1
        or args.kv_heads != 1
        or (args.paged and args.causal)
        or not args.random
    ):
        parser.error("paged/causal smoke currently requires random FP16 TND, one head and one mode")
    if args.varlen and (
        args.layout != "tnd"
        or args.paged
        or args.causal
        or args.q_length < 2
        or any(kv < 2 for kv in args.lengths)
    ):
        parser.error("varlen smoke requires dense TND with Q/KV lengths >= 2")
    random_mode = args.random
    lengths = args.lengths
    mode = (
        "paged" if args.paged else "causal" if args.causal else "varlen" if args.varlen else "dense"
    )
    batch = 2 if args.layout == "bsnd" or args.varlen else 1
    if random_mode:
        import numpy as np
        import torch

        sys.path.insert(0, str(repo))
        from tests.common.attention_ref import ref_flash_attention_pair
        from tests.common.compare import assert_fa_close

        torch.set_num_threads(1)
    for run_index, kv in enumerate(lengths):
        env = os.environ.copy()
        if random_mode:
            rng = np.random.default_rng(512 + kv + args.q_length)
            dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

            def random_tensor(rows):
                return torch.from_numpy(rng.standard_normal((rows, 512)).astype(np.float32)).to(
                    dtype
                )

            q = random_tensor((batch * args.q_length - int(args.varlen)) * args.heads)
            k = random_tensor((batch * kv - int(args.varlen)) * args.kv_heads)
            v = random_tensor((batch * kv - int(args.varlen)) * args.kv_heads)
            input_path, output_path = build / f"input-{kv}.bin", build / f"output-{kv}.bin"
            with input_path.open("wb") as data:
                data.write(q.view(torch.uint16).numpy().tobytes())
                for tensor in (k, v):
                    if args.paged:
                        blocks = (kv + 127) // 128
                        padded = torch.zeros((blocks * 128, 512), dtype=dtype)
                        padded[:kv] = tensor
                        tensor = padded.reshape(blocks, 128, 512).flip(0).contiguous()
                    data.write(tensor.view(torch.uint16).numpy().tobytes())
            env.update(FA_D512_RANDOM_INPUT=str(input_path), FA_D512_RANDOM_OUTPUT=str(output_path))
        with (build / f"run-{run_index}-{kv}.log").open("w") as log:
            try:
                subprocess.run(
                    [
                        str(build / "smoke"),
                        str(kv),
                        str(args.q_length),
                        str(int(args.dtype == "bf16")),
                        str(int(args.layout == "bsnd")),
                        str(args.cores),
                        str(args.heads),
                        str(args.kv_heads),
                        str(int(args.paged)),
                        str(int(args.causal)),
                        str(int(args.varlen)),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=env,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                print(f"KV={kv}: timeout; inspect {build}/run-{run_index}-{kv}.log", flush=True)
                return 124
        print((build / f"run-{run_index}-{kv}.log").read_text(), end="")
        if random_mode:
            query = q.reshape(-1, args.heads, 512)
            key = k.reshape(-1, args.kv_heads, 512)
            value = v.reshape(-1, args.kv_heads, 512)
            actual = (
                torch.from_numpy(np.fromfile(output_path, dtype=np.uint16))
                .view(dtype)
                .reshape(-1, args.heads, 512)
            )
            q_start = k_start = 0
            if args.varlen:
                for i, (q_len, k_len) in enumerate(
                    [(args.q_length, kv), (args.q_length - 1, kv - 1)]
                ):
                    ref, _, base, _ = ref_flash_attention_pair(
                        query[q_start : q_start + q_len].unsqueeze(0),
                        key[k_start : k_start + k_len].unsqueeze(0),
                        value[k_start : k_start + k_len].unsqueeze(0),
                        512**-0.5,
                        None,
                        dtype,
                        differentiable=True,
                    )
                    got = actual[q_start : q_start + q_len].unsqueeze(0)
                    err = (got.float() - ref.float()).abs().max().item()
                    assert_fa_close(got, ref, base, name=f"batch={i} Q={q_len} KV={k_len}")
                    print(
                        f"{args.dtype} varlen batch={i} Q={q_len} KV={k_len} max_abs={err:.6g}: PASS",
                        flush=True,
                    )
                    q_start += q_len
                    k_start += k_len
                continue
            query = query.reshape(batch, args.q_length, args.heads, 512)
            key = key.reshape(batch, kv, args.kv_heads, 512)
            value = value.reshape(batch, kv, args.kv_heads, 512)
            mask = (
                torch.triu(
                    torch.ones((args.q_length, kv), dtype=torch.bool),
                    diagonal=1 + kv - args.q_length,
                )
                if args.causal
                else None
            )
            reference, _, baseline, _ = ref_flash_attention_pair(
                query, key, value, 512**-0.5, mask, dtype, differentiable=True
            )
            actual = actual.reshape(batch, args.q_length, args.heads, 512)
            error = (actual.float() - reference.float()).abs()
            baseline_error = (baseline.float() - reference.float()).abs().max().item()
            print(
                f"{args.dtype} {args.layout} {mode} B={batch} H={args.heads}/{args.kv_heads} cores={args.cores} Q={args.q_length} KV={kv}: max_abs={error.max().item():.6g}, "
                f"rms={error.square().mean().sqrt().item():.6g}, "
                f"baseline_max_abs={baseline_error:.6g}",
                flush=True,
            )
            for start in range(0, args.q_length, 128):
                task_error = error[:, start : start + 128]
                print(
                    f"  Q rows [{start}, {min(start + 128, args.q_length)}): "
                    f"max_abs={task_error.max().item():.6g}",
                    flush=True,
                )
            assert_fa_close(actual, reference, baseline, name=f"Q={args.q_length} KV={kv}")
            print("  repository dual-reference comparison: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
