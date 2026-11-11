"""Persistent-cache recovery and real frontend-free subprocess launches."""

import json
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from flash_attn_npu_4.catlass_dsl import persistent
from flash_attn_npu_4.catlass_dsl.cache_utils import JITCache, JITPersistentCache, get_jit_cache


def factory(value):
    def callback(x, scale=3):
        return x * value * scale

    return callback


def test_callable_fingerprint_captures_constants():
    first, same, different = factory(2), factory(2), factory(4)
    assert persistent.key_digest((first,)) is not None
    assert persistent.key_digest((first,)) == persistent.key_digest((same,))
    assert persistent.key_digest((first,)) != persistent.key_digest((different,))
    assert persistent.key_digest((factory(object()),)) is None
    assert persistent.key_digest((factory([2]),)) is None
    assert persistent.key_digest((factory({"scale": 2}),)) is None


def test_named_cache_configuration(monkeypatch):
    monkeypatch.delenv("FLASH_ATTENTION_CATLASS_DSL_CACHE_ENABLED", raising=False)
    assert type(get_jit_cache("fwd")) is JITCache
    monkeypatch.setenv("FLASH_ATTENTION_CATLASS_DSL_CACHE_ENABLED", "1")
    assert isinstance(get_jit_cache("fwd"), JITPersistentCache)
    with pytest.raises(ValueError):
        get_jit_cache("../fwd")


@pytest.fixture
def disk_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_ATTENTION_CATLASS_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CATLASS_DSL_FORCE_RECOMPILE", raising=False)
    monkeypatch.setattr(persistent, "source_fingerprint", lambda: "source-abi-v1")
    monkeypatch.setattr(
        persistent,
        "export_function",
        lambda fn, directory: (directory / "value.json").write_text(json.dumps(fn)),
    )
    monkeypatch.setattr(
        persistent,
        "load_function",
        lambda directory: json.loads((directory / "value.json").read_text()),
    )
    return JITPersistentCache("test")


def test_disk_hit_corruption_clear_and_force(disk_cache, monkeypatch):
    key = ("fwd", "3510", "fp16")
    compiler = Mock(side_effect=[1, 2, 3, 4])
    assert disk_cache.get_or_compile(key, compiler) == 1
    disk_cache.clear()
    assert disk_cache.get_or_compile(key, compiler) == 1
    assert compiler.call_count == 1
    entry = next(disk_cache._directory().glob("fwd/*/value.json"))
    entry.write_text("broken")
    disk_cache.clear()
    assert disk_cache.get_or_compile(key, compiler) == 2
    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "1")
    assert disk_cache.get_or_compile(key, compiler) == 3
    monkeypatch.delenv("CATLASS_DSL_FORCE_RECOMPILE")
    disk_cache.clear(persistent=True)
    assert disk_cache.get_or_compile(key, compiler) == 4


@pytest.mark.parametrize("error", [OSError("disk full"), KeyError("invalid backend manifest")])
def test_disk_failure_keeps_working_memory_cache(disk_cache, monkeypatch, error):
    monkeypatch.setattr(persistent, "export_function", Mock(side_effect=error))
    compile_fn = Mock(return_value=11)
    with pytest.warns(RuntimeWarning, match="export failed"):
        assert disk_cache.get_or_compile(("bwd",), compile_fn) == 11
    assert disk_cache.get_or_compile(("bwd",), compile_fn) == 11
    compile_fn.assert_called_once()


def test_device_owners_are_isolated_but_disk_keys_are_shared():
    import torch

    keys = [("fwd", torch.device(f"npu:{index}"), "3510") for index in (0, 1)]
    cache = JITCache()
    assert cache.get_or_compile(keys[0], lambda: 1) == 1
    assert cache.get_or_compile(keys[1], lambda: 2) == 2
    assert len(cache) == 2
    assert persistent.key_digest(keys[0]) == persistent.key_digest(keys[1])


def test_failed_compile_never_publishes(disk_cache):
    with pytest.raises(RuntimeError, match="compiler failure"):
        disk_cache.get_or_compile(("mask",), Mock(side_effect=RuntimeError("compiler failure")))
    assert not disk_cache
    assert not list(disk_cache._directory().rglob("value.json"))
    assert disk_cache.get_or_compile(("mask",), lambda: 17) == 17


def test_unavailable_namespace_falls_back_once(disk_cache, monkeypatch):
    monkeypatch.setattr(
        persistent, "source_fingerprint", Mock(side_effect=RuntimeError("requires CATLASS 2.2"))
    )
    with pytest.warns(RuntimeWarning, match="requires CATLASS 2.2"):
        assert disk_cache.get_or_compile(("fwd",), lambda: 11) == 11
    assert disk_cache.get_or_compile(("bwd",), lambda: 12) == 12


def test_new_source_or_abi_namespace_does_not_reuse(disk_cache, monkeypatch):
    assert disk_cache.get_or_compile(("fwd",), lambda: 1) == 1
    monkeypatch.setattr(persistent, "source_fingerprint", lambda: "source-abi-v2")
    next_cache = JITPersistentCache("test")
    assert next_cache.get_or_compile(("fwd",), lambda: 2) == 2
    disk_cache.clear()
    assert disk_cache.get_or_compile(("fwd",), Mock(side_effect=AssertionError("must load"))) == 1


def test_competing_processes_compile_once(tmp_path):
    worker = r"""
import json, sys, time
from pathlib import Path
from flash_attn_npu_4.catlass_dsl import persistent
from flash_attn_npu_4.catlass_dsl.cache_utils import JITPersistentCache
persistent.source_fingerprint = lambda: "concurrent"
persistent.export_function = lambda fn, p: (p/"value.json").write_text(json.dumps(fn))
persistent.load_function = lambda p: json.loads((p/"value.json").read_text())
def compile_fn():
    with Path(sys.argv[1]).open("a") as log:
        log.write("compiled\\n")
    time.sleep(0.2)
    return 17
assert JITPersistentCache("race").get_or_compile(("mask",), compile_fn) == 17
"""
    env = dict(
        os.environ,
        FLASH_ATTENTION_CATLASS_DSL_CACHE_DIR=str(tmp_path),
        CATLASS_DSL_FORCE_RECOMPILE="0",
    )
    log = tmp_path / "compile.log"
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(log)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=60)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert log.read_text().count("compiled") == 1


_WORKER = r"""
import sys
import torch, torch_npu
import catlass.tla as tla
from flash_attn_npu_4.catlass_dsl import flash_attn_func, runtime
from flash_attn_npu_4.catlass_dsl.cache_utils import JITPersistentCache
from tests.catlass_dsl.test_utils import make_inputs, oracle, check
length, forbid = int(sys.argv[1]), int(sys.argv[2])
kind = sys.argv[3]
assert isinstance(runtime.compile_cache, JITPersistentCache)
compile_original = tla.compile
calls = []
def compile_once(*args, **kwargs):
    assert not forbid, "disk hit must bypass tla.compile and frontend lowering"
    calls.append(1)
    return compile_original(*args, **kwargs)
tla.compile = compile_once
if kind == "mask":
    from flash_attn_npu_4.catlass_dsl import compute_block_sparsity
    from tests.catlass_dsl.test_dynamic_cache import causal_mask
    from tests.catlass_dsl.test_block_sparsity import active_lists
    from tests.catlass_dsl.reference import reference_block_sparsity
    result = compute_block_sparsity(128, 128, 1, 1, length, length+6, causal_mask, None, device='npu:0')
    expected = reference_block_sparsity(1, 1, length, length+6, lambda b,h,q,k,i,a,c: q>=k)
    assert active_lists(result) == active_lists(expected)
elif kind == "mxfp8":
    from flash_attn_npu_4.catlass_dsl import flash_attn_varlen_func
    from tests.catlass_dsl.mxfp8_reference import make_case, reference, to_device, check_quantized_lse
    case = make_case(q_lengths=(length,), k_lengths=(length+6,), q_packed=True, k_packed=True)
    inputs, options = to_device(case, 'npu:0')
    out, lse = flash_attn_varlen_func(*inputs, **options, return_lse=True)
    expected, logs = reference(case)
    check(out, expected)
    check_quantized_lse(lse.cpu(), logs)
else:
    cpu, inputs, _, _, _ = make_inputs('npu:0', torch.float16, sq=length, sk=length+6)
    if kind == "aux":
        from tests.catlass_dsl.test_runtime import score, score_bwd
        from tests.catlass_dsl.test_utils import check_attention
        data = torch.linspace(-0.2,0.2,length)
        out,lse = flash_attn_func(*inputs,score_mod=score,score_mod_bwd=score_bwd,
                                  aux_tensors=(data.to('npu:0'),),aux_scalars=(0.7,),return_lse=True)
        check_attention(cpu,inputs,out,lse,loss="both",
                        score_mod=lambda s,b,h,q,k,i,a,c:s*c[0]+a[0][q],
                        aux_tensors=(data,),aux_scalars=(0.7,))
    else:
        with torch.no_grad():
            out, lse = flash_attn_func(*inputs, return_lse=True)
            expected, logs = oracle(cpu)
            check(out, expected)
            check(lse, logs)
assert len(calls) == (0 if forbid else 2 if kind == "aux" else 1), calls
print("CACHE_WORKER_OK")
"""


@pytest.mark.npu
@pytest.mark.parametrize("kind", ["dense", "aux", "mxfp8", "mask"])
def test_subprocess_load_bypasses_frontend_for_new_length(tmp_path, kind):
    env = dict(
        os.environ,
        FLASH_ATTENTION_CATLASS_DSL_CACHE_ENABLED="1",
        FLASH_ATTENTION_CATLASS_DSL_CACHE_DIR=str(tmp_path),
        CATLASS_DSL_FORCE_RECOMPILE="0",
    )
    for length, forbid in ((17, 0), (33, 1)):
        process = subprocess.run(
            [sys.executable, "-c", _WORKER, str(length), str(forbid), kind],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert "CACHE_WORKER_OK" in process.stdout
    assert list(tmp_path.rglob("manifest.json"))
    if kind == "dense":
        # Corruption is a miss, including for a previously successful binary.
        next(tmp_path.rglob("kernel.o")).write_bytes(b"truncated")
        for length, forbid in ((49, 0), (65, 1)):
            process = subprocess.run(
                [sys.executable, "-c", _WORKER, str(length), str(forbid), kind],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert process.returncode == 0, process.stdout + process.stderr
