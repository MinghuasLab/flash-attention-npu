# Copyright (c) 2026, Minghua Shen.
"""Persistent, best-effort cache for CPU reference (golden) tensors.

The cache is deliberately independent from pytest and NPU state.  Callers pass
case metadata, the CPU inputs used by the reference, and a function that
returns a mapping of tensors.  A cache failure only causes a recomputation.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping

import torch


_FORMAT_VERSION = 1
_DEFAULT_CACHE_DIR = "/var/cache/flash-attention-npu/golden_cache"


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().to(device="cpu").contiguous()
    data = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def input_digest(inputs: Any) -> Any:
    """Return a deterministic, content-based digest description for inputs."""
    if isinstance(inputs, torch.Tensor):
        return _tensor_digest(inputs)
    if isinstance(inputs, Mapping):
        return {str(key): input_digest(inputs[key]) for key in sorted(inputs, key=str)}
    if isinstance(inputs, (list, tuple)):
        return [input_digest(item) for item in inputs]
    return _json_value(inputs)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_digest(paths: list[str] | tuple[str, ...] | None) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths or ()):
        hasher.update(path.encode())
        try:
            hasher.update(Path(path).read_bytes())
        except OSError:
            hasher.update(b"<missing>")
    return hasher.hexdigest()


def _cache_root() -> Path:
    return Path(os.environ.get("GOLDEN_CACHE_DIR", _DEFAULT_CACHE_DIR))


def _cache_enabled() -> bool:
    return os.environ.get("GOLDEN_CACHE_MODE", "off").strip().lower() not in {
        "off", "0", "false", "disabled"
    }


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _limits() -> tuple[int, int]:
    def get(name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, default)))
        except ValueError:
            return default

    max_common = get("GOLDEN_CACHE_MAX_DIRS", 5)
    return max_common, get("GOLDEN_CACHE_MAX_TEST_DIRS", max_common)


@contextlib.contextmanager
def _exclusive_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "golden_cache.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_artifact(path: Path, expected: dict[str, Any]) -> dict[str, torch.Tensor]:
    with tempfile.TemporaryDirectory(prefix="golden-read-", dir=path.parent) as temp:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError("cache archive contains an unsupported member")
                target = Path(temp, member.name).resolve()
                if not str(target).startswith(str(Path(temp).resolve()) + os.sep):
                    raise ValueError("cache archive contains an unsafe path")
            archive.extractall(temp)
        metadata = json.loads(Path(temp, "metadata.json").read_text())
        if metadata != expected:
            raise ValueError("cache metadata does not match current case")
        result: dict[str, torch.Tensor] = {}
        for tensor_path in sorted(Path(temp).glob("*.pt")):
            with tensor_path.open("rb") as stream:
                try:
                    result[tensor_path.stem] = torch.load(
                        stream, map_location="cpu", weights_only=True
                    )
                except TypeError:  # torch versions before weights_only
                    stream.seek(0)
                    result[tensor_path.stem] = torch.load(stream, map_location="cpu")
        if not result:
            raise ValueError("cache artifact contains no tensors")
        expected_names = set(expected["value_names"])
        if set(result) != expected_names:
            raise ValueError(
                f"cache artifact tensors {sorted(result)} do not match "
                f"expected {sorted(expected_names)}"
            )
        return result


def _write_artifact(
    path: Path,
    metadata: dict[str, Any],
    values: Mapping[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".case-", suffix=".tar.gz", dir=path.parent, delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
    try:
        with tempfile.TemporaryDirectory(prefix="golden-write-", dir=path.parent) as temp:
            Path(temp, "metadata.json").write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            )
            for name, value in values.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"golden value {name!r} is not a tensor")
                with Path(temp, f"{_safe_name(name)}.pt").open("wb") as stream:
                    torch.save(value.detach().to(device="cpu"), stream)
            with tarfile.open(temp_path, "w:gz") as archive:
                for child in sorted(Path(temp).iterdir()):
                    archive.add(child, arcname=child.name)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _evict(root: Path) -> None:
    max_common, max_test = _limits()
    common_dirs = [path for path in root.glob("common_*") if path.is_dir()]
    common_dirs.sort(
        key=lambda path: max(
            (item.stat().st_mtime for item in path.rglob("*.tar.gz")),
            default=path.stat().st_mtime,
        )
    )
    for path in common_dirs[:-max_common]:
        shutil.rmtree(path, ignore_errors=True)
    for common in common_dirs[-max_common:]:
        test_dirs = [path for path in common.glob("test_*") if path.is_dir()]
        test_dirs.sort(
            key=lambda path: max(
                (item.stat().st_mtime for item in path.glob("case_*.tar.gz")),
                default=path.stat().st_mtime,
            )
        )
        for path in test_dirs[:-max_test]:
            shutil.rmtree(path, ignore_errors=True)


def get_or_compute_golden(
    *,
    nodeid: str,
    metadata: Mapping[str, Any],
    inputs: Any,
    compute_fn: Callable[[], Mapping[str, torch.Tensor]],
    expected_keys: tuple[str, ...],
    source_files: list[str] | tuple[str, ...] | None = None,
    test_source_files: list[str] | tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Load a case artifact or compute and atomically persist it.

    ``compute_fn`` is called only on a miss, a damaged artifact, or when
    ``GOLDEN_CACHE_REFRESH=1`` is set.  Returned tensors are detached CPU
    tensors on a cache hit and retain the caller's tensors on a miss.
    """
    if not _cache_enabled():
        return dict(compute_fn())

    value_names = sorted(set(expected_keys))
    if len(value_names) != len(expected_keys):
        raise ValueError("expected_keys must contain unique names")
    case_metadata = {
        "format_version": _FORMAT_VERSION,
        "nodeid": nodeid,
        "metadata": _json_value(dict(metadata)),
        "inputs": input_digest(inputs),
        "source_digest": _source_digest(source_files),
        "test_source_digest": _source_digest(test_source_files),
        "runtime": {"torch": torch.__version__},
        "value_names": value_names,
    }
    common_hash = _sha256_json(
        {"source": case_metadata["source_digest"], "runtime": case_metadata["runtime"]}
    )[:16]
    seed = _safe_name(
        str(dict(metadata).get("seed", os.environ.get("CI_TORCH_SEED", "per-case")))
    )
    test_hash = _sha256_json(
        {
            "test_file": nodeid.split("::", 1)[0],
            "source": case_metadata["test_source_digest"],
        }
    )[:16]
    case_hash = _sha256_json(case_metadata)[:32]
    root = _cache_root()
    artifact = (
        root
        / f"common_{common_hash}_seed_{seed}"
        / f"test_{test_hash}"
        / f"case_{case_hash}.tar.gz"
    )

    refresh = _env_bool("GOLDEN_CACHE_REFRESH")
    if not refresh and artifact.is_file():
        try:
            result = _load_artifact(artifact, case_metadata)
            print(f"[golden-cache] hit {nodeid}")
            return result
        except Exception as exc:  # cache is an optimization, never a test failure
            warnings.warn(
                f"golden cache read failed for {nodeid}: {exc}; recomputing",
                RuntimeWarning,
            )

    values = dict(compute_fn())
    if set(values) != set(value_names):
        raise ValueError(
            f"computed golden tensors {sorted(values)} do not match "
            f"expected {value_names}"
        )
    try:
        with _exclusive_lock(root):
            _write_artifact(artifact, case_metadata, values)
            _evict(root)
        print(f"[golden-cache] {'refresh' if refresh else 'miss'} {nodeid}")
    except Exception as exc:
        warnings.warn(
            f"golden cache write failed for {nodeid}: {exc}; continuing",
            RuntimeWarning,
        )
    return values


__all__ = ["get_or_compute_golden", "input_digest"]
