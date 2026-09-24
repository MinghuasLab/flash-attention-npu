# Copyright (c) 2026, Minghua Shen.
"""CI quick-mode random sampling.

When ``--random-sample=N`` is passed, pytest collects normally and then this
hook keeps at most N items per test function (grouped by ``file::func``,
i.e. the nodeid with its parametrize suffix stripped). Selection uses a fixed
seed (env ``CI_RANDOM_SEED``, default 0) so the same commit re-runs the same
subset; failures can be reproduced by re-running with the same seed.

Only enabled by ``--random-sample`` (CI quick mode). full mode passes no such
flag and runs everything. Compatible with ``-k`` (filter applies during
collection, before this hook) and ``pytest-xdist -n`` (sampling happens in the
master collection phase, before items are dispatched to workers).
"""

import os
import random
import functools
import time
import pytest
from collections import defaultdict
from tests.common.timing import add, begin, finish, set_current_state


def _wrap_kernel_functions(module):
    """Time imported flash-attention entry points without editing every test."""
    names = {
        "flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache",
        "_flash_attn_backward",
    }
    for name in names:
        function = getattr(module, name, None)
        if not callable(function) or getattr(function, "_ci_timed", False):
            continue

        part = "backward" if name == "_flash_attn_backward" else "forward"

        @functools.wraps(function)
        def timed(*args, __function=function, __part=part, **kwargs):
            state = getattr(timed, "_ci_state", None)
            if state is None:
                return __function(*args, **kwargs)
            import torch
            sync = getattr(getattr(torch, "npu", None), "synchronize", None)
            # Synchronization changes the execution semantics of asynchronous
            # NPU launches and can break graph/compile tests.  Keep timing
            # observational by default; opt in only when exact kernel timing
            # is explicitly requested.
            sync_enabled = (
                os.environ.get("CI_TIMING_SYNC", "0") == "1"
                and not getattr(timed, "_ci_sync_disabled", False)
            )
            if sync and sync_enabled:
                sync()
            started = time.perf_counter()
            try:
                return __function(*args, **kwargs)
            finally:
                if sync and sync_enabled:
                    sync()
                if state is not None:
                    add(state, __part, time.perf_counter() - started)

        timed._ci_timed = True
        setattr(module, name, timed)


def _wrap_reference_functions(module):
    """Time direct CPU/reference helpers used by dedicated backward tests."""
    names = {
        "torch_ref_fwd_bsnd", "golden_bsnd_bwd_from_fwd", "golden_tnd_bwd_from_fwd",
        "ref_flash_attention",
    }
    for name in names:
        function = getattr(module, name, None)
        if not callable(function) or getattr(function, "_ci_ref_timed", False):
            continue

        @functools.wraps(function)
        def timed(*args, __function=function, **kwargs):
            state = getattr(timed, "_ci_state", None)
            started = time.perf_counter()
            try:
                return __function(*args, **kwargs)
            finally:
                if state is not None:
                    add(state, "ref", time.perf_counter() - started)

        timed._ci_ref_timed = True
        setattr(module, name, timed)


def _wrap_prepare_functions(module):
    """Time common input/cache/mask preparation helpers."""
    parts = {
        "make_random_tensor": "input",
        "make_packed_random_tensor": "input",
        "make_attention_inputs": "input",
        "make_paged_kv_cache": "cache",
        "make_block_table": "index",
        "gather_paged_kv_batch": "postprocess",
        "pad_packed_tensor": "postprocess",
        "make_padded_varlen_mask": "postprocess",
        "make_local_attention_mask": "postprocess",
    }
    for name, part in parts.items():
        function = getattr(module, name, None)
        if not callable(function) or getattr(function, "_ci_prepare_timed", False):
            continue

        @functools.wraps(function)
        def timed(*args, __function=function, __part=part, **kwargs):
            state = getattr(timed, "_ci_state", None)
            started = time.perf_counter()
            try:
                return __function(*args, **kwargs)
            finally:
                if state is not None:
                    add(state, __part, time.perf_counter() - started)

        timed._ci_prepare_timed = True
        timed._ci_prepare_part = part
        setattr(module, name, timed)


def pytest_runtest_setup(item):
    _wrap_kernel_functions(item.module)
    _wrap_reference_functions(item.module)
    _wrap_prepare_functions(item.module)
    # Synchronizing an NPU stream while it is inside ``torch.npu.graph``
    # capture raises ACL_ERROR_RT_STREAM_CAPTURED (107027).  Graph tests still
    # exercise the same entry points, but their launch timing is necessarily
    # asynchronous, so disable only the timing hook's synchronization for the
    # duration of those tests.
    is_graph_test = "_graph" in str(item.fspath).lower()
    for name in ("flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache"):
        function = getattr(item.module, name, None)
        if callable(function) and getattr(function, "_ci_timed", False):
            function._ci_sync_disabled = is_graph_test


def pytest_addoption(parser):
    parser.addoption(
        "--random-sample",
        action="store",
        type=int,
        default=0,
        help="Quick mode: randomly sample at most N items per test function "
        "(fixed seed via CI_RANDOM_SEED, default 0). 0 = no sampling.",
    )


def pytest_report_header(config):
    n = config.getoption("--random-sample") or 0
    if n <= 0:
        return []
    seed = int(os.environ.get("CI_RANDOM_SEED", "0"))
    return [
        f"random-sample: at most {n} items per test function, "
        f"seed={seed} (override via CI_RANDOM_SEED)"
    ]


def pytest_collection_modifyitems(config, items):
    n = config.getoption("--random-sample") or 0
    if n <= 0:
        return
    seed = int(os.environ.get("CI_RANDOM_SEED", "0"))
    rng = random.Random(seed)

    # Group by test function: nodeid is "tests/x.py::test_func[a-b-c]";
    # stripping the "[...]" suffix gives "tests/x.py::test_func".
    groups = defaultdict(list)
    for it in items:
        groups[it.nodeid.split("[", 1)[0]].append(it)

    keep = set()
    # sorted() over keys so the per-group rng.sample() order is deterministic
    # across runs and across pytest versions -> reproducible selection.
    for key in sorted(groups):
        grp = groups[key]
        chosen = grp if len(grp) <= n else rng.sample(grp, n)
        keep.update(it.nodeid for it in chosen)

    # Preserve original collection order.
    items[:] = [it for it in items if it.nodeid in keep]


@pytest.fixture(autouse=True)
def _seed_per_case(request):
    """Seed torch's default RNG (CPU + NPU) per test case for reproducibility.

    Default: stable distinct seed per case = crc32(pytest node id), so a failure
    can be reproduced by re-running that exact case. Set env CI_TORCH_SEED to
    override with one global seed for all cases (e.g. CI_TORCH_SEED=0 gives
    torch.manual_seed(0) everywhere). Rand calls passing an explicit
    ``generator=`` use that generator and are unaffected.
    """
    import torch
    import zlib
    env_seed = os.environ.get("CI_TORCH_SEED")
    seed = int(env_seed) if env_seed else zlib.crc32(
        request.node.nodeid.encode("utf-8"))
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed"):
        torch.npu.manual_seed(seed)
    os.environ["GOLDEN_CACHE_NODEID"] = request.node.nodeid
    os.environ["GOLDEN_CACHE_TEST_FILE"] = str(request.path)
    state = begin()
    set_current_state(state)
    import tests.common.golden_cache as golden_cache
    import tests.common.compare as compare
    golden_cache.get_or_compute_golden._ci_timing_state = state
    compare.assert_fa_close._ci_timing_state = state
    for name in ("flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache"):
        function = getattr(request.module, name, None)
        if callable(function) and getattr(function, "_ci_timed", False):
            function._ci_state = state
    for name in ("_flash_attn_backward", "torch_ref_fwd_bsnd",
                 "golden_bsnd_bwd_from_fwd", "golden_tnd_bwd_from_fwd",
                 "ref_flash_attention"):
        function = getattr(request.module, name, None)
        if callable(function) and getattr(function, "_ci_timed", False):
            function._ci_state = state
        elif callable(function) and getattr(function, "_ci_ref_timed", False):
            function._ci_state = state
    for name in ("make_random_tensor", "make_packed_random_tensor", "make_paged_kv_cache",
                 "make_block_table", "gather_paged_kv_batch", "pad_packed_tensor",
                 "make_padded_varlen_mask", "make_local_attention_mask",
                 "make_attention_inputs"):
        function = getattr(request.module, name, None)
        if callable(function) and getattr(function, "_ci_prepare_timed", False):
            function._ci_state = state
    yield
    timing = finish(state, request.node.nodeid)
    request.node.user_properties.append(("ci_timing", timing))
    for name in ("flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache"):
        function = getattr(request.module, name, None)
        if callable(function) and getattr(function, "_ci_timed", False):
            function._ci_state = None
    for name in ("_flash_attn_backward", "torch_ref_fwd_bsnd",
                 "golden_bsnd_bwd_from_fwd", "golden_tnd_bwd_from_fwd",
                 "ref_flash_attention"):
        function = getattr(request.module, name, None)
        if callable(function) and (getattr(function, "_ci_timed", False)
                                   or getattr(function, "_ci_ref_timed", False)):
            function._ci_state = None
    for name in ("make_random_tensor", "make_packed_random_tensor", "make_paged_kv_cache",
                 "make_block_table", "gather_paged_kv_batch", "pad_packed_tensor",
                 "make_padded_varlen_mask", "make_local_attention_mask",
                 "make_attention_inputs"):
        function = getattr(request.module, name, None)
        if callable(function) and getattr(function, "_ci_prepare_timed", False):
            function._ci_state = None
    golden_cache.get_or_compute_golden._ci_timing_state = None
    compare.assert_fa_close._ci_timing_state = None
    set_current_state(None)
    os.environ.pop("GOLDEN_CACHE_NODEID", None)
    os.environ.pop("GOLDEN_CACHE_TEST_FILE", None)
