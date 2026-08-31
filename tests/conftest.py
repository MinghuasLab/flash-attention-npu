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
import pytest
from collections import defaultdict


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
    yield
    os.environ.pop("GOLDEN_CACHE_NODEID", None)
    os.environ.pop("GOLDEN_CACHE_TEST_FILE", None)
