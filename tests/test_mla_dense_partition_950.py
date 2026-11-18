# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Dense MLA core ranges that start/end inside a resident KV group."""

import pytest
import torch

from tests.test_mla_k_reuse_950 import test_mla_k_reuse as _check_mla_backward


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,seqlen,heads,kv_heads",
    [
        (1, 1, 2, 1),
        (1, 64, 6, 2),
        (2, 129, 8, 2),
        (3, 257, 6, 2),
        (2, 513, 8, 1),
    ],
)
def test_mla_dense_partition(version, dtype, batch, seqlen, heads, kv_heads):
    _check_mla_backward(version, dtype, batch, seqlen, heads, kv_heads, False)
