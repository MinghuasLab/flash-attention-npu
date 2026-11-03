"""Executor rebinding and producer/consumer streams through the public API."""

import catlass.tla as tla
import pytest
import torch
from .test_utils import DTYPES, check_attention, copy_aux_to_device, make_inputs

from flash_attn_npu_4.catlass_dsl import flash_attn_func, runtime

pytestmark = pytest.mark.npu


@tla.jit
def score(s, b, h, q, k, info, aux, scalars):
    return s * scalars[0] + aux[0][q]


@tla.jit
def score_bwd(g, s, b, h, q, k, info, aux, scalars):
    return g * scalars[0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_cache_rebinds_inputs_aux_and_scalars(device, dtype):
    previous = []
    keys = None
    for seed, value in ((701, 0.3), (702, 1.2)):
        cpu, inputs, _, _, _ = make_inputs(device, dtype, seed=seed)
        data = torch.linspace(-value, value, 34).reshape(17, 2)[:, 1]
        aux = copy_aux_to_device(data, device)
        out, lse = flash_attn_func(
            *inputs,
            score_mod=score,
            score_mod_bwd=score_bwd,
            aux_tensors=[aux],
            aux_scalars=(value,),
            return_lse=True,
        )
        reference = lambda s, b, h, q, k, info, buffers, scalars: s * scalars[0] + buffers[0][q]
        check_attention(
            cpu, inputs, out, lse, score_mod=reference, aux_tensors=[data], aux_scalars=(value,)
        )
        if previous:
            assert set(runtime.compile_cache) == keys
            assert all(old.data_ptr() != new.data_ptr() for old, new in zip(previous, inputs))
        keys = set(runtime.compile_cache)
        previous = inputs


@pytest.mark.parametrize("dtype", DTYPES)
def test_producer_consumer_stream(device, dtype):
    producer, consumer = torch.npu.Stream(device=device), torch.npu.Stream(device=device)
    cpu, _, _, _, _ = make_inputs("cpu", dtype, sq=128, sk=128)
    with torch.npu.stream(producer):
        inputs = [t.detach().to(device, non_blocking=True).requires_grad_() for t in cpu]
    with torch.npu.stream(consumer):
        consumer.wait_stream(producer)
        out, lse = flash_attn_func(*inputs, return_lse=True)
        check_attention(cpu, inputs, out, lse, loss="both")
    torch.npu.current_stream(device).wait_stream(consumer)
