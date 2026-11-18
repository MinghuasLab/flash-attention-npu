"""Executor rebinding and producer/consumer streams through the public API."""

import gc
import weakref
from unittest.mock import Mock

import catlass.tla as tla
import pytest
import torch
import torch_npu

from .test_utils import DTYPES, check, check_attention, copy_aux_to_device, make_inputs, oracle

from flash_attn_npu_4.catlass_dsl import builtin, flash_attn_func, flash_attn_varlen_func, runtime
from flash_attn_npu_4.catlass_dsl.cache_utils import get_jit_cache

pytestmark = pytest.mark.npu


@pytest.mark.parametrize(
    "failure", [None, "binding", "launch"], ids=["success", "binding-error", "launch-error"]
)
def test_launch_releases_tensor_owners_without_gc(monkeypatch, failure):
    """Launch providers must not retain tensors until cyclic GC runs."""
    monkeypatch.setattr(torch.npu, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    original_tensor = runtime._tensor
    calls = 0

    def bind(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "binding" and calls == 2:
            raise RuntimeError("binding failed")
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(runtime, "_tensor", bind)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:

        def invoke():
            tensor = torch.empty(32)
            other = torch.empty(32)
            references = [weakref.ref(tensor), weakref.ref(other)]

            def executor(*args, **kwargs):
                if failure == "launch":
                    raise RuntimeError("launch failed")

            if failure:
                try:
                    runtime._launch(executor, (tensor, other), block_num=1, device=tensor.device)
                except RuntimeError as error:
                    assert str(error) == f"{failure} failed"
                else:
                    pytest.fail("launch did not raise")
            else:
                runtime._launch(executor, (tensor, other), block_num=1, device=tensor.device)
            return references

        references = invoke()
        assert all(reference() is None for reference in references)
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("cross_stream", [False, True], ids=["same-stream", "cross-stream"])
def test_launch_preserves_only_needed_stream_records(device, cross_stream):
    """Same-stream reuse is immediate; side-stream use still defers release."""
    if (
        not callable(getattr(torch_npu, "erase_stream", None))
        or torch.npu.get_allocator_backend() != "native"
    ):
        pytest.skip("native allocator erase_stream is unavailable")
    torch.npu.synchronize(device)
    gc.collect()
    torch.npu.empty_cache()
    producer = torch.npu.Stream(device=device)
    consumer = torch.npu.Stream(device=device) if cross_stream else producer
    before = torch.npu.memory_stats(device)
    with torch.npu.stream(producer):
        tensor = torch.empty(1024 * 1024, dtype=torch.float32, device=device)
    size = tensor.numel() * tensor.element_size()
    reference = weakref.ref(tensor)
    with torch.npu.stream(consumer):
        # A no-op executor isolates allocator lifetime rules from kernel work.
        # Duplicate/nested arguments must keep the same storage contract.
        runtime._launch(lambda *a, **kw: None, (tensor, (tensor,)), block_num=1, device=device)
        del tensor
        assert reference() is None
        after = torch.npu.memory_stats(device)
        assert after["allocated_bytes.all.current"] == before["allocated_bytes.all.current"]
        pending = after["active_bytes.all.current"] - before["active_bytes.all.current"]
        if cross_stream:
            assert pending >= size
        else:
            assert pending == 0
    torch.npu.synchronize(device)


@pytest.mark.parametrize("dtype", DTYPES)
def test_cross_stream_inputs_survive_early_release_and_reuse(device, dtype):
    """A producer must not overwrite inputs still consumed by asynchronous FA."""
    producer = torch.npu.Stream(device=device)
    consumer = torch.npu.Stream(device=device)
    shape = (1, 1024, 4, 128)
    cpu = tuple(torch.randn(shape, dtype=dtype) for _ in range(3))
    expected, expected_lse = oracle(cpu)
    delay_done = torch.npu.Event()
    with torch.inference_mode(), torch.npu.stream(consumer):
        # Warm public operators before the race, including their allocations.
        delay_input = torch.ones((4096, 4096), dtype=torch.float32, device=device)
        delay_output = torch.empty_like(delay_input)
        torch.mm(delay_input, delay_input, out=delay_output)
        delay_done.record(consumer)
        warm = tuple(t.to(device) for t in cpu)
        flash_attn_func(*warm, return_lse=True)
        del warm
    with torch.inference_mode(), torch.npu.stream(producer):
        warm_pressure = torch.full(shape, 99, dtype=dtype, device=device)
        del warm_pressure
    torch.npu.synchronize(device)
    gc.collect()
    with torch.inference_mode():
        with torch.npu.stream(producer):
            inputs = tuple(t.to(device) for t in cpu)
        references = tuple(weakref.ref(t) for t in inputs)
        input_ptrs = {t.data_ptr() for t in inputs}
        try:
            with torch.npu.stream(consumer):
                consumer.wait_stream(producer)
                # Keep FA queued while the producer tries to reuse its inputs.
                # Reuse buffers to avoid output allocations during the delay.
                for _ in range(32):
                    torch.mm(delay_input, delay_input, out=delay_output)
                delay_done.record(consumer)
                output, lse = flash_attn_func(*inputs, return_lse=True)
            del inputs
            assert all(reference() is None for reference in references)
            with torch.npu.stream(producer):
                # Recycle allocations of the same size before waiting for FA.
                pressure = [torch.full(shape, 99, dtype=dtype, device=device) for _ in range(8)]
            assert not delay_done.query(), "consumer delay completed before allocator pressure"
            assert input_ptrs.isdisjoint(t.data_ptr() for t in pressure)
        finally:
            torch.npu.synchronize(device)
        check(output, expected)
        check(lse, expected_lse)
        del pressure


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
    producer = torch.npu.Stream(device=device)
    consumers = [torch.npu.Stream(device=device) for _ in range(2)]
    keys = None
    for step, consumer in enumerate(consumers):
        cpu, _, _, _, _ = make_inputs("cpu", dtype, sq=128, sk=128, seed=1909 + step)
        with torch.npu.stream(producer):
            inputs = [t.detach().to(device, non_blocking=True).requires_grad_() for t in cpu]
        with torch.npu.stream(consumer):
            consumer.wait_stream(producer)
            out, lse = flash_attn_func(*inputs, return_lse=True)
            check_attention(cpu, inputs, out, lse, loss="both")
        torch.npu.current_stream(device).wait_stream(consumer)
        if keys is not None:
            assert set(runtime.compile_cache) == keys
        keys = set(runtime.compile_cache)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", [False, True], ids=["bsnd", "tnd"])
def test_cached_plan_rebinds_scale_and_segment_boundaries(device, dtype, packed):
    """A warm plan must read fresh scales, aux values and packed boundaries."""
    mask = builtin.causal(alignment="bottom_right")
    keys = None
    for step, (scale, value) in enumerate(((0.0, 0.3), (-0.125, 1.2))):
        cpu, inputs, cq, ck, options = make_inputs(
            device,
            dtype,
            qp=packed,
            kp=packed,
            sq=17,
            sk=23,
            q_lengths=(5, 12) if step == 0 else (8, 9),
            k_lengths=(13, 10) if step == 0 else (9, 14),
            seed=701 + step,
        )
        data = torch.linspace(-value, value, 34).reshape(17, 2)[:, 1]
        out, lse = flash_attn_varlen_func(
            *inputs,
            **options,
            max_seqlen_q=17,
            max_seqlen_k=23,
            score_mod=score,
            score_mod_bwd=score_bwd,
            mask_mod=mask,
            softmax_scale=scale,
            aux_tensors=[copy_aux_to_device(data, device)],
            aux_scalars=(value,),
            return_lse=True,
        )
        check_attention(
            cpu,
            inputs,
            out,
            lse,
            cq=cq,
            ck=ck,
            loss="both",
            scale=scale,
            score_mod=lambda s, b, h, q, k, info, aux, scalars: s * scalars[0] + aux[0][q],
            mask_mod=lambda b, h, q, k, info, aux, scalars: k <= q + info[1] - info[0],
            aux_tensors=(data,),
            aux_scalars=(value,),
        )
        if keys is not None:
            assert set(runtime.compile_cache) == keys
        keys = set(runtime.compile_cache)


def test_attention_plan_eviction_and_clear(device, monkeypatch):
    """Both item eviction and clear must discard the prepared executor."""
    cache = get_jit_cache()
    monkeypatch.setattr(runtime, "compile_cache", cache)
    compiler = Mock(wraps=runtime._compile_kernel)
    monkeypatch.setattr(runtime, "_compile_kernel", compiler)
    for step, expected_compilations in enumerate((1, 1, 2, 3)):
        if step == 2:
            cache.pop(next(iter(cache)))
        elif step == 3:
            runtime.clear_compile_cache()
            assert not cache
        cpu, inputs, _, _, _ = make_inputs(device, torch.float16, seed=903 + step)
        with torch.no_grad():
            out, lse = flash_attn_func(*inputs, return_lse=True)
            expected, expected_lse = oracle(cpu)
        check(out, expected)
        check(lse, expected_lse)
        assert compiler.call_count == expected_compilations
