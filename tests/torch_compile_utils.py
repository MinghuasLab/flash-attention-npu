import importlib
import inspect

import pytest
import torch
import torch_npu


DEVICE = "npu:0"
ATOL = 2e-3
RTOL = 2e-3


def soc_name():
    if not hasattr(torch, "npu"):
        return ""
    if not torch.npu.is_available():
        return ""
    return torch.npu.get_device_name(0)


def require_soc(prefix):
    name = soc_name()

    if prefix not in name:
        pytest.skip(
            f"requires Ascend {prefix}, current device is {name!r}"
        )

    return name


def sync():
    torch.npu.synchronize()


def load_api(module_name):
    return importlib.import_module(module_name)


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item

    raise AssertionError(
        f"expected Tensor or tuple/list containing Tensor, got {type(value)}"
    )


def check_required_parameters(sig, kwargs, ignored=()):
    missing = []

    for name, param in sig.parameters.items():
        if name in ignored:
            continue

        if name in kwargs:
            continue

        if param.default is inspect._empty:
            missing.append(name)

    if missing:
        pytest.fail(
            f"unsupported required parameters in current API: {missing}; "
            f"signature={sig}"
        )


def metadata_kwargs(api, causal, window_size):
    sig = inspect.signature(api.get_scheduler_metadata)

    candidates = {
        "batch_size": 2,
        "max_seqlen_q": 16,
        "max_seqlen_k": 16,
        "num_heads_q": 6,
        "num_heads_kv": 6,
        "headdim": 32,
        "headdim_v": 32,
        "qkv_dtype": torch.float16,

        "cu_seqlens_q": None,
        "cu_seqlens_k": None,
        "cu_seqlens_k_new": None,
        "seqused_q": None,
        "cache_leftpad": None,

        "page_size": None,
        "max_seqlen_k_new": 0,

        "causal": causal,
        "window_size": window_size,
        "window_size_left": window_size[0],
        "window_size_right": window_size[1],

        "attention_chunk": 0,
        "softcap": 0.0,
        "num_splits": 1,
        "pack_gqa": None,
        "sm_margin": 0,
        "deterministic": False,

        "softmax_scale": 32 ** -0.5,
        "alibi_slopes_batch_stride": 0,
        "learnable_sink": None,
    }

    kwargs = {
        name: candidates[name]
        for name in sig.parameters
        if name in candidates
    }

    check_required_parameters(
        sig,
        kwargs,
        ignored=("cache_seqlens",),
    )

    return kwargs


def run_metadata_compile_test(api, expected_sizes=None):
    cache_seqlens = torch.tensor(
        [16, 16],
        dtype=torch.int32,
        device=DEVICE,
    )

    cases = [
        ("NO_MASK", False, (-1, -1)),
        ("CAUSAL", True, (-1, -1)),
        ("LOCAL_LEFT", False, (4, -1)),
        ("LOCAL_RIGHT", False, (-1, 4)),
        ("FULL_WINDOW_COLLAPSE", False, (16, 16)),
    ]

    observed_sizes = {}

    for case_name, causal, window_size in cases:
        static_kwargs = metadata_kwargs(
            api,
            causal=causal,
            window_size=window_size,
        )

        def fn(cache):
            return api.get_scheduler_metadata(
                cache_seqlens=cache,
                **static_kwargs,
            )

        eager = fn(cache_seqlens)

        assert isinstance(eager, torch.Tensor)
        assert eager.dtype == torch.uint8
        assert eager.device.type == "npu"
        assert eager.dim() == 1
        assert eager.is_contiguous()
        assert not eager.requires_grad

        if expected_sizes is not None:
            assert eager.numel() == expected_sizes[case_name]

        eager_fp = getattr(
            eager,
            "_fa_scheduler_params",
            None,
        )

        torch._dynamo.reset()

        compiled_fn = torch.compile(
            fn,
            backend="aot_eager",
            fullgraph=True,
        )

        compiled = compiled_fn(cache_seqlens)
        sync()

        assert isinstance(compiled, torch.Tensor)
        assert compiled.shape == eager.shape
        assert compiled.numel() == eager.numel()
        assert compiled.dtype == eager.dtype
        assert compiled.device == eager.device
        assert compiled.stride() == eager.stride()
        assert compiled.is_contiguous() == eager.is_contiguous()
        assert compiled.requires_grad == eager.requires_grad

        compiled_fp = getattr(
            compiled,
            "_fa_scheduler_params",
            None,
        )

        assert (compiled_fp is None) == (eager_fp is None)

        observed_sizes[case_name] = compiled.numel()

    assert (
        observed_sizes["NO_MASK"]
        == observed_sizes["FULL_WINDOW_COLLAPSE"]
    )

    base = observed_sizes["NO_MASK"]

    assert observed_sizes["CAUSAL"] > base
    assert observed_sizes["LOCAL_LEFT"] > base
    assert observed_sizes["LOCAL_RIGHT"] > base


def varlen_kwargs(api, cu_q, cu_k, max_seq):
    sig = inspect.signature(api.flash_attn_varlen_func)

    candidates = {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": max_seq,
        "max_seqlen_k": max_seq,

        "seqused_q": None,
        "seqused_k": None,

        "softmax_scale": 32 ** -0.5,
        "causal": False,

        "qv": None,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,

        "window_size": (-1, -1),
        "attention_chunk": 0,
        "softcap": 0.0,

        "num_splits": 1,
        "pack_gqa": None,
        "deterministic": False,
        "sm_margin": 0,

        "return_attn_probs": False,
        "return_lse": True,

        "scheduler_metadata": None,
        "disable_scheduler_metadata": False,

        "alibi_slopes": None,
        "learnable_sink": None,
    }

    kwargs = {
        name: candidates[name]
        for name in sig.parameters
        if name in candidates
    }

    check_required_parameters(
        sig,
        kwargs,
        ignored=("q", "k", "v"),
    )

    return kwargs


def run_varlen_compile_test(api, backward):
    torch.manual_seed(20260909)

    total_q = 6
    total_k = 6
    nheads = 6
    head_dim = 32
    max_seq = 3

    q_base = torch.randn(
        total_q,
        nheads,
        head_dim,
        dtype=torch.float16,
        device=DEVICE,
    )

    k_base = torch.randn(
        total_k,
        nheads,
        head_dim,
        dtype=torch.float16,
        device=DEVICE,
    )

    v_base = torch.randn(
        total_k,
        nheads,
        head_dim,
        dtype=torch.float16,
        device=DEVICE,
    )

    cu_q = torch.tensor(
        [0, 3, 6],
        dtype=torch.int32,
        device=DEVICE,
    )

    cu_k = torch.tensor(
        [0, 3, 6],
        dtype=torch.int32,
        device=DEVICE,
    )

    kwargs = varlen_kwargs(
        api,
        cu_q=cu_q,
        cu_k=cu_k,
        max_seq=max_seq,
    )

    def fn(q, k, v):
        return api.flash_attn_varlen_func(
            q,
            k,
            v,
            **kwargs,
        )

    q_eager = q_base.detach().clone()
    k_eager = k_base.detach().clone()
    v_eager = v_base.detach().clone()

    q_compiled = q_base.detach().clone()
    k_compiled = k_base.detach().clone()
    v_compiled = v_base.detach().clone()

    if backward:
        q_eager.requires_grad_(True)
        k_eager.requires_grad_(True)
        v_eager.requires_grad_(True)

        q_compiled.requires_grad_(True)
        k_compiled.requires_grad_(True)
        v_compiled.requires_grad_(True)

    eager_ret = fn(
        q_eager,
        k_eager,
        v_eager,
    )

    eager_out = first_tensor(eager_ret)
    sync()

    torch._dynamo.reset()

    compiled_fn = torch.compile(
        fn,
        backend="aot_eager",
        fullgraph=True,
    )

    compiled_ret = compiled_fn(
        q_compiled,
        k_compiled,
        v_compiled,
    )

    compiled_out = first_tensor(compiled_ret)
    sync()

    assert compiled_out.shape == eager_out.shape
    assert compiled_out.dtype == eager_out.dtype
    assert compiled_out.device == eager_out.device

    torch.testing.assert_close(
        compiled_out.float(),
        eager_out.float(),
        atol=ATOL,
        rtol=RTOL,
    )

    if not backward:
        return

    eager_out.float().sum().backward()
    sync()

    compiled_out.float().sum().backward()
    sync()

    for eager_grad, compiled_grad in [
        (q_eager.grad, q_compiled.grad),
        (k_eager.grad, k_compiled.grad),
        (v_eager.grad, v_compiled.grad),
    ]:
        assert eager_grad is not None
        assert compiled_grad is not None

        torch.testing.assert_close(
            compiled_grad.float(),
            eager_grad.float(),
            atol=ATOL,
            rtol=RTOL,
        )



def run_fixed_compile_test(api, backward=True):
    """
    Verify fixed-length FlashAttention correctness.

    Compare:
    - eager forward output
    - torch.compile forward output
    - eager gradients
    - torch.compile gradients
    """

    torch.manual_seed(20260909)

    batch = 2
    seq = 16
    heads = 6
    dim = 32


    q = torch.randn(
        batch,
        seq,
        heads,
        dim,
        dtype=torch.float16,
        device=DEVICE,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)


    def fn(q,k,v):
        return api.flash_attn_func(
            q,
            k,
            v,
        )


    q1 = q.clone().detach()
    k1 = k.clone().detach()
    v1 = v.clone().detach()


    q2 = q.clone().detach()
    k2 = k.clone().detach()
    v2 = v.clone().detach()


    if backward:
        q1.requires_grad_(True)
        k1.requires_grad_(True)
        v1.requires_grad_(True)

        q2.requires_grad_(True)
        k2.requires_grad_(True)
        v2.requires_grad_(True)


    out_eager = fn(q1,k1,v1)

    out_eager = (
        out_eager[0]
        if isinstance(out_eager, tuple)
        else out_eager
    )


    torch._dynamo.reset()

    compiled_fn = torch.compile(
        fn,
        backend="aot_eager",
        fullgraph=True,
    )


    out_compile = compiled_fn(q2,k2,v2)

    out_compile = (
        out_compile[0]
        if isinstance(out_compile, tuple)
        else out_compile
    )


    torch.testing.assert_close(
        out_compile.float(),
        out_eager.float(),
        atol=2e-3,
        rtol=2e-3,
    )


    if not backward:
        return


    out_eager.float().sum().backward()
    out_compile.float().sum().backward()


    for a,b in [
        (q1.grad,q2.grad),
        (k1.grad,k2.grad),
        (v1.grad,v2.grad),
    ]:
        assert a is not None
        assert b is not None

        torch.testing.assert_close(
            a.float(),
            b.float(),
            atol=2e-3,
            rtol=2e-3,
        )
