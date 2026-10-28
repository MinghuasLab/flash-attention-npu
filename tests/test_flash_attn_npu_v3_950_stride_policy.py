"""CPU checks for the real 950 wrapper's cache preparation policy.

Requires the built 950 extension, but does not launch an NPU kernel.
"""

import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest
import torch
import torch_npu  # Register the npu device type before loading custom ops.


@pytest.fixture(scope="module")
def interface():
    # Avoid package __init__ dispatching to the local machine's 910 backend.
    path = Path(__file__).resolve().parents[1] / "flash_attn_npu_3/flash_attn_npu_interface_950.py"
    spec = importlib.util.spec_from_file_location("_fa3_950_stride_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", ["BBND", "BNBD"])
def test_supported_paged_views_preserve_storage(interface, layout):
    inner = (128, 2, 64) if layout == "BBND" else (2, 128, 64)
    k = torch.randn(8, *inner)[::2]
    v = torch.randn(12, *inner)[1::3]
    if layout == "BNBD":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    q = torch.randn(2, 3, 4, 64)[:, ::2]
    prepared = interface._prepare_forward_qkv(q, k, v, torch.empty(2, 2), append_kv=False)
    assert prepared[0].is_contiguous()
    assert prepared[1] is k and prepared[2] is v
    assert interface._paged_kv_inner_layout(k) == layout
    assert k.stride(0) != v.stride(0) and v.storage_offset() > 0


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("target", ["k", "v", "both"])
def test_unsupported_read_views_are_materialized(interface, paged, target):
    q = torch.randn(2, 4, 4, 64)[:, ::2]
    dense_k, dense_v = torch.randn(4, 128, 2, 64), torch.randn(4, 128, 2, 64)

    def strided(t):
        storage = torch.empty(t.shape[0], t.shape[1] * 2, *t.shape[2:])
        view = storage[:, ::2]
        view.copy_(t)
        return view

    k = strided(dense_k) if target in ("k", "both") else dense_k
    v = strided(dense_v) if target in ("v", "both") else dense_v
    q_out, k_out, v_out = interface._prepare_forward_qkv(
        q, k, v, torch.empty(2, 2) if paged else None, append_kv=False
    )
    assert all(t.is_contiguous() for t in (q_out, k_out, v_out))
    for got, expected in zip((q_out, k_out, v_out), (q, dense_k, dense_v)):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_mixed_paged_layouts_fall_back_together(interface):
    q = torch.randn(1, 4, 64)
    k = torch.randn(4, 128, 2, 64)
    v = torch.randn(4, 2, 128, 64).transpose(1, 2)
    _, k_out, v_out = interface._prepare_forward_qkv(q, k, v, torch.empty(1, 4), append_kv=False)
    assert k_out.is_contiguous() and v_out.is_contiguous()
    torch.testing.assert_close(k_out, k)
    torch.testing.assert_close(v_out, v)


@pytest.mark.parametrize("paged", [False, True])
def test_append_rejects_copying_the_update_target(interface, paged):
    q = torch.randn(1, 4, 64)
    k = torch.randn(4, 256, 2, 64)[:, ::2]
    v = torch.randn_like(k.contiguous())
    before = k.clone()
    with pytest.raises(RuntimeError, match="append-KV requires"):
        interface._prepare_forward_qkv(q, k, v, torch.empty(1, 4) if paged else None, append_kv=True)
    torch.testing.assert_close(k, before, rtol=0, atol=0)


def test_host_window_normalization_against_api_mask(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "window_host"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(root / "csrc/ascend950/flash_attn_npu_3"),
         str(Path(__file__).with_name("test_flash_attn_npu_v3_950_window_host.cpp")), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "mask decisions matched" in result.stdout


def test_empty_query_only_allows_the_declared_zero_length_path(interface):
    q = torch.empty(0, 4, 64)
    cu_q = torch.zeros(3, dtype=torch.int32)
    interface._validate_varlen_query(q, cu_q, max_seqlen_q=0)
    with pytest.raises(RuntimeError, match="total_q must be positive"):
        interface._validate_varlen_query(q, cu_q, max_seqlen_q=1)
    with pytest.raises(RuntimeError, match="total_q must be positive"):
        interface._validate_varlen_query(q, cu_q)
