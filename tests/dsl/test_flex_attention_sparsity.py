# Adapted from CATLASS DSL tests.
"""Direct sparse bindings, native lowering, and opt-in device regressions."""

from contextlib import contextmanager
from importlib import import_module
import os
from types import SimpleNamespace

from importlib.util import find_spec

import pytest

# Only an absent optional runtime is skipped; a broken installed runtime fails.
if find_spec("catlass") is None:
    pytest.skip("CATLASS DSL runtime is not installed", allow_module_level=True)

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor
from flash_attn_npu_dsl import block_sparsity, interface, simd

if find_spec("torch") is None:
    pytest.skip("PyTorch is not installed", allow_module_level=True)
import torch

classifier = import_module("flash_attn_npu_dsl.compute_block_sparsity")


def right_causal_mask(b, h, q, kv, info, tensors):
    return kv <= q + info.seqlen_k - info.seqlen_q


@simd
def simd_causal_mask(b, h, q, kv, info, tensors):
    return tla.cmp(kv, q + info.seqlen_k - info.seqlen_q, "le")


@pytest.fixture
def shape(monkeypatch):
    monkeypatch.setattr(block_sparsity, "_require_npu", lambda device: torch.device("cpu"))
    monkeypatch.setattr(classifier, "_require_npu", lambda device: torch.device("cpu"))
    return dict(batch_size=2, num_heads_q=3, seqlen_q=129, seqlen_k=513, device="cpu")


def _blocks(partial=((3, 0), ()), full=((4, 1), (2,))):
    def pair(rows):
        counts = torch.tensor([[list(map(len, rows))]], dtype=torch.int32)
        indices = torch.full((1, 1, 2, 2), -99, dtype=torch.int32)
        for row, values in enumerate(rows):
            indices[0, 0, row, : len(values)] = torch.tensor(values, dtype=torch.int32)
        return counts, indices

    return block_sparsity.BlockSparseTensorsTorch(
        *pair(partial), *pair(full), block_size=(128, 128)
    )


def test_metadata_retains_original_unsorted_buffers_and_independent_broadcasts(shape):
    blocks = _blocks()
    blocks = blocks._replace(
        mask_block_cnt=blocks.mask_block_cnt.expand(2, 1, 2).contiguous(),
        mask_block_idx=blocks.mask_block_idx.expand(1, 3, 2, 2).contiguous(),
    )
    sparse = block_sparsity.validate_block_sparse(blocks, **shape)
    assert all(actual is expected for actual, expected in zip(sparse.tensors, blocks[:4]))
    assert sparse.strides == ((2, 0, 1), (0, 4, 2), (0, 0, 1), (0, 0, 2))
    assert blocks.mask_block_idx[0, 2].tolist() == [[3, 0], [-99, -99]]


@pytest.mark.parametrize("full_present", [False, True])
def test_zero_capacity_and_optional_full_list_keep_counts_without_dummy_indices(
    shape, full_present
):
    count = torch.zeros((1, 1, 2), dtype=torch.int32)
    index = torch.empty((1, 1, 2, 0), dtype=torch.int32)
    blocks = block_sparsity.BlockSparseTensorsTorch(
        count,
        index,
        count.clone() if full_present else None,
        index.clone() if full_present else None,
        block_size=(128, 128),
    )
    sparse = block_sparsity.validate_block_sparse(blocks, **shape)
    assert sparse.tensors[0] is count and sparse.tensors[1] is None
    assert sparse.tensors[2] is blocks.full_block_cnt and sparse.tensors[3] is None
    assert sparse.strides == (
        (0, 0, 1),
        (0, 0, 0),
        (0, 0, 1) if full_present else None,
        (0, 0, 0) if full_present else None,
    )


@pytest.mark.parametrize(
    "partial_capacity,full_capacity",
    [(0, None), (0, 0), (0, 5), (5, None), (5, 0), (1, 3), (3, 1), (5, 5)],
)
def test_independent_compact_capacities_are_not_expanded_or_padded(
    shape, partial_capacity, full_capacity
):
    def pair(rows, capacity):
        counts = torch.tensor([[list(map(len, rows))]], dtype=torch.int32)
        indices = torch.full((1, 1, 2, capacity), -99, dtype=torch.int32)
        for index, values in enumerate(rows):
            indices[0, 0, index, : len(values)] = torch.tensor(values, dtype=torch.int32)
        return counts, indices

    # These are caller-provided unsorted lists, not classifier-specific data.
    partial_rows = ([4, 1][:partial_capacity], [2][:partial_capacity])
    partial_count, partial_index = pair(partial_rows, partial_capacity)
    full_rows = ([3, 0][:full_capacity], []) if full_capacity is not None else ([], [])
    full_count, full_index = (
        pair(full_rows, full_capacity) if full_capacity is not None else (None, None)
    )
    blocks = block_sparsity.BlockSparseTensorsTorch(
        partial_count.expand(2, 1, 2).contiguous(),
        partial_index.expand(1, 3, 2, partial_capacity).contiguous(),
        None if full_count is None else full_count.expand(1, 3, 2).contiguous(),
        None if full_index is None else full_index.expand(2, 1, 2, full_capacity).contiguous(),
        block_size=(128, 128),
    )
    sparse = block_sparsity.validate_block_sparse(blocks, **shape)
    for actual, original in zip(sparse.tensors, blocks[:4]):
        expected = None if original is None or original.numel() == 0 else original
        assert actual is expected
    assert sparse.strides[:2] == (
        (2, 0, 1),
        (0, 2 * partial_capacity, partial_capacity),
    )
    assert sparse.strides[2:] == (
        ((0, 2, 1), (2 * full_capacity, 0, full_capacity))
        if full_capacity is not None
        else (None, None)
    )


@pytest.mark.parametrize(
    "field,value,error_type,error",
    [
        ("mask_block_cnt", torch.zeros(1, 1, 2), TypeError, "int32"),
        ("mask_block_idx", torch.zeros(1, 1, 2, 2), TypeError, "int32"),
        ("full_block_cnt", torch.zeros(1, 1, 2), TypeError, "int32"),
        ("full_block_idx", torch.zeros(1, 1, 2, 2), TypeError, "int32"),
        ("mask_block_cnt", torch.zeros(2, dtype=torch.int32), ValueError, "shape"),
        (
            "mask_block_idx",
            torch.zeros(1, 2, 2, dtype=torch.int32),
            ValueError,
            "shape",
        ),
        (
            "mask_block_cnt",
            torch.zeros(3, 1, 2, dtype=torch.int32),
            ValueError,
            "extents",
        ),
        (
            "mask_block_idx",
            torch.zeros(1, 2, 2, 2, dtype=torch.int32),
            ValueError,
            "extents",
        ),
        (
            "full_block_cnt",
            torch.zeros(1, 1, 3, dtype=torch.int32),
            ValueError,
            "extents",
        ),
        (
            "full_block_idx",
            torch.zeros(1, 1, 2, 6, dtype=torch.int32),
            ValueError,
            "capacity",
        ),
        (
            "mask_block_cnt",
            torch.zeros(1, 1, 4, dtype=torch.int32)[..., ::2],
            ValueError,
            "contiguous",
        ),
        (
            "full_block_idx",
            torch.zeros(1, 1, 2, 4, dtype=torch.int32)[..., ::2],
            ValueError,
            "contiguous",
        ),
        ("full_block_idx", None, ValueError, "both be present"),
        ("full_block_cnt", None, ValueError, "both be present"),
    ],
)
def test_structural_metadata_errors_are_rejected_without_reading_contents(
    shape, field, value, error_type, error
):
    with pytest.raises(error_type, match=error):
        block_sparsity.validate_block_sparse(_blocks()._replace(**{field: value}), **shape)


def test_metadata_device_and_block_size_validation(shape):
    blocks = _blocks()
    with pytest.raises(ValueError, match="must be on"):
        block_sparsity.validate_block_sparse(
            blocks._replace(mask_block_cnt=blocks.mask_block_cnt.to("meta")), **shape
        )
    with pytest.raises(NotImplementedError, match="128, 128"):
        block_sparsity.validate_block_sparse(blocks._replace(block_size=(64, 128)), **shape)


@pytest.fixture
def sparse_host(monkeypatch, shape):
    state = SimpleNamespace(compiled=[], launched=[])

    @contextmanager
    def guard(device):
        yield

    def compile(kernel, *args, **kwargs):
        state.compiled.append(args)
        return lambda *args, **kwargs: state.launched.append(args)

    monkeypatch.setattr(interface, "_validate_inputs", lambda *args: None)
    monkeypatch.setattr(interface, "from_dlpack", lambda tensor, *, layout_tag: tensor)
    monkeypatch.setattr(interface.tla, "compile", compile)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device=guard,
            get_device_properties=lambda index: SimpleNamespace(cube_core_num=2),
        ),
        raising=False,
    )
    monkeypatch.setenv("CATLASS_DSL_CACHE", "1")
    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "0")
    state.inputs = (
        torch.empty(2, 129, 3, 64, dtype=torch.float16),
        torch.empty(2, 513, 1, 64, dtype=torch.float16),
        torch.empty(2, 513, 1, 64, dtype=torch.float16),
    )
    interface._compiled_kernels.clear()
    yield state
    interface._compiled_kernels.clear()


@pytest.mark.parametrize(
    "empty,full_present", [(False, True), (False, False), (True, True), (True, False)]
)
def test_public_sparse_call_has_no_device_preprocessing(
    sparse_host, monkeypatch, empty, full_present
):
    from torch.utils._python_dispatch import TorchDispatchMode

    blocks = _blocks()
    if empty:
        blocks = blocks._replace(
            mask_block_cnt=torch.zeros_like(blocks.mask_block_cnt),
            mask_block_idx=torch.empty(1, 1, 2, 0, dtype=torch.int32),
            full_block_cnt=torch.zeros_like(blocks.full_block_cnt),
            full_block_idx=torch.empty(1, 1, 2, 0, dtype=torch.int32),
        )
    if not full_present:
        blocks = blocks._replace(full_block_cnt=None, full_block_idx=None)
    operations = []

    class RecordOperations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operations.append(str(func))
            return func(*args, **(kwargs or {}))

    def no_host_read(*args, **kwargs):
        pytest.fail("sparse metadata must not be read back to the host")

    with monkeypatch.context() as context:
        for name in ("cpu", "numpy", "item", "tolist"):
            context.setattr(torch.Tensor, name, no_host_read)
        with RecordOperations():
            interface.flash_attn_func(
                *sparse_host.inputs, block_sparse_tensors=blocks, return_lse=True
            )
    # The only Torch operations are output allocation and zero-copy flattening
    # for DLPack. In particular: no range, validation reduction, sort, merge,
    # count addition, expansion, clone, transfer, or sparse scratch allocation.
    assert set(operations) <= {
        "aten.empty_like.default",
        "aten.empty.memory_format",
        "aten.view.default",
    }
    assert operations.count("aten.empty_like.default") == 1
    assert operations.count("aten.empty.memory_format") == 1
    bound = sparse_host.launched[0][5]
    assert len(bound) == 4
    for actual, original in zip(bound, blocks[:4]):
        if original is None or original.numel() == 0:
            assert actual is None
        else:
            assert actual.untyped_storage()._cdata == original.untyped_storage()._cdata
            assert actual.data_ptr() == original.data_ptr()


def test_public_sparse_reuses_compilation_after_mutation_and_address_rebinding(
    sparse_host,
):
    blocks = _blocks()
    interface.flash_attn_func(*sparse_host.inputs, block_sparse_tensors=blocks)
    blocks.mask_block_cnt[0, 0] = torch.tensor([1, 1], dtype=torch.int32)
    blocks.mask_block_idx[0, 0, :, 0] = torch.tensor([3, 4], dtype=torch.int32)
    interface.flash_attn_func(*sparse_host.inputs, block_sparse_tensors=blocks)
    rebound = blocks._replace(
        **{
            name: getattr(blocks, name).clone()
            for name in (
                "mask_block_cnt",
                "mask_block_idx",
                "full_block_cnt",
                "full_block_idx",
            )
        }
    )
    interface.flash_attn_func(*sparse_host.inputs, block_sparse_tensors=rebound)
    assert len(sparse_host.compiled) == 1 and len(sparse_host.launched) == 3
    first, second, third = (args[5] for args in sparse_host.launched)
    for before, updated, moved, original in zip(first, second, third, rebound[:4]):
        assert before.data_ptr() == updated.data_ptr() != moved.data_ptr()
        assert moved.data_ptr() == original.data_ptr()
        assert torch.equal(updated, moved)


def test_public_sparse_specializes_each_broadcast_capacity_and_optional_full_pair(
    sparse_host,
):
    blocks = _blocks()
    variants = [blocks]
    for name in (
        "mask_block_cnt",
        "mask_block_idx",
        "full_block_cnt",
        "full_block_idx",
    ):
        value = getattr(blocks, name)
        for axis, size in ((0, 2), (1, 3)):
            extents = list(value.shape)
            extents[axis] = size
            variants.append(blocks._replace(**{name: value.expand(extents).contiguous()}))
    for name in ("mask_block_idx", "full_block_idx"):
        value = getattr(blocks, name)
        padded = torch.full((*value.shape[:-1], 3), -99, dtype=torch.int32)
        padded[..., :2] = value
        variants.append(blocks._replace(**{name: padded}))
    variants.extend(
        [
            blocks._replace(full_block_cnt=None, full_block_idx=None),
            blocks._replace(
                full_block_cnt=torch.zeros_like(blocks.full_block_cnt),
                full_block_idx=torch.empty(1, 1, 2, 0, dtype=torch.int32),
            ),
        ]
    )
    for expected, variant in enumerate(variants, start=1):
        interface.flash_attn_func(*sparse_host.inputs, block_sparse_tensors=variant)
        assert len(sparse_host.compiled) == expected


@pytest.mark.parametrize(
    "callback,kernel",
    [
        (right_causal_mask, classifier.flex_attention_classify_blocks_kernel),
        (simd_causal_mask, classifier.flex_attention_classify_blocks_simd_kernel),
    ],
)
def test_public_builder_selects_mode_and_direct_compile(monkeypatch, shape, callback, kernel):
    events = []

    @contextmanager
    def guard(device):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def compile(actual_kernel, key, compile_args):
        assert actual_kernel is kernel
        args = compile_args()
        assert len(args) == 12
        assert args[4] is callback and args[-2] == [] and args[-1] is None
        events.append("compile")

        def launch(*args, **kwargs):
            assert len(args) == 6
            assert args[-2] == [] and args[-1] is None
            assert kwargs == {"block_num": 12}
            events.append("launch")

        return launch

    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=guard), raising=False)
    monkeypatch.setattr(classifier, "from_dlpack", lambda tensor, *, layout_tag: tensor)
    monkeypatch.setattr(classifier, "_compile_kernel", compile)
    blocks = classifier.compute_block_sparsity(
        128, 128, 2, 3, 129, 513, callback, [], "cpu", aux_scalars=()
    )
    assert isinstance(blocks, block_sparsity.BlockSparseTensorsTorch)
    assert blocks.mask_block_cnt.shape == (2, 3, 2)
    assert blocks.mask_block_idx.shape == (2, 3, 2, 5)
    assert events == ["enter", "compile", "launch", "exit"]


@pytest.mark.parametrize(
    "callback,kernel,mode",
    [
        (right_causal_mask, classifier.flex_attention_classify_blocks_kernel, "simt"),
        (
            simd_causal_mask,
            classifier.flex_attention_classify_blocks_simd_kernel,
            "simd",
        ),
    ],
)
def test_classifier_native_lowering(callback, kernel, mode):
    count = make_fake_tensor(tla.Int32, (4,), (1,), layout_tag=tla.arch.RowMajor)
    index = make_fake_tensor(tla.Int32, (4, 3), (3, 1), layout_tag=tla.arch.RowMajor)
    text = kernel.dump_mlir(
        type_args=(
            count,
            index,
            count,
            index,
            callback,
            2,
            2,
            3,
            129,
            257,
            None,
            None,
        )
    )
    assert "tla.func" in text and mode in text


def _direct_sparse_regression_blocks():
    # Partial then full traversal has 0/1/2/3/4 ordinals; the one-element
    # physical KV tail (block 4) appears at ordinals 0/1/2/3. Transitions at
    # ordinals 1/2/3 exercise delayed PV and the three-slot pipeline ring.
    partial = ((), (4,), (1,), (4, 0), (2, 0, 3), (0, 4), (), (), (0,))
    full = ((), (), (4,), (2,), (4,), (1, 3), (4, 2, 1, 3), (), (3, 4))

    def pair(rows):
        counts = torch.tensor([[list(map(len, rows))]], dtype=torch.int32)
        indices = torch.full((1, 1, len(rows), max(map(len, rows))), -99, dtype=torch.int32)
        for row, values in enumerate(rows):
            indices[0, 0, row, : len(values)] = torch.tensor(values, dtype=torch.int32)
        return counts, indices

    pc, pi = pair(partial)
    fc, fi = pair(full)
    # Every field broadcasts a different axis. Vary both active prefixes and
    # list order so that incorrect row addressing changes the expected output.
    pc, pi = pc.expand(2, 1, 9).contiguous(), pi.expand(1, 2, 9, 3).contiguous()
    fc, fi = fc.expand(1, 2, 9).contiguous(), fi.expand(2, 1, 9, 4).contiguous()
    pc[1] = (pc[1] - 1).clamp_min(0)
    fc[:, 1] = (fc[:, 1] - 1).clamp_min(0)
    for row, values in enumerate(partial):
        pi[0, 1, row, : len(values)] = torch.tensor(values[::-1], dtype=torch.int32)
    for row, values in enumerate(full):
        fi[1, 0, row, : len(values)] = torch.tensor(values[::-1], dtype=torch.int32)
    return block_sparsity.BlockSparseTensorsTorch(pc, pi, fc, fi, block_size=(128, 128))


def _regression_visibility(blocks):
    visible = torch.zeros(2, 2, 1041, 513, dtype=torch.bool)

    def row(tensor, batch, head, query):
        return tensor[batch % tensor.shape[0], head % tensor.shape[1], query]

    for batch in range(2):
        for head in range(2):
            for query in range(9):
                seen = set()
                for kind, (counts, indices) in enumerate((blocks[:2], blocks[2:4])):
                    count = int(row(counts, batch, head, query))
                    assert 0 <= count <= indices.shape[-1]
                    for block in row(indices, batch, head, query)[:count].tolist():
                        assert 0 <= block < 5 and block not in seen
                        seen.add(block)
                        # Only block 0 is genuinely partial; conservative
                        # partial entries elsewhere are also safe. Full blocks
                        # are always genuinely fully visible to the callback.
                        assert not kind or block > 0
                        start, stop = max(block * 128, 64), min((block + 1) * 128, 513)
                        visible[batch, head, query * 128 : (query + 1) * 128, start:stop] = True
    return visible


def _regression_mask(b, h, q, kv, info, tensors):
    return kv >= 64


@simd
def _regression_vector_mask(b, h, q, kv, info, tensors):
    return tla.cmp(kv, 64, "ge")


def _regression_score(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return value * 0.75


@simd
def _regression_vector_score(value, b, h, *, q_idx, kv_idx, seqlen_info, aux_tensors):
    return value * 0.75


def test_device_regression_metadata_contains_only_safe_active_prefixes(shape):
    blocks = _direct_sparse_regression_blocks()
    sparse = block_sparsity.validate_block_sparse(
        blocks, **(shape | dict(num_heads_q=2, seqlen_q=1041))
    )
    assert sparse.strides == ((9, 0, 1), (0, 27, 3), (0, 9, 1), (36, 0, 4))
    visible = _regression_visibility(blocks)
    assert not visible[:, :, :128].any()
    assert not visible[:, :, 7 * 128 : 8 * 128].any()
    assert visible[:, :, 8 * 128 :].any()


@pytest.mark.skipif(
    not os.getenv("FLASH_ATTN_DSL_TEST_DEVICE"),
    reason="set FLASH_ATTN_DSL_TEST_DEVICE to opt into NPU execution",
)
@pytest.mark.parametrize(
    "dtype,dim,return_lse,mask,score",
    [
        (torch.float16, 64, True, _regression_mask, _regression_vector_score),
        (torch.bfloat16, 96, False, _regression_vector_mask, _regression_score),
        (torch.bfloat16, 128, True, _regression_vector_mask, _regression_vector_score),
    ],
)
def test_direct_sparse_device_ordinals_tail_broadcast_and_rebinding(
    monkeypatch, dtype, dim, return_lse, mask, score
):
    import torch_npu  # noqa: F401

    device = f"npu:{int(os.environ['FLASH_ATTN_DSL_TEST_DEVICE'])}"
    torch.npu.set_device(device)
    # A single persistent core guarantees active -> empty -> nonempty task
    # transitions, irrespective of the device's available core count.
    original_compile = interface._compile_kernel

    def single_core_compile(*args):
        compiled = original_compile(*args)
        return lambda *args, **kwargs: compiled(*args, **(kwargs | {"block_num": 1}))

    monkeypatch.setattr(interface, "_compile_kernel", single_core_compile)
    monkeypatch.setenv("CATLASS_DSL_CACHE", "1")
    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "0")
    generator = torch.Generator().manual_seed(17 + dim)
    q = torch.randn(2, 1041, 2, dim, generator=generator, dtype=dtype).to(device)
    k = torch.randn(2, 513, 1, dim, generator=generator, dtype=dtype).to(device)
    v = torch.randn(2, 513, 1, dim, generator=generator, dtype=dtype).to(device)
    cpu_blocks = _direct_sparse_regression_blocks()
    blocks = block_sparsity.BlockSparseTensorsTorch(
        *(tensor.to(device) for tensor in cpu_blocks[:4]), block_size=(128, 128)
    )

    q_ref, k_ref, v_ref = (tensor.detach().float().cpu() for tensor in (q, k, v))
    k_ref, v_ref = (tensor.repeat_interleave(2, dim=2) for tensor in (k_ref, v_ref))
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref, k_ref) / dim**0.5 * 0.75

    interface._compiled_kernels.clear()
    try:
        for iteration in range(3):
            if iteration == 1:
                cpu_blocks = block_sparsity.BlockSparseTensorsTorch(
                    *(tensor.roll(1, dims=2) for tensor in cpu_blocks[:4]),
                    block_size=(128, 128),
                )
                for target, source in zip(blocks[:4], cpu_blocks[:4]):
                    target.copy_(source)
            elif iteration == 2:
                blocks = block_sparsity.BlockSparseTensorsTorch(
                    *(tensor.clone() for tensor in blocks[:4]), block_size=(128, 128)
                )
            masked = scores.masked_fill(~_regression_visibility(cpu_blocks), -torch.inf)
            expected_lse = torch.logsumexp(masked, dim=-1)
            empty = torch.isneginf(expected_lse)
            probability = torch.exp(masked - torch.where(empty, 0, expected_lse)[..., None])
            expected_out = torch.einsum("bhqk,bkhd->bqhd", probability, v_ref)
            out, lse = interface.flash_attn_func(
                q,
                k,
                v,
                block_sparse_tensors=blocks,
                mask_mod=mask,
                score_mod=score,
                return_lse=return_lse,
            )
            torch.npu.synchronize()
            assert (lse is not None) == return_lse
            actual_out = out.detach().float().cpu()
            assert torch.isfinite(actual_out).all()
            torch.testing.assert_close(actual_out, expected_out, atol=0.05, rtol=0)
            assert (actual_out.permute(0, 2, 1, 3)[empty] == 0).all()
            if lse is not None:
                actual_lse = lse.detach().float().cpu()
                assert torch.equal(torch.isneginf(actual_lse), empty)
                torch.testing.assert_close(actual_lse, expected_lse, atol=0.05, rtol=0)
            assert len(interface._compiled_kernels) == 1
    finally:
        interface._compiled_kernels.clear()
