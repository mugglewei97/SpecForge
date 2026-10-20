"""Rollout BlockMask layout, physical KV holes, and attention equivalence."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention
from torch.utils._python_dispatch import TorchDispatchMode

from specforge.core.tv_rollout_cache import BlockRolloutCache


class _NoLengthReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name in {
            "aten::_local_scalar_dense",
            "aten::sum",
            "aten::max",
            "aten::any",
        }:
            raise AssertionError(f"Rollout cache must use session lengths: {func}")
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_cache_uses_host_width_without_mask_reductions_or_scalar_reads(device):
    ids = torch.arange(16, device=device).reshape(2, 8)
    output = SimpleNamespace(
        input_ids=ids,
        hidden_states=torch.randn(2, 8, 4, device=device),
        attention_mask=None,  # Lengths come from the session, never this mask.
    )
    draft = Mock(side_effect=lambda **kwargs: kwargs["noise_embedding"])
    model = SimpleNamespace(
        block_size=7,
        attention_backend="sdpa",
        draft_model=draft,
        _create_noise_embed=lambda *_args: torch.zeros(2, 7, 4, device=device),
    )
    cache = BlockRolloutCache(proposal_start=0)
    physical_width = 0
    previous = (0, 0)
    for lengths, flags in (
        ((4, 2), (True, True)),
        ((6, 2), (True, False)),
        ((6, 2), (False, False)),  # No advancement on this rank.
        ((8, 5), (True, True)),
    ):
        device_lengths = torch.tensor(lengths, device=device)
        active = torch.tensor(flags, device=device)
        with _NoLengthReads():
            anchors, _ = cache.forward(
                model, output, active, lengths=device_lengths, lengths_cpu=lengths
            )
        expected_anchors = tuple(length - 1 for length in lengths)
        width = max(a - p for a, p in zip(expected_anchors, previous, strict=True))
        physical_width += width
        assert tuple(anchors.tolist()) == expected_anchors
        assert cache.lengths_cpu == expected_anchors
        assert tuple(cache.valid_keys.sum(-1).tolist()) == expected_anchors
        assert cache.valid_keys.shape == (2, physical_width)
        assert draft.call_args.kwargs["target_hidden"].shape == (2, width, 4)
        mask = draft.call_args.kwargs["attention_mask"]
        assert not mask[~active].any()
        previous = expected_anchors

    old_keys, old_lengths = cache.valid_keys, cache.lengths
    with _NoLengthReads(), pytest.raises(ValueError, match="cannot move backwards"):
        cache.forward(
            model,
            output,
            active,
            lengths=torch.tensor([7, 5], device=device),
            lengths_cpu=(7, 5),
        )
    assert cache.valid_keys is old_keys and cache.lengths is old_lengths
    assert draft.call_count == 4


def _keys(width, device="cpu"):
    keys = torch.ones(4, width, dtype=torch.bool, device=device)
    keys[1, 2::3] = False
    keys[1, 128:256] = False  # A fully empty tile between visible tokens.
    keys[2, :128] = False  # A leading empty tile, followed by visible tokens.
    keys[3] = False  # An inactive row must have zero attention output.
    return keys


def _materialize_mask(mask):
    """Interpret both sparse tile metadata and mask_mod, including padded edges."""
    batch, heads, query_blocks, kv_blocks = mask.kv_indices.shape
    query_tile, kv_tile = mask.BLOCK_SIZE
    result = torch.zeros(
        batch, heads, query_blocks * query_tile, kv_blocks * kv_tile, dtype=torch.bool
    )
    for counts, indices, full in (
        (mask.kv_num_blocks, mask.kv_indices, False),
        (mask.full_kv_num_blocks, mask.full_kv_indices, True),
    ):
        if counts is None:
            continue
        for b in range(batch):
            for h in range(heads):
                for q_block in range(query_blocks):
                    q = torch.arange(q_block * query_tile, (q_block + 1) * query_tile)
                    for k_block in indices[b, h, q_block, : counts[b, h, q_block]]:
                        k = torch.arange(k_block * kv_tile, (k_block + 1) * kv_tile)
                        visible = (
                            True
                            if full
                            else mask.mask_mod(torch.tensor(b), h, q[:, None], k[None])
                        )
                        result[b, h, q[:, None], k[None]] = visible
    return result


@pytest.mark.parametrize("backend", ["TRITON", "FLASH"])
@pytest.mark.parametrize("query_length", [7, 8])
@pytest.mark.parametrize("key_width", [8, 127, 128, 129, 256, 263])
def test_block_mask_preserves_physical_holes_inactive_rows_and_tile_edges(
    monkeypatch, backend, query_length, key_width
):
    monkeypatch.setenv("SPECFORGE_FLEX_ATTENTION_BACKEND", backend)
    keys = _keys(key_width)
    mask = BlockRolloutCache()._block_mask(keys, query_length)
    actual = _materialize_mask(mask)
    expected = torch.zeros_like(actual)
    expected[:, 0, :query_length, :key_width] = keys[:, None]
    torch.testing.assert_close(actual, expected)
    assert mask.seq_lengths == (query_length, key_width)
    assert mask.BLOCK_SIZE == ((256, 128) if backend == "FLASH" else (128, 128))
    assert mask.q_num_blocks is None and mask.q_indices is None
    # FLASH needs the full-block fields to exist, even when all tiles are partial.
    assert mask.full_kv_indices is not None
    assert torch.count_nonzero(mask.full_kv_num_blocks) == 0


def test_layout_reuse_refreshes_visibility_and_sequence_lengths(monkeypatch):
    monkeypatch.setenv("SPECFORGE_FLEX_ATTENTION_BACKEND", "TRITON")
    cache = BlockRolloutCache()
    previous = None
    previous_dense = None
    for step, width in enumerate((120, 125, 128, 129, 133, 257)):
        keys = _keys(width)
        keys[step % 4] = False
        mask = cache._block_mask(keys, 7)
        if previous is not None:
            same_bucket = (width - 1) // 128 == (previous.seq_lengths[1] - 1) // 128
            assert (mask.kv_num_blocks is previous.kv_num_blocks) == same_bucket
            assert (mask.kv_indices is previous.kv_indices) == same_bucket
        expected = keys[:, None, None].expand(-1, 1, 7, -1)
        torch.testing.assert_close(_materialize_mask(mask)[:, :, :7, :width], expected)
        # An old mask must still describe its original round after cache reuse.
        if previous is not None:
            torch.testing.assert_close(_materialize_mask(previous), previous_dense)
        assert mask.seq_lengths == (7, width)
        previous, previous_dense = mask, _materialize_mask(mask)


@pytest.mark.parametrize("batch", [1, 4, 16])
@pytest.mark.parametrize("key_width", [8, 129, 4096])
def test_block_indices_support_triton_batch_offsets(monkeypatch, batch, key_width):
    monkeypatch.setenv("SPECFORGE_FLEX_ATTENTION_BACKEND", "TRITON")
    mask = BlockRolloutCache()._block_mask(
        torch.ones(batch, key_width, dtype=torch.bool), 7
    )
    for indices in (mask.kv_indices, mask.full_kv_indices):
        # The Triton forward flattens batch/head using stride(1), assuming
        # contiguous metadata; it does not use the tensor's batch stride.
        # FULL_KV_IDX is also read once when full_kv_num_blocks is zero.
        storage_size = indices.untyped_storage().nbytes() // indices.element_size()
        for row in range(batch):
            offset = indices.storage_offset() + row * indices.size(1) * indices.stride(
                1
            )
            assert offset < storage_size, (
                f"Kernel index load for batch row {row} is outside storage: "
                f"offset={offset}, storage_size={storage_size}, stride={indices.stride()}"
            )
        assert indices.is_contiguous()
    assert mask.kv_num_blocks.is_contiguous()
    assert mask.full_kv_num_blocks.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
@pytest.mark.parametrize("backend", ["TRITON", "FLASH"])
@pytest.mark.parametrize("query_length", [7, 8])
@torch.no_grad()
def test_cuda_rollout_attention_matches_sdpa(monkeypatch, backend, query_length):
    if backend == "FLASH":
        pytest.importorskip("flash_attn.cute")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("FLASH test requires Hopper or newer")
    monkeypatch.setenv("SPECFORGE_FLEX_ATTENTION_BACKEND", backend)
    torch.manual_seed(42)
    cache = BlockRolloutCache()
    compiled = torch.compile(flex_attention, fullgraph=True)
    q = torch.randn(4, 4, query_length, 64, device="cuda", dtype=torch.bfloat16)
    for width in (128, 129, 263, 270):
        keys = _keys(width, "cuda")
        k = torch.randn(4, 2, width, 64, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        actual = compiled(
            q,
            k,
            v,
            block_mask=cache._block_mask(keys, query_length),
            enable_gqa=True,
            kernel_options={"BACKEND": backend},
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=keys[:, None, None, :], enable_gqa=True
        )
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual[3]) == 0
