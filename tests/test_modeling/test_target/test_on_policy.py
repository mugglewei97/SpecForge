"""Incremental target parity and page ownership across rejection/bonus rounds."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from specforge.inference.target_engine.dflash_target_model import HFDFlashTargetEngine
from tests.test_modeling.test_target.test_proposal_verification import make_target
from tests.test_modeling.test_target.test_sglang_proposal_cache import (
    Backend,
    session_class,  # noqa: F401 - shared pytest fixture installs the SGLang stubs
)


def _reference(engine, ids):
    ids = ids[None]
    return engine.generate_dflash_data(ids, torch.ones_like(ids), torch.ones_like(ids))


def _assert_committed(engine, session):
    for row, (ids, aux, final, _) in enumerate(session._rows):
        reference = _reference(engine, ids)
        count = len(ids) - int(session._pending[row])
        torch.testing.assert_close(
            aux[:count], reference.hidden_states[0, :count], atol=2e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            final[:count], reference.last_hidden_states[0, :count], atol=2e-6, rtol=1e-5
        )
        if session._pending[row]:
            assert aux[-1].count_nonzero() == final[-1].count_nonzero() == 0


@pytest.mark.parametrize("backend", ["hf", "sglang"])
@pytest.mark.parametrize("reserve", [False, True])
def test_committed_length_snapshots_follow_seed_and_commit(
    sglang_session_class, backend, reserve
):
    session = (
        make_target().on_policy_rollout()
        if backend == "hf"
        else sglang_session_class(SimpleNamespace(_backend=Backend(4)))
    )
    snapshots = []

    def check(expected):
        assert session.lengths_cpu == expected
        assert session.lengths.tolist() == list(expected)
        assert session.lengths.dtype == torch.long
        assert session.lengths.device == session.output.input_ids.device
        assert tuple(session.output.attention_mask.sum(-1).tolist()) == expected
        if session._cpu_tokens is not None:
            assert tuple(map(len, session._cpu_tokens)) == expected
        snapshots.append((session.lengths, session.lengths_cpu))
        for device_lengths, cpu_lengths in snapshots:
            assert tuple(device_lengths.tolist()) == cpu_lengths

    with session:
        assert session.lengths is None and session.lengths_cpu == ()
        if reserve:
            session.reserve_capacity(32)
        ids = torch.tensor([[1, 2, 3], [4, 0, 0]])
        mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
        session.capture(ids, mask, mask.float())
        check((3, 1))
        session.seed(torch.tensor([10, 9]), torch.tensor([True, False]))
        check((4, 1))
        proposals = torch.tensor([[11, 12], [13, 14]])
        for active, accepted, emit, expected in (
            # Rejection + correction in row 0; unseeded row 1 accepts without bonus.
            ([True, True], [0, 2], [True, False], (5, 3)),
            # Inactive rows ignore both counts and emit flags.
            ([True, False], [2, -999], [True, True], (8, 3)),
            ([False, False], [-999, 999], [True, True], (8, 3)),
            # Consuming a pending anchor without appending must not grow lengths.
            ([True, True], [0, 0], [False, False], (8, 3)),
        ):
            active = torch.tensor(active)
            before = session.lengths
            session.verify(proposals, torch.ones_like(proposals), active)
            assert session.lengths is before
            if active[0]:
                with pytest.raises(ValueError, match="accepted counts"):
                    session.commit(
                        torch.tensor([3, 0]), torch.tensor([15, 16]), active, active
                    )
                assert session.lengths is before
                check(tuple(before.tolist()))
            session.commit(
                torch.tensor(accepted),
                torch.tensor([15, 16]),
                active,
                torch.tensor(emit),
            )
            check(expected)
    assert session.lengths is None and session.lengths_cpu == ()
    for device_lengths, cpu_lengths in snapshots:
        assert tuple(device_lengths.tolist()) == cpu_lengths


@pytest.mark.parametrize("reserve", [False, True])
def test_hf_ragged_seed_reject_bonus_and_inactive_rows_match_full_causal_forward(
    reserve,
):
    engine = make_target()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    with (
        engine.on_policy_rollout() as session,
        patch.object(
            engine.model.model, "forward", wraps=engine.model.model.forward
        ) as forward,
        patch.object(
            engine.model.lm_head,
            "forward",
            side_effect=AssertionError("unneeded vocab projection"),
        ),
    ):
        if reserve:
            session.reserve_capacity(32)
        session.capture(ids, mask, mask.float())
        assert session._cpu_tokens is None
        storage = session.output.hidden_states.untyped_storage().data_ptr()
        session.seed(torch.tensor([10, 11]), torch.ones(2, dtype=torch.bool))
        assert forward.call_count == 1
        steps = [
            (
                torch.tensor([[12, 13, 14], [15, 16, 0]]),
                torch.tensor([[1, 1, 1], [1, 1, 0]]),
                [1, 1],
                [1, 0],
                [21, 22],
                [1, 1],
            ),
            (
                torch.tensor([[23, 24], [25, 26]]),
                torch.ones(2, 2),
                [1, 0],
                [2, 0],
                [27, 0],
                [0, 0],
            ),
            (
                torch.tensor([[28], [29]]),
                torch.ones(2, 1),
                [1, 1],
                [0, 1],
                [30, 0],
                [1, 0],
            ),
        ]
        for tokens, valid, active, accepted, next_tokens, emit in steps:
            active = torch.tensor(active, dtype=torch.bool)
            before = forward.call_count
            prefixes = [row[0].clone() for row in session._rows]
            actual = session.verify(tokens, valid, active)
            assert forward.call_count == before + 1
            for row in range(2):
                if active[row]:
                    n = int(valid[row].sum())
                    ref = _reference(
                        engine, torch.cat((prefixes[row], tokens[row, :n]))
                    )
                    offset = len(prefixes[row]) - 1
                    torch.testing.assert_close(
                        actual[row, : n + 1],
                        ref.last_hidden_states[0, offset : offset + n + 1],
                        atol=2e-6,
                        rtol=1e-5,
                    )
                else:
                    assert actual[row].count_nonzero() == 0
            before = forward.call_count
            session.commit(
                torch.tensor(accepted),
                torch.tensor(next_tokens),
                active,
                torch.tensor(emit, dtype=torch.bool),
            )
            assert forward.call_count == before, (
                "commit must never run another target forward"
            )
            _assert_committed(engine, session)
            if reserve:
                assert (
                    session.output.hidden_states.untyped_storage().data_ptr() == storage
                )
            for row, enabled in enumerate(active):
                expected = (
                    torch.cat(
                        (
                            prefixes[row],
                            tokens[row, : accepted[row]],
                            torch.tensor(next_tokens[row : row + 1])
                            if emit[row]
                            else tokens[row, :0],
                        )
                    )
                    if enabled
                    else prefixes[row]
                )
                torch.testing.assert_close(session._rows[row][0], expected)
                cached = session._cache_rows[row][0][0].size(2)
                assert cached == len(expected) - int(session._pending[row])
    assert session.closed and not session._cache_rows


def test_hf_unseeded_bonus_only_and_session_order_checks():
    engine = make_target()
    with engine.on_policy_rollout() as session:
        initial = session.capture(
            torch.tensor([[1]]), torch.ones(1, 1), torch.zeros(1, 1)
        )
        result = session.verify(
            torch.empty(1, 0, dtype=torch.long),
            torch.empty(1, 0, dtype=torch.bool),
            torch.tensor([True]),
        )
        torch.testing.assert_close(result[:, 0], initial.last_hidden_states[:, -1])
        with pytest.raises(RuntimeError, match="commit"):
            session.verify(torch.tensor([[2]]), torch.ones(1, 1), torch.tensor([True]))
        with pytest.raises(ValueError, match="accepted counts"):
            session.commit(
                torch.tensor([1]),
                torch.tensor([2]),
                torch.tensor([True]),
                torch.tensor([True]),
            )
        session.commit(
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([True]),
            torch.tensor([True]),
        )
        with pytest.raises(RuntimeError, match="seed"):
            session.seed(torch.tensor([3]), torch.tensor([True]))


def test_hf_verification_error_releases_cache():
    engine = make_target()
    with pytest.raises(RuntimeError, match="test failure"):
        with engine.on_policy_rollout() as session:
            session.capture(torch.tensor([[1, 2]]), torch.ones(1, 2), torch.ones(1, 2))
            with patch.object(
                engine.model.model, "forward", side_effect=RuntimeError("test failure")
            ):
                session.verify(
                    torch.tensor([[3]]), torch.ones(1, 1), torch.tensor([True])
                )
    assert session.closed and not session._cache_rows and session.output is None


def test_unsupported_target_fails_closed():
    engine = HFDFlashTargetEngine(
        SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
    )
    with pytest.raises(NotImplementedError, match="full-attention Qwen3"):
        engine.on_policy_rollout()


@pytest.fixture
def sglang_session_class(request):
    request.getfixturevalue("session_class")
    name = "specforge.inference.target_engine.sglang_backend.on_policy"
    try:
        yield importlib.import_module(name).SGLangOnPolicyTargetSession
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("page_size", [1, 4, 16])
@pytest.mark.parametrize("reserve", [False, True])
def test_sglang_rounds_preserve_partial_pages_and_release_rejected_suffix(
    sglang_session_class, page_size, reserve
):
    backend = Backend(page_size)
    allocator = backend.model_runner.token_to_kv_pool_allocator
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    with sglang_session_class(SimpleNamespace(_backend=backend)) as session:
        if reserve:
            session.reserve_capacity(64)
        session.capture(ids, mask, mask.float())
        assert session._cpu_tokens == [[1, 2, 3, 4, 5], [6, 7]]
        storage = session.output.hidden_states.untyped_storage().data_ptr()
        session.seed(torch.tensor([8, 9]), torch.tensor([True, True]))
        assert session._cpu_tokens == [[1, 2, 3, 4, 5, 8], [6, 7, 9]]
        for step in range(5):
            active = torch.tensor([True, step != 1])
            proposals = torch.tensor(
                [[10 + step, 11, 12, 13, 14], [15, 16 + step, 17, 18, 19]]
            )
            valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
            prefix_lengths = [len(indices) for indices in session._indices]
            prefix_ids = [row[0].clone() for row in session._rows]
            pending = list(session._pending)
            previous_extends = len(backend.extends)
            scores = session.verify(proposals, valid, active)
            expected_extends = []
            for row in range(2):
                if not active[row]:
                    assert scores[row].count_nonzero() == 0
                    continue
                n = int(valid[row].sum())
                reference = (
                    torch.cat((prefix_ids[row], proposals[row, :n])).float().cumsum(0)
                )
                offset = len(prefix_ids[row]) - 1
                torch.testing.assert_close(
                    scores[row, : n + 1, 0], reference[offset : offset + n + 1]
                )
                expected_extends.append(
                    prefix_lengths[row] % page_size + int(pending[row]) + n
                )
            assert backend.extends[previous_extends:] == expected_extends
            accepted = torch.tensor([step % 6, min(step, 3)])
            accepted[~active] = 0
            emit = torch.tensor([step != 3, step != 2]) & active
            before = len(backend.extends)
            session.commit(accepted, torch.tensor([20, 21]), active, emit)
            assert len(backend.extends) == before
            if reserve:
                assert (
                    session.output.hidden_states.untyped_storage().data_ptr() == storage
                )
            live = set()
            for row, indices in enumerate(session._indices):
                live.update((indices // page_size).tolist())
                actual_ids, aux, final, _ = session._rows[row]
                assert session._cpu_tokens[row] == actual_ids.tolist()
                count = len(actual_ids) - int(session._pending[row])
                assert len(indices) == count
                torch.testing.assert_close(
                    final[:count, 0], actual_ids[:count].float().cumsum(0)
                )
                torch.testing.assert_close(
                    aux[:count, 0], 2 * actual_ids[:count].float().cumsum(0)
                )
                assert [allocator.tokens[i] for i in indices.tolist()] == actual_ids[
                    :count
                ].tolist()
            assert allocator.live_pages == live
            assert len(backend.model_runner.req_to_token_pool.available) == 19
    assert session.closed and backend.clears == 1 and not allocator.live_pages
    assert session._cpu_tokens is None


@pytest.mark.parametrize("prefix_width", [16, 80])
@pytest.mark.parametrize("batch", [1, 4])
def test_sglang_round_transfers_only_new_tokens_and_batched_metadata(
    sglang_session_class, prefix_width, batch
):
    backend = Backend(4)
    ids = torch.arange(1, prefix_width + 1).expand(batch, -1).clone()
    lengths = prefix_width - torch.arange(batch)
    mask = torch.arange(prefix_width)[None] < lengths[:, None]
    expected = [row[:length].tolist() for row, length in zip(ids, lengths)]
    transfers = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    # Real SGLang owns the Req inputs. Ensure even an in-place mutation of that
    # list cannot corrupt the session's independently owned CPU prefix.
    original_forward = backend._forward_extend

    def consume_requests(requests):
        output = original_forward(requests)
        for request in requests:
            request.origin_input_ids.clear()
        return output

    backend._forward_extend = consume_requests
    with sglang_session_class(SimpleNamespace(_backend=backend)) as session:
        session.reserve_capacity(128)
        session.capture(ids, mask, mask.float())
        assert session._cpu_tokens == expected
        seed = torch.arange(batch) + 100
        active = torch.ones(batch, dtype=torch.bool)
        with patch.object(torch.Tensor, "cpu", record_cpu):
            session.seed(seed, active)
        assert transfers == [(batch, 2)]
        for row in range(batch):
            expected[row].append(int(seed[row]))
        for step in range(4):
            transfers.clear()
            proposals = torch.arange(3 * batch).reshape(batch, 3) + 200 + 10 * step
            valid = torch.ones_like(proposals, dtype=torch.bool)
            valid[-1, -1] = False
            active = torch.ones(batch, dtype=torch.bool)
            if batch > 1:
                active[-1] = step != 1
            if step == 3:
                active.zero_()
            accepted = torch.ones(batch, dtype=torch.long)
            accepted[~active] = 0
            emit = active & (step != 1)
            next_tokens = torch.arange(batch) + 300 + step
            proposals_before = proposals.clone()
            with patch.object(torch.Tensor, "cpu", record_cpu):
                session.verify(proposals, valid, active)
                # Both the GPU branch and CPU proposal snapshot must remain
                # independent of caller-owned sampling buffers.
                proposals.fill_(-1)
                session.commit(accepted, next_tokens, active, emit)
            assert transfers == [(batch, 6), (batch, 4)]
            for row in range(batch):
                if active[row]:
                    expected[row].extend(
                        proposals_before[row, : accepted[row]].tolist()
                    )
                    if emit[row]:
                        expected[row].append(int(next_tokens[row]))
                assert session._cpu_tokens[row] == expected[row]
                assert session._rows[row][0].tolist() == expected[row]
    assert session._cpu_tokens is None


@pytest.mark.parametrize("reserve", [False, True])
def test_sglang_cpu_tokens_preserve_large_integer_ids_and_empty_proposals(
    sglang_session_class, reserve
):
    backend = Backend(4)
    large = 2**53 + 1
    with sglang_session_class(SimpleNamespace(_backend=backend)) as session:
        if reserve:
            session.reserve_capacity(8)
        session.capture(torch.tensor([[large]]), torch.ones(1, 1), torch.ones(1, 1))
        empty = torch.empty(1, 0, dtype=torch.long)
        active = torch.tensor([True])
        # Exercise the unseeded, empty-suffix path and bonus-only commit.
        session.verify(empty, empty.bool(), active)
        session.commit(torch.tensor([0]), torch.tensor([large + 1]), active, active)
        session.verify(torch.tensor([[large + 2, large + 3]]), torch.ones(1, 2), active)
        session.commit(torch.tensor([1]), torch.tensor([large + 4]), active, active)
        expected = [large, large + 1, large + 2, large + 4]
        assert session._cpu_tokens == [expected]
        assert session._rows[0][0].tolist() == expected


def test_sglang_rejected_metadata_leaves_cpu_prefix_and_branch_retryable(
    sglang_session_class,
):
    backend = Backend(4)
    with sglang_session_class(SimpleNamespace(_backend=backend)) as session:
        session.reserve_capacity(5)
        session.capture(torch.tensor([[1, 2]]), torch.ones(1, 2), torch.ones(1, 2))
        active = torch.tensor([True])
        session.seed(torch.tensor([3]), active)
        proposals = torch.tensor([[4, 5, 6]])
        for tokens, valid, error in (
            (proposals.float(), torch.ones(1, 3), "integer token IDs"),
            (proposals, torch.tensor([[1, 0, 1]]), "contiguous prefix"),
        ):
            with pytest.raises(ValueError, match=error):
                session.verify(tokens, valid, active)
            assert session._cpu_tokens == [[1, 2, 3]]
            assert session._branches is None
        session.verify(proposals, torch.ones(1, 3), active)
        for count, enabled, error in (
            (torch.tensor([1.5]), active, "accepted counts"),
            (torch.tensor([4]), active, "accepted counts"),
            (torch.tensor([1]), ~active, "active rows"),
            (torch.tensor([2]), active, "exceed reserved"),
        ):
            with pytest.raises(ValueError, match=error):
                session.commit(count, torch.tensor([7]), enabled, active)
            assert session._cpu_tokens == [[1, 2, 3]]
            assert session._rows[0][0].tolist() == [1, 2, 3]
            assert session._branches is not None
        session.commit(torch.tensor([1]), torch.tensor([7]), active, active)
        assert session._cpu_tokens == [[1, 2, 3, 4, 7]]
        assert session._rows[0][0].tolist() == [1, 2, 3, 4, 7]


def test_reserved_capacity_overflow_does_not_partially_commit():
    engine = make_target()
    with engine.on_policy_rollout() as session:
        session.reserve_capacity(3)
        session.capture(torch.tensor([[1, 2]]), torch.ones(1, 2), torch.ones(1, 2))
        session.seed(torch.tensor([3]), torch.tensor([True]))
        session.verify(torch.tensor([[4]]), torch.ones(1, 1), torch.tensor([True]))
        with pytest.raises(ValueError, match="exceed reserved"):
            session.commit(
                torch.tensor([1]),
                torch.tensor([5]),
                torch.tensor([True]),
                torch.tensor([True]),
            )
        torch.testing.assert_close(session.output.input_ids, torch.tensor([[1, 2, 3]]))
        session.commit(
            torch.tensor([0]),
            torch.tensor([5]),
            torch.tensor([True]),
            torch.tensor([False]),
        )
        _assert_committed(engine, session)


@pytest.mark.parametrize("during_capture", [False, True])
def test_sglang_failure_after_partial_allocation_cleans_all_pools(
    sglang_session_class, during_capture
):
    backend = Backend(4)
    with pytest.raises(RuntimeError, match="allocated then failed"):
        with sglang_session_class(SimpleNamespace(_backend=backend)) as session:
            backend.fail = during_capture
            session.capture(torch.tensor([[1, 2]]), torch.ones(1, 2), torch.ones(1, 2))
            backend.fail = True
            session.verify(
                torch.tensor([[3, 4, 5]]), torch.ones(1, 3), torch.tensor([True])
            )
    assert session.closed and backend.clears == 1
    assert session._cpu_tokens is None
    assert not backend._proposal_session_active
    assert not backend.model_runner.token_to_kv_pool_allocator.live_pages
