"""CPU allocator contract tests for SGLang's session-owned prefix pages.

The GPU runner is replaced with a causal token-sum oracle; production session
code runs unchanged against int32 pool indices and paged allocation/free.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def session_class(monkeypatch):
    # Keep GPU SGLang imports out of CPU tests, without importing the backend's
    # eager __init__. Only the two external request-construction types are mocked.
    package = "specforge.inference.target_engine.sglang_backend"
    stub_package = ModuleType(package)
    stub_package.__path__ = [
        str(
            Path(__file__).resolve().parents[3]
            / "specforge/inference/target_engine/sglang_backend"
        )
    ]
    monkeypatch.setitem(sys.modules, package, stub_package)

    class Req:
        def __init__(self, *, origin_input_ids, **kwargs):
            self.origin_input_ids = origin_input_ids
            self.prefix_indices = torch.empty(0, dtype=torch.int64)
            self.req_pool_idx = None

        def set_extend_range(self, start, end):
            self.extend_range = SimpleNamespace(start=start, end=end)

    schedule = ModuleType("sglang.srt.managers.schedule_batch")
    schedule.Req = Req
    sampling = ModuleType("sglang.srt.sampling.sampling_params")
    sampling.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, schedule.__name__, schedule)
    monkeypatch.setitem(sys.modules, sampling.__name__, sampling)
    name = package + ".proposal_cache"
    try:
        yield importlib.import_module(name).SGLangProposalVerificationSession
    finally:
        sys.modules.pop(name, None)


class Pool:
    def __init__(self):
        self.req_to_token = torch.zeros(20, 128, dtype=torch.int32)
        self.available = list(range(1, 20))

    def alloc(self, request):
        request.req_pool_idx = self.available.pop()

    def free(self, request):
        assert request.req_pool_idx not in self.available
        self.available.append(request.req_pool_idx)
        request.req_pool_idx = None

    def clear(self):
        self.available = list(range(1, 20))


class Allocator:
    def __init__(self, page_size):
        self.page_size = page_size
        self.next_page = 1
        self.live_pages = set()
        self.freed_pages = []
        self.tokens = {}

    def alloc(self, tokens):
        pages = (len(tokens) + self.page_size - 1) // self.page_size
        start = self.next_page * self.page_size
        self.live_pages.update(range(self.next_page, self.next_page + pages))
        self.next_page += pages
        indices = torch.arange(start, start + len(tokens), dtype=torch.int64)
        self.tokens.update(zip(indices.tolist(), tokens))
        return indices

    def free(self, indices):
        pages = set((indices.long() // self.page_size).tolist())
        assert pages <= self.live_pages, "double-free or unowned page"
        self.live_pages -= pages
        self.freed_pages.extend(pages)

    def clear(self):
        self.live_pages.clear()
        self.tokens.clear()


class Backend:
    def __init__(self, page_size):
        self.model_runner = SimpleNamespace(
            req_to_token_pool=Pool(),
            token_to_kv_pool_allocator=Allocator(page_size),
            server_args=SimpleNamespace(page_size=page_size),
        )
        self.extends = []
        self.fail = False
        self.clears = 0

    def _set_eagle3_logits_flags(self, last, logits, shard):
        assert (last, logits, shard) == (True, False, False)

    def _clear_pools(self):
        self.clears += 1
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    def _forward_extend(self, requests):
        states = []
        pool = self.model_runner.req_to_token_pool
        allocator = self.model_runner.token_to_kv_pool_allocator
        for req in requests:
            shared = len(req.prefix_indices)
            assert req.prefix_indices.dtype == torch.int64
            assert shared % allocator.page_size == 0
            assert req.extend_range.start == shared
            assert req.extend_input_len == len(req.origin_input_ids) - shared
            assert [
                allocator.tokens[i] for i in req.prefix_indices.tolist()
            ] == req.origin_input_ids[:shared]
            pool.alloc(req)
            new = allocator.alloc(req.origin_input_ids[shared:])
            indices = torch.cat((req.prefix_indices, new))
            pool.req_to_token[req.req_pool_idx, : len(indices)] = indices.to(
                torch.int32
            )
            self.extends.append(req.extend_input_len)
            if self.fail:
                raise RuntimeError("allocated then failed")
            state = torch.tensor(req.origin_input_ids).float().cumsum(0)[shared:, None]
            states.append(state)
        final = torch.cat(states)
        return SimpleNamespace(last_hidden_states=final, aux_hidden_states=final * 2)


@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_shared_full_pages_private_partial_pages_and_release(session_class, page_size):
    backend = Backend(page_size)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 0, 0, 0, 0]])
    mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4])
    with session_class(SimpleNamespace(_backend=backend)) as session:
        captured = session.capture(ids, mask, mask.float())
        allocator = backend.model_runner.token_to_kv_pool_allocator
        original_pages = allocator.live_pages.copy()
        assert session.default_batch_size == 64
        torch.testing.assert_close(
            captured.last_hidden_states[1, :5, 0], ids[1, :5].float().cumsum(0)
        )
        assert captured.last_hidden_states[1, 5:].count_nonzero() == 0
        for _ in range(2):
            rows, prefixes = [0, 0, 1], [5, 8, 3]
            suffixes = [
                torch.tensor([11, 12]),
                torch.tensor([13]),
                torch.tensor([14, 15, 16]),
            ]
            results = session.score_suffixes(rows, prefixes, suffixes)
            for row, prefix, suffix, result in zip(rows, prefixes, suffixes, results):
                expected = (
                    torch.cat((ids[row, :prefix], suffix))
                    .float()
                    .cumsum(0)[prefix:, None]
                )
                torch.testing.assert_close(result, expected)
            assert backend.extends[-3:] == [
                prefix % page_size + len(suffix)
                for prefix, suffix in zip(prefixes, suffixes)
            ]
            assert allocator.live_pages == original_pages
            assert not set(allocator.freed_pages) & original_pages
            assert len(backend.model_runner.req_to_token_pool.available) == 19
        with pytest.raises(RuntimeError, match="overlap"):
            with session_class(SimpleNamespace(_backend=backend)):
                pass
    assert session.closed and backend.clears == 1
    assert not allocator.live_pages
    assert not backend._proposal_session_active


@pytest.mark.parametrize("fail_capture", [False, True])
def test_partial_allocation_failure_releases_entire_session(
    session_class, fail_capture
):
    backend = Backend(4)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with pytest.raises(RuntimeError, match="allocated then failed"):
        with session_class(SimpleNamespace(_backend=backend)) as session:
            backend.fail = fail_capture
            session.capture(ids, torch.ones_like(ids), torch.ones_like(ids))
            backend.fail = True
            session.score_suffixes([0], [5], [torch.tensor([6])])
    assert session.closed and backend.clears == 1
    assert not backend.model_runner.token_to_kv_pool_allocator.live_pages
    assert len(backend.model_runner.req_to_token_pool.available) == 19


def test_sglang_rejects_noncausal_padding_layout(session_class):
    backend = Backend(4)
    with pytest.raises(ValueError, match="right-padded"):
        with session_class(SimpleNamespace(_backend=backend)) as session:
            session.capture(
                torch.ones(1, 3).long(), torch.tensor([[1, 0, 1]]), torch.ones(1, 3)
            )
    assert backend.clears == 1
