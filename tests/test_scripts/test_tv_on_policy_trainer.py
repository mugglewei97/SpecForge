"""The trainer must hand raw examples to the on-policy prompt collector."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from specforge.legacy.dspark_training import trainer


def _args(*, seeded=False):
    return SimpleNamespace(
        seed=731,
        dspark_loss_type="tv-acceptance-beta",
        tv_anchor_mode="on-policy",
        multi_teacher_oracle_export_dir=None,
        multi_teacher_oracle_cache=None,
        step_seeded_rollouts=seeded,
    )


def _batch():
    return {
        "input_ids": torch.tensor([[2, 3, 71, 72, 73], [4, 5, 81, 82, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 0]]),
        "sample_id": torch.tensor([101, 102]),
        "source_id": torch.tensor([0, 1]),
    }


class _Target:
    def __init__(self):
        self.session = object()
        self.events = []

    @contextmanager
    def on_policy_rollout(self):
        self.events.append("enter")
        try:
            yield self.session
        finally:
            self.events.append("exit")

    def proposal_verification(self):
        pytest.fail("on-policy examples must not enter the reference-prefix session")

    def generate_dflash_data(self, *_args, **_kwargs):
        pytest.fail("trainer must not capture the original reference responses")


@pytest.mark.parametrize("seeded", [False, True])
def test_on_policy_trainer_defers_capture_and_preserves_prompt_boundaries(
    monkeypatch, seeded
):
    monkeypatch.setattr(trainer.dist, "get_rank", lambda: 3)
    data = _batch()
    target = _Target()
    expected = object()

    def draft(**kwargs):
        assert target.events == ["enter"]
        assert kwargs["tv_on_policy_session"] is target.session
        assert kwargs["hidden_states"] is None
        assert "target_last_hidden_states" not in kwargs
        for name in ("input_ids", "attention_mask", "loss_mask"):
            torch.testing.assert_close(kwargs[name], data[name])
        torch.testing.assert_close(kwargs["sample_ids"], data["sample_id"])
        torch.testing.assert_close(kwargs["source_ids"], data["source_id"])
        return expected

    result = trainer._forward_dspark_data_batch(
        data,
        _args(seeded=seeded),
        "cpu",
        target,
        draft,
        needs_target_hidden=True,
        global_step=17,
    )
    assert result is expected
    assert target.events == ["enter", "exit"]


def test_on_policy_trainer_closes_target_session_when_collection_fails(monkeypatch):
    monkeypatch.setattr(trainer.dist, "get_rank", lambda: 0)
    target = _Target()

    def draft(**_kwargs):
        raise RuntimeError("verification failed")

    with pytest.raises(RuntimeError, match="verification failed"):
        trainer._forward_dspark_data_batch(
            _batch(),
            _args(),
            "cpu",
            target,
            draft,
            needs_target_hidden=True,
            global_step=1,
        )
    assert target.events == ["enter", "exit"]


def test_step_seeded_on_policy_rollout_is_reproducible_and_microbatch_distinct(
    monkeypatch,
):
    monkeypatch.setattr(trainer.dist, "get_rank", lambda: 2)

    def run(micro_index):
        return trainer._forward_dspark_data_batch(
            _batch(),
            _args(seeded=True),
            "cpu",
            _Target(),
            lambda **_: torch.rand(16),
            needs_target_hidden=True,
            global_step=31,
            micro_index=micro_index,
        )

    first = run(0)
    torch.rand(19)
    torch.testing.assert_close(run(0), first)
    assert not torch.equal(run(1), first)
