"""CARH sampled-policy replay and gradient coverage for the TV beta objective."""

from unittest.mock import patch

import pytest
import torch

from specforge.core.tv_acceptance import configure_tv_acceptance
from specforge.core.tv_acceptance_beta import (
    sample_beta_proposals,
    sampled_tv_beta_forward,
    tv_acceptance_beta_terms,
)
from specforge.inference.target_engine.proposal_verification import (
    ProposalVerificationSession,
)

from .test_tv_acceptance import make_model


def make_carh_model(**head_options):
    from specforge.legacy.dspark import ContextAwareCausalResidualHead

    model, target = make_model()
    head = ContextAwareCausalResidualHead(
        vocab_size=16,
        markov_rank=4,
        hidden_size=16,
        block_size=7,
        **head_options,
    )
    # In particular, undo the output projection's zero initialization: otherwise
    # incorrect hidden/depth conditioning has no effect on the proposal policy.
    with torch.no_grad():
        head.markov_w2.weight.normal_(std=0.25)
        head.hidden_proj.weight.normal_(std=0.2)
        head.depth_embedding.weight.normal_(std=0.3)
        head.gate_proj.weight.normal_(std=0.25)
        head.gate_proj.bias.fill_(0.2)
        head.fusion_norm.bias.normal_(std=0.1)
    model.draft_model.markov_head = head
    config = model.draft_model.config
    config.markov_head_type = "carh"
    config.carh_predecessor_count = head.predecessor_count
    config.carh_predecessor_context_mode = head.predecessor_context_mode
    config.carh_sampled_prefix_memory_rank = head.sampled_prefix_memory_rank
    return model, target


def configure_beta(model, chunk_size=2):
    configure_tv_acceptance(
        model,
        temperature=0.8,
        chunk_blocks=chunk_size,
        eos_token_ids=[],
        beta=True,
    )


def test_carh_sampled_policy_matches_block_replay_with_hidden_and_depth():
    torch.manual_seed(207)
    model, _ = make_carh_model()
    model.double()
    configure_beta(model)
    hidden = torch.randn(3, 7, 16, dtype=torch.double, requires_grad=True)
    rollout = sample_beta_proposals(
        model, hidden, torch.tensor([1, 2, 3]), torch.ones(3, 7).bool()
    )
    head = model.draft_model.markov_head
    replay_logits = head.apply_block_logits(
        model.lm_head(hidden),
        token_ids=rollout.previous,
        hidden_states=hidden,
    )
    step_logits = torch.stack(
        [
            model.lm_head(hidden[:, depth])
            + head.compute_step_bias(
                rollout.previous[:, depth], hidden[:, depth], depth
            )
            for depth in range(7)
        ],
        dim=1,
    )
    torch.testing.assert_close(replay_logits, step_logits)
    replay_log_probs = torch.log_softmax(replay_logits / model.tv_temperature, -1)
    replay_log_probs = replay_log_probs.gather(
        -1, rollout.tokens[..., None]
    ).squeeze(-1)
    torch.testing.assert_close(replay_log_probs, rollout.log_probs)
    assert not rollout.log_probs.requires_grad
    assert torch.equal(rollout.previous[:, 1:], rollout.tokens[:, :-1])

    tokens, h = rollout.previous[:, 1], hidden[:, 1]
    bias = head.compute_step_bias(tokens, h, 1)
    assert not torch.allclose(bias, head.compute_step_bias(tokens, h, 0))
    assert not torch.allclose(
        bias, head.compute_step_bias(tokens, torch.zeros_like(h), 1)
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_carh_checkpoint_value_and_all_gradients_match_step_reference(chunk_size):
    torch.manual_seed(208)
    model, target = make_carh_model()
    model.double()
    configure_beta(model, chunk_size)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    anchors = torch.tensor([[0, 2, 5]])
    hidden = torch.randn(1, 3, 7, 16, dtype=torch.double, requires_grad=True)
    valid = torch.tensor([[[1] * 7, [1] * 5 + [0] * 2, [1] * 3 + [0] * 4]]).bool()
    saved_rollouts = []

    def record_sample(*args):
        rollout = sample_beta_proposals(*args)
        saved_rollouts.append(rollout)
        return rollout

    with ProposalVerificationSession(target) as session:
        original = session.capture(ids, torch.ones_like(ids), torch.ones_like(ids))
        original.last_hidden_states.requires_grad_()
        with patch(
            "specforge.core.tv_acceptance_beta.sample_beta_proposals",
            side_effect=record_sample,
        ) as sampler:
            result = sampled_tv_beta_forward(
                model,
                ids,
                torch.ones_like(ids),
                anchors,
                hidden,
                valid,
                original.last_hidden_states,
                session,
            )
            sampler.assert_called_once()
        target_requests = list(target.requests)
    with patch("torch.multinomial", side_effect=AssertionError("backward resampled")):
        result[0].backward()
    assert target.requests == target_requests
    actual_hidden_grad = hidden.grad.clone()
    head = model.draft_model.markov_head
    actual_head_grads = {}
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
        actual_head_grads[name] = parameter.grad.clone()
    assert original.last_hidden_states.grad is None
    assert model.lm_head.weight.grad is None
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.draft_model.confidence_head.parameters()
    )

    model.zero_grad()
    reference_hidden = hidden.detach().clone().requires_grad_()
    rollout = saved_rollouts[0]
    reference = hidden.new_zeros(())
    # Build each complete candidate independently. Use sequential CARH steps,
    # rather than the vectorized block API under test in checkpoint replay.
    for block, anchor in enumerate(anchors[0].tolist()):
        length = int(rollout.valid[block].sum())
        candidate = torch.cat(
            (ids[0, : anchor + 1], rollout.tokens[block, : length - 1])
        )[None]
        output = target.generate_dflash_data(
            candidate, torch.ones_like(candidate), torch.ones_like(candidate)
        )
        h = reference_hidden[:, block, :length]
        logits = torch.stack(
            [
                model.lm_head(h[:, depth])
                + head.compute_step_bias(
                    rollout.previous[block : block + 1, depth], h[:, depth], depth
                )
                for depth in range(length)
            ],
            dim=1,
        )
        target_h = output.last_hidden_states[:, anchor : anchor + length].to(h)
        reference += tv_acceptance_beta_terms(
            logits,
            model.lm_head(target_h),
            rollout.valid[block : block + 1, :length],
            rollout.tokens[block : block + 1, :length],
            rollout.log_probs[block : block + 1, :length],
            model.tv_temperature,
        ).loss_sum / anchors.numel()
    reference.backward()
    torch.testing.assert_close(result[0], reference)
    torch.testing.assert_close(actual_hidden_grad, reference_hidden.grad)
    for name, parameter in head.named_parameters():
        torch.testing.assert_close(actual_head_grads[name], parameter.grad)
    assert "tv_acceptance_length_estimate" in result[-1]


@pytest.mark.parametrize(
    "head_options",
    [
        {"predecessor_count": 2},
        {"predecessor_count": 3},
        {"predecessor_context_mode": "context"},
        {"predecessor_context_mode": "innovation"},
        {"sampled_prefix_memory_rank": 2},
    ],
)
def test_carh_beta_rejects_unsupported_conditioning_before_sampling(head_options):
    model, _ = make_carh_model(**head_options)
    with pytest.raises(ValueError, match="CARH"):
        configure_beta(model)


def test_carh_beta_rejects_memory_module_with_zero_rank_metadata():
    model, _ = make_carh_model(sampled_prefix_memory_rank=2)
    model.draft_model.markov_head.sampled_prefix_memory_rank = 0
    model.draft_model.config.carh_sampled_prefix_memory_rank = 0
    with pytest.raises(ValueError, match="CARH"):
        configure_beta(model)


def test_carh_teacher_forced_tv_is_still_rejected():
    model, _ = make_carh_model()
    with pytest.raises(ValueError, match="vanilla"):
        configure_tv_acceptance(
            model,
            temperature=0.8,
            chunk_blocks=2,
            eos_token_ids=[],
            beta=False,
        )
