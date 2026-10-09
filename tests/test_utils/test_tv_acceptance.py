"""CPU formula, prefix alignment, checkpoint and real DSpark gradient tests."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from specforge.core.tv_acceptance import (
    candidate_tv_forward,
    configure_tv_acceptance,
    sample_fixed_candidates,
    sample_tv_anchors,
    tv_acceptance_terms,
)
from specforge.inference.target_engine.candidate_verifier import CandidatePrefixVerifier


def test_formula_and_single_token_tv():
    p = torch.tensor([[[0.8, 0.2], [0.5, 0.5]]], dtype=torch.double)
    q = torch.tensor([[[0.6, 0.4], [0.2, 0.8]]], dtype=torch.double)
    terms = tv_acceptance_terms(q.log(), p.log(), torch.ones(1, 2).bool())
    torch.testing.assert_close(terms.loss_sum, torch.tensor(0.32, dtype=torch.double))
    torch.testing.assert_close(terms.length_sum, torch.tensor(1.36, dtype=torch.double))
    one = tv_acceptance_terms(q[:, :1].log(), p[:, :1].log(), torch.ones(1, 1).bool())
    torch.testing.assert_close(one.loss_sum, torch.tensor(0.2, dtype=torch.double))


def test_partial_blocks_holes_empty_blocks_and_nan_padding():
    q = torch.tensor([[[0.6, 0.4]] * 3] * 3, dtype=torch.double).log().requires_grad_()
    p = torch.tensor([[[0.8, 0.2]] * 3] * 3, dtype=torch.double).log().requires_grad_()
    mask = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 0, 0]]).bool()
    with torch.no_grad():
        q[2] = float("nan")
        p[2] = float("nan")
    terms = tv_acceptance_terms(q, p, mask)
    assert terms.block_count.item() == 2
    # K=2: 1-(.8+.64)/2=.28; K=1: .2. The empty block has no mass.
    torch.testing.assert_close(terms.loss_sum, torch.tensor(0.48, dtype=torch.double))
    terms.loss_sum.backward()
    assert torch.isfinite(q.grad).all()
    assert torch.count_nonzero(q.grad[1, 1:]) == 0
    assert torch.count_nonzero(q.grad[2]) == 0
    assert p.grad is None


def test_identical_disjoint_and_all_empty_distributions():
    q = torch.tensor([[[2.0, -2.0], [0.0, 0.0]]], requires_grad=True)
    mask = torch.ones(1, 2).bool()
    equal = tv_acceptance_terms(q, q.clone(), mask)
    assert equal.loss_sum.item() == 0
    assert equal.length_sum.item() == 2
    empty = tv_acceptance_terms(q, q, ~mask)
    assert empty.loss_sum.item() == empty.block_count.item() == 0
    empty.loss_sum.backward()
    assert torch.count_nonzero(q.grad) == 0
    disjoint = tv_acceptance_terms(
        torch.tensor([[[0.0, -torch.inf], [0.0, 0.0]]]),
        torch.tensor([[[-torch.inf, 0.0], [0.0, 0.0]]]),
        mask,
    )
    assert disjoint.loss_sum.item() == 1
    assert disjoint.length_sum.item() == 0


def test_gradcheck_and_all_prefix_credit_matches_naive_reference():
    torch.manual_seed(51)
    q = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    p = torch.randn_like(q)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]]).bool()
    objective = lambda value: tv_acceptance_terms(value, p, mask, 0.7).loss_sum
    assert torch.autograd.gradcheck(objective, (q,))
    a = 1 - 0.5 * (torch.softmax(q / 0.7, -1) - torch.softmax(p / 0.7, -1)).abs().sum(
        -1
    )
    losses = []
    for row, k in enumerate([3, 2]):
        prefix = q.new_ones(())
        length = q.new_zeros(())
        for position in range(k):
            prefix = prefix * a[row, position]
            length = length + prefix
        losses.append(1 - length / k)
    reference = sum(losses)
    torch.testing.assert_close(objective(q), reference)
    torch.testing.assert_close(
        torch.autograd.grad(objective(q), q)[0],
        torch.autograd.grad(reference, q)[0],
    )


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_sampling_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        tv_acceptance_terms(
            torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.ones(1, 1), temperature
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_promotes_to_fp32(dtype):
    q = torch.tensor([[[1.0, 0.0], [0.3, 0.7]]], dtype=dtype, requires_grad=True)
    p = torch.zeros_like(q)
    terms = tv_acceptance_terms(q, p, torch.ones(1, 2))
    assert terms.loss_sum.dtype == torch.float32
    terms.loss_sum.backward()
    assert torch.isfinite(q.grad).all()


class RecordingTarget:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.requests = []

    def generate_dflash_data(self, input_ids, attention_mask, loss_mask):
        assert not torch.is_grad_enabled()
        self.requests.extend(
            [row[mask.bool()].tolist() for row, mask in zip(input_ids, attention_mask)]
        )
        # Causal state depends on every token in the ACTUAL prefix.
        state = (input_ids * attention_mask).double().cumsum(-1)[..., None]
        return SimpleNamespace(
            last_hidden_states=state.expand(-1, -1, self.hidden_size)
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_verifier_replaces_gold_prefix_independently_and_aligns_prediction_rows(
    batch_size,
):
    target = RecordingTarget(2)
    verifier = CandidatePrefixVerifier(target, batch_size=batch_size, pad_token_id=99)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    anchors = torch.tensor([[1, 2, 4]])
    proposals = torch.tensor([[[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
    mask = torch.tensor([[[1, 1, 1], [1, 1, 0], [0, 0, 0]]]).bool()
    result = verifier(ids, torch.ones_like(ids), anchors, proposals, mask, 2)
    assert target.requests == [[1, 2, 9, 8], [1, 2, 3, 6]]
    assert result[0, 0, :, 0].tolist() == [3, 12, 20]
    assert result[0, 1, :, 0].tolist() == [6, 12, 0]
    assert result[0, 2].count_nonzero() == 0


def test_verifier_rejects_missing_hidden_and_left_padding():
    target = SimpleNamespace(
        generate_dflash_data=lambda *args: SimpleNamespace(last_hidden_states=None)
    )
    verifier = CandidatePrefixVerifier(target)
    args = (
        torch.tensor([[1, 2, 3]]),
        torch.ones(1, 3).long(),
        torch.tensor([[1]]),
        torch.tensor([[[4]]]),
        torch.ones(1, 1, 1).bool(),
        2,
    )
    with pytest.raises(ValueError, match="final hidden"):
        verifier(*args)
    with pytest.raises(ValueError, match="right-padded"):
        verifier(args[0], torch.tensor([[0, 1, 1]]), *args[2:])


def make_model():
    from specforge.core.dspark import OnlineDSparkModel
    from specforge.legacy.dspark import DSparkConfig, DSparkDraftModel

    config = DSparkConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        block_size=7,
        num_target_layers=4,
        dflash_config={"target_layer_ids": [1]},
        markov_rank=4,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    config._attn_implementation = "sdpa"
    model = OnlineDSparkModel(
        DSparkDraftModel(config),
        nn.Linear(16, 16, bias=False),
        nn.Embedding(16, 16),
        mask_token_id=15,
        block_size=7,
        attention_backend="sdpa",
        num_anchors=2,
    )
    model.embed_tokens.requires_grad_(False)
    target = RecordingTarget(16)
    configure_tv_acceptance(
        model,
        temperature=0.8,
        chunk_blocks=1,
        verifier=CandidatePrefixVerifier(target, 2),
        eos_token_ids=[],
    )
    return model, target


def test_sampling_uses_sampled_predecessors_and_truncates_after_eos():
    model, _ = make_model()
    hidden = torch.randn(2, 3, 16)
    first = torch.tensor([3, 4])
    model.tv_objective_chunk_blocks = 2
    model.tv_eos_token_ids = (7,)
    draws = [
        torch.tensor([[7], [5]]),
        torch.tensor([[8], [7]]),
        torch.tensor([[6], [9]]),
    ]
    head = model.draft_model.markov_head
    with (
        patch("torch.multinomial", side_effect=draws),
        patch.object(head, "compute_step_bias", wraps=head.compute_step_bias) as spy,
    ):
        sampled, prev, mask = sample_fixed_candidates(
            model, hidden, first, torch.ones(2, 3)
        )
    assert sampled.tolist() == [[7, 8, 6], [5, 7, 9]]
    assert prev.tolist() == [[3, 7, 8], [4, 5, 7]]
    assert mask.tolist() == [[True, False, False], [True, True, False]]
    assert [c.args[0].tolist() for c in spy.call_args_list] == [[3, 4], [7, 5], [8, 7]]
    assert not sampled.requires_grad


def test_checkpoint_replay_matches_fixed_candidate_reference_and_never_rescores():
    torch.manual_seed(21)
    model, target = make_model()
    hidden = torch.randn(1, 2, 3, 16, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    anchors = torch.tensor([[1, 3]])
    valid = torch.tensor([[[1, 1, 1], [1, 1, 0]]]).bool()
    sampled = torch.tensor([[8, 9, 10], [11, 12, 13]])
    previous = torch.tensor([[2, 8, 9], [4, 11, 12]])
    with patch(
        "specforge.core.tv_acceptance.sample_fixed_candidates",
        return_value=(sampled, previous, valid.reshape(2, 3)),
    ) as sample:
        result = candidate_tv_forward(
            model, ids, torch.ones_like(ids), anchors, hidden, valid
        )
        requests_before_backward = copy.deepcopy(target.requests)
        result[0].backward()
        sample.assert_called_once()
    assert target.requests == requests_before_backward
    grad = hidden.grad.clone()
    markov_grad = model.draft_model.markov_head.markov_w2.weight.grad.clone()
    model.zero_grad()
    reference_hidden = hidden.detach().clone().requires_grad_()
    teacher = model.tv_candidate_verifier(
        ids, torch.ones_like(ids), anchors, sampled.reshape(1, 2, 3), valid, 16
    )
    logits = model.draft_model.markov_head.apply_block_logits(
        model.lm_head(reference_hidden),
        token_ids=previous.reshape(1, 2, 3),
        hidden_states=reference_hidden,
    )
    terms = tv_acceptance_terms(logits, model.lm_head(teacher.float()), valid, 0.8)
    reference = terms.loss_sum / terms.block_count
    reference.backward()
    torch.testing.assert_close(result[0], reference)
    torch.testing.assert_close(grad, reference_hidden.grad)
    torch.testing.assert_close(
        markov_grad, model.draft_model.markov_head.markov_w2.weight.grad
    )
    assert model.lm_head.weight.grad is None


def test_real_dspark_forward_backward_uses_new_loss_without_gold_teacher_states():
    torch.manual_seed(31)
    model, target = make_model()
    ids = torch.randint(0, 15, (2, 12))
    result = model(
        input_ids=ids,
        hidden_states=torch.randn(2, 12, 16),
        loss_mask=torch.ones(2, 12),
        attention_mask=torch.ones_like(ids),
    )
    assert len(result) == 6
    loss, _, _, _, _, metrics = result
    assert 0 <= loss.item() <= 1
    torch.testing.assert_close(loss.detach(), metrics["tv_acceptance_loss"])
    assert target.requests
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.draft_model.layers.parameters()
    )
    assert model.draft_model.markov_head.markov_w2.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad is None
    assert all(p.grad is None for p in model.draft_model.confidence_head.parameters())


def test_disabled_objective_keeps_legacy_loss():
    model, target = make_model()
    model.tv_acceptance_enabled = False
    ids = torch.randint(0, 15, (1, 12))
    loss, *_rest, metrics = model(
        input_ids=ids,
        hidden_states=torch.randn(1, 12, 16),
        loss_mask=torch.ones(1, 12),
        last_hidden_states=torch.randn(1, 12, 16),
    )
    assert torch.isfinite(loss)
    assert "tv_acceptance_loss" not in metrics
    assert not target.requests


@pytest.mark.parametrize("empty_rank", [False, True])
def test_distributed_gradient_scaling_uses_global_block_count(empty_rank):
    """A rank with 2 (or 0) blocks must not get the weight of a rank with 1."""
    torch.manual_seed(23)
    model, _ = make_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    anchors = torch.tensor([[1, 3]])
    hidden = torch.randn(1, 2, 3, 16, requires_grad=True)
    valid = torch.tensor([[[1, 1, 1], [1, 1, 0]]]).bool()
    if empty_rank:
        valid.zero_()
    sampled = torch.tensor([[8, 9, 10], [11, 12, 13]])
    previous = torch.tensor([[2, 8, 9], [4, 11, 12]])
    global_blocks = 1 if empty_rank else 3
    calls = []

    def all_reduce(tensor):
        calls.append(tuple(tensor.shape))
        if tensor.shape == (3,):
            tensor[2] = global_blocks

    with (
        patch(
            "specforge.core.tv_acceptance.sample_fixed_candidates",
            return_value=(sampled, previous, valid.reshape(2, 3)),
        ),
        patch("specforge.core.tv_acceptance.dist.is_initialized", return_value=True),
        patch("specforge.core.tv_acceptance.dist.get_world_size", return_value=2),
        patch("specforge.core.tv_acceptance.dist.all_reduce", side_effect=all_reduce),
    ):
        loss = candidate_tv_forward(
            model, ids, torch.ones_like(ids), anchors, hidden, valid
        )[0]
        loss.backward()
    assert calls == [(3,), (3, 3)]
    actual = hidden.grad.clone()
    hidden.grad = None
    teacher = model.tv_candidate_verifier(
        ids, torch.ones_like(ids), anchors, sampled.reshape(1, 2, 3), valid, 16
    )
    logits = model.draft_model.markov_head.apply_block_logits(
        model.lm_head(hidden),
        token_ids=previous.reshape(1, 2, 3),
        hidden_states=hidden,
    )
    terms = tv_acceptance_terms(logits, model.lm_head(teacher.float()), valid, 0.8)
    # FSDP then averages the two rank gradients. Each block has mass 1/global_blocks.
    (terms.loss_sum / global_blocks).backward()
    torch.testing.assert_close(actual / 2, hidden.grad)
    assert torch.isfinite(actual).all()


def test_all_empty_ranks_and_unsupported_head_fail_explicitly():
    model, _ = make_model()
    with pytest.raises(ValueError, match="no valid candidate blocks"):
        candidate_tv_forward(
            model,
            torch.tensor([[1, 2, 3]]),
            torch.ones(1, 3).long(),
            torch.tensor([[0]]),
            torch.randn(1, 1, 2, 16),
            torch.zeros(1, 1, 2),
        )
    model.draft_model.markov_head.markov_head_type = "carh"
    with pytest.raises(ValueError, match="vanilla"):
        configure_tv_acceptance(
            model,
            temperature=1,
            chunk_blocks=1,
            verifier=model.tv_candidate_verifier,
            eos_token_ids=[],
        )


def test_anchor_sampler_keeps_partial_tail_and_empty_rank_placeholder():
    anchors, keep = sample_tv_anchors(
        torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 0, 0]]),
        torch.ones(2, 5),
        512,
    )
    assert anchors.tolist() == [[3], [0]]
    assert keep.tolist() == [[True], [False]]
    model, _ = make_model()
    with pytest.raises(ValueError, match="no valid candidate blocks"):
        model(
            input_ids=torch.ones(1, 8).long(),
            hidden_states=torch.randn(1, 8, 16),
            loss_mask=torch.zeros(1, 8),
            attention_mask=torch.ones(1, 8).long(),
        )


@pytest.mark.parametrize("model_type", ["dspark", "qwen3"])
@pytest.mark.parametrize("weight_format", ["safetensors", "sharded", "bin"])
def test_checkpoint_post_training_preserves_weights_and_takes_tv_step(
    tmp_path, model_type, weight_format
):
    from specforge.legacy.dspark_training.initialization import (
        initialize_draft_weights,
        load_checkpoint_model,
        load_draft_config,
        prepare_warm_start,
    )

    torch.manual_seed(12)
    source, _ = make_model()
    source.draft_model.config.dflash_config["mask_token_id"] = 13
    # Make the checkpoint observably trained instead of just another identical
    # zero-initialized Markov projection.
    with torch.no_grad():
        source.draft_model.markov_head.markov_w2.weight.normal_(std=0.01)
    if weight_format == "bin":
        source.draft_model.config.save_pretrained(tmp_path)
        torch.save(source.draft_model.state_dict(), tmp_path / "pytorch_model.bin")
    else:
        source.draft_model.save_pretrained(
            tmp_path, max_shard_size="2KB" if weight_format == "sharded" else "1GB"
        )
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(model_type=model_type, architectures=["Qwen3DSparkModel"])
    config_path.write_text(json.dumps(config))
    # Post-training must not attempt to read optimizer/counters from this file.
    (tmp_path / "training_state.pt").write_bytes(b"not an optimizer checkpoint")
    args = SimpleNamespace(
        init_draft_model_path=str(tmp_path),
        resume=False,
        output_dir=str(tmp_path / "new-run"),
        allow_conv_position_pruning=False,
    )
    prepare_warm_start(args)
    resolved = load_draft_config(args.draft_config_path)
    assert resolved.block_size == 7
    assert resolved.dflash_config["target_layer_ids"] == [1]
    loaded = load_checkpoint_model(args.init_draft_model_path, dtype=torch.float32)
    assert loaded.mask_token_id == 13
    assert loaded.markov_rank == 4
    model, _ = make_model()
    initialize_draft_weights(model.draft_model, args)
    for key, expected in source.draft_model.state_dict().items():
        torch.testing.assert_close(
            model.draft_model.state_dict()[key], expected, rtol=0, atol=0
        )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )
    assert not optimizer.state
    before = model.draft_model.markov_head.markov_w2.weight.detach().clone()
    ids = torch.randint(0, 13, (1, 12))
    loss, *_ = model(
        input_ids=ids,
        hidden_states=torch.randn(1, 12, 16),
        loss_mask=torch.ones(1, 12),
        attention_mask=torch.ones_like(ids),
    )
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(model.draft_model.markov_head.markov_w2.weight, before)


@pytest.mark.parametrize("invalid", ["missing", "unexpected"])
def test_warm_start_rejects_silently_reinitialized_source_weights(tmp_path, invalid):
    from safetensors.torch import load_file, save_file

    from specforge.legacy.dspark_training.initialization import load_checkpoint_model

    model, _ = make_model()
    model.draft_model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    if invalid == "missing":
        del state["markov_head.markov_w2.weight"]
    else:
        state["unknown.weight"] = torch.ones(1)
    save_file(state, path)
    with pytest.raises(ValueError, match="Incomplete or incompatible"):
        load_checkpoint_model(tmp_path, dtype=torch.float32)
