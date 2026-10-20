"""Native DFlash2 beta: proposal law, gradient, alignment and checkpoint tests."""

import contextlib
import json
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

from specforge.core.dflash2_tv_acceptance import (
    OnlineDFlash2TVBetaModel,
    dflash2_proposal_logits,
    sample_dflash2_beta_proposals,
)
from specforge.core.tv_acceptance_beta import tv_acceptance_beta_terms
from specforge.inference.target_engine.proposal_verification import (
    ProposalVerificationSession,
)
from specforge.legacy.dspark_training.dflash2 import (
    dflash2_beta_resume_contract,
    prepare_dflash2_config,
    validate_dflash2_beta_resume,
)
from specforge.legacy.dspark_training.initialization import (
    initialize_draft_weights,
    load_checkpoint_model,
    prepare_warm_start,
)
from specforge.modeling.draft.dflash2 import DFlash2DraftModel

from tests.test_scripts.test_train_dspark import training_args
from .test_tv_acceptance import RecordingTarget


def tiny_config(**overrides):
    return Qwen3Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=16,
        layer_types=["full_attention"],
        dflash_config={
            "block_size": 8,
            "conv_group_size": 4,
            "conv_kernel_size": 2,
            "mask_token_id": 15,
            "selector_rank": 4,
            "selector_top_k": 4,
            "target_layer_ids": [1],
            **overrides,
        },
    )


def make_dflash2(**overrides):
    config = tiny_config(**overrides)
    config._attn_implementation = "sdpa"
    model = OnlineDFlash2TVBetaModel(
        DFlash2DraftModel(config),
        nn.Linear(16, 16, bias=False),
        nn.Embedding(16, 16),
        mask_token_id=15,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=2,
    )
    model.configure_beta(
        temperature=0.8, chunk_blocks=2, eos_token_ids=(), verification_batch_size=2
    )
    with torch.no_grad():
        selector = model.draft_model.candidate_selector
        selector.predecessor_codebook.normal_(std=0.6)
        selector.successor_codebook.normal_(std=0.6)
        selector.hidden_projection.weight.normal_(std=0.3)
    return model, RecordingTarget(16)


@pytest.mark.parametrize("chunk", [1, 3])
def test_selector_sampling_and_replay_use_same_support_and_conditional_probability(
    chunk,
):
    torch.manual_seed(83)
    model, _ = make_dflash2(output_multiplier=1.4, final_logit_softcapping=2.0)
    model.tv_objective_chunk_blocks = chunk
    hidden = torch.randn(3, 7, 16, requires_grad=True)
    rollout = sample_dflash2_beta_proposals(
        model, hidden, torch.tensor([1, 2, 3]), torch.ones(3, 7)
    )
    logits = dflash2_proposal_logits(
        model, hidden, rollout.previous, rollout.candidate_ids
    )
    log_q = torch.log_softmax(logits / model.tv_temperature, -1)
    torch.testing.assert_close(
        log_q.gather(-1, rollout.tokens[..., None]).squeeze(-1), rollout.log_probs
    )
    expected_ids = (
        model.draft_model.transform_unary_logits(model.lm_head(hidden))
        .topk(4, -1)
        .indices
    )
    assert torch.equal(rollout.candidate_ids, expected_ids)
    assert torch.equal(rollout.previous[:, 1:], rollout.tokens[:, :-1])
    assert rollout.previous[:, 0].tolist() == [1, 2, 3]
    assert torch.isfinite(logits).sum(-1).eq(4).all()
    assert (logits.softmax(-1) == 0).sum(-1).eq(12).all()
    assert not rollout.log_probs.requires_grad


def test_sampled_eos_is_included_and_truncates_suffix():
    model, _ = make_dflash2()
    hidden = torch.randn(2, 7, 16)
    unary = model.draft_model.transform_unary_logits(model.lm_head(hidden))
    eos = unary[0, 0].topk(4).indices[0].item()
    model.tv_eos_token_ids = (eos,)
    with patch("torch.multinomial", return_value=torch.zeros(2, 1, dtype=torch.long)):
        rollout = sample_dflash2_beta_proposals(
            model, hidden, torch.tensor([1, 2]), torch.ones(2, 7)
        )
    assert rollout.tokens[0, 0] == eos
    assert rollout.valid[0].tolist() == [True] + [False] * 6


@pytest.mark.parametrize("with_eos", [False, True])
def test_enumerated_selector_policy_gradient_matches_exact_acceptance_tree(with_eos):
    torch.manual_seed(73)
    model, _ = make_dflash2(selector_top_k=2)
    hidden = torch.randn(1, 3, 16, requires_grad=True)
    unary = model.draft_model.transform_unary_logits(model.lm_head(hidden))
    ids = unary.detach().topk(2, -1).indices
    anchor = 3
    eos = ids[0, 0, 0].item() if with_eos else -1
    target = torch.randn(7, 16).softmax(-1)
    selector = model.draft_model.candidate_selector

    def score(depth, predecessor):
        # Independent dense reference for the bilinear candidate distribution.
        candidates = ids[0, depth]
        context = selector.predecessor_codebook[predecessor] * (
            hidden[0, depth] @ selector.hidden_projection.weight.T
        )
        scores = (
            unary[0, depth, candidates]
            + selector.successor_codebook[candidates] @ context
        )
        return (scores / model.tv_temperature).softmax(-1)

    def exact(node=0, depth=0, predecessor=anchor):
        mass = torch.minimum(target[node, ids[0, depth]], score(depth, predecessor))
        value = mass.sum()
        if depth < 2:
            value = value + sum(
                mass[j] * exact(2 * node + 1 + j, depth + 1, token)
                for j, token in enumerate(ids[0, depth].tolist())
                if token != eos
            )
        return value

    estimate = hidden.new_zeros(())
    for choices in product(range(2), repeat=3):
        tokens = ids[0].gather(-1, torch.tensor(choices)[:, None]).T
        prev = torch.cat((torch.tensor([[anchor]]), tokens[:, :-1]), -1)
        logits = dflash2_proposal_logits(model, hidden, prev, ids)
        old = (
            torch.log_softmax(logits.detach() / model.tv_temperature, -1)
            .gather(-1, tokens[..., None])
            .squeeze(-1)
        )
        node, nodes = 0, []
        for choice in choices:
            nodes.append(node)
            node = node * 2 + 1 + choice
        valid = torch.ones_like(tokens, dtype=torch.bool)
        if eos in tokens[0].tolist():
            valid[:, tokens[0].tolist().index(eos) + 1 :] = False
        terms = tv_acceptance_beta_terms(
            logits,
            target[nodes].log()[None] * model.tv_temperature,
            valid,
            tokens,
            old,
            model.tv_temperature,
        )
        estimate = estimate + old.exp().prod() * terms.loss_sum
    reference = 1 - exact() / 7
    parameters = [hidden, *selector.parameters()]
    torch.testing.assert_close(estimate, reference)
    actual = torch.autograd.grad(estimate, parameters, retain_graph=True)
    expected = torch.autograd.grad(reference, parameters)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_target_mass_outside_candidates_is_not_renormalized():
    q = torch.tensor([[[0.0, -torch.inf, -torch.inf]]])
    tokens = torch.zeros(1, 1, dtype=torch.long)
    terms = tv_acceptance_beta_terms(
        q, torch.zeros_like(q), torch.ones(1, 1), tokens, torch.zeros(1, 1)
    )
    torch.testing.assert_close(terms.length_sum, torch.tensor(1 / 3))


@pytest.mark.parametrize("chunk", [1, 4])
def test_native_forward_alignment_gradients_and_no_verification_in_backward(chunk):
    torch.manual_seed(71)
    model, target = make_dflash2()
    model.tv_objective_chunk_blocks = chunk
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    anchors = torch.tensor([[0, 4]])
    kept = torch.ones_like(anchors, dtype=torch.bool)
    captured, rollouts = [], []
    handle = model.draft_model.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    original_sampler = sample_dflash2_beta_proposals

    def sampling(*args):
        rollout = original_sampler(*args)
        rollouts.append(rollout)
        return rollout

    with ProposalVerificationSession(target) as session:
        output = session.capture(ids, torch.ones_like(ids), torch.ones_like(ids))
        with (
            patch.object(
                model, "_sample_anchor_positions", return_value=(anchors, kept)
            ),
            patch(
                "specforge.core.dflash2_tv_acceptance.sample_dflash2_beta_proposals",
                side_effect=sampling,
            ),
        ):
            result = model(
                ids,
                output.hidden_states,
                torch.ones_like(ids),
                last_hidden_states=output.last_hidden_states,
                tv_verification_session=session,
            )
    handle.remove()
    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.valid.sum(-1).tolist() == [7, 5]
    reference_hidden = (
        captured[0].detach().reshape(1, 2, 8, 16)[:, :, 1:].clone().requires_grad_()
    )
    reference = reference_hidden.new_zeros(())
    for block, anchor in enumerate(anchors[0].tolist()):
        n = rollout.valid[block].sum().item()
        candidate = torch.cat((ids[0, : anchor + 1], rollout.tokens[block, : n - 1]))[
            None
        ]
        teacher = target.generate_dflash_data(
            candidate, torch.ones_like(candidate), torch.ones_like(candidate)
        )
        logits = dflash2_proposal_logits(
            model,
            reference_hidden[:, block, :n],
            rollout.previous[block : block + 1, :n],
            rollout.candidate_ids[block : block + 1, :n],
        )
        reference = (
            reference
            + tv_acceptance_beta_terms(
                logits,
                model.lm_head(teacher.last_hidden_states[:, anchor : anchor + n]),
                rollout.valid[block : block + 1, :n],
                rollout.tokens[block : block + 1, :n],
                rollout.log_probs[block : block + 1, :n],
                model.tv_temperature,
            ).loss_sum
            / 2
        )
    torch.testing.assert_close(result[0], reference)
    selector_parameters = list(model.draft_model.candidate_selector.parameters())
    expected = torch.autograd.grad(reference, selector_parameters)
    with (
        patch("torch.multinomial", side_effect=AssertionError("backward resampled")),
        patch.object(
            target,
            "generate_dflash_data",
            side_effect=AssertionError("backward verified"),
        ),
    ):
        result[0].backward()
    for parameter, grad in zip(selector_parameters, expected, strict=True):
        torch.testing.assert_close(parameter.grad, grad, atol=2e-6, rtol=2e-5)
    conv = model.draft_model.layers[0].attention_conv.kernel_projection.weight
    assert torch.isfinite(conv.grad).all() and conv.grad.count_nonzero() > 0
    assert model.lm_head.weight.grad is None and model.embed_tokens.weight.grad is None
    assert "tv_acceptance_length_estimate" in result[-1]
    assert result[2].shape == (7,)
    optimizer = torch.optim.AdamW(model.draft_model.parameters(), lr=1e-3)
    before = conv.detach().clone()
    optimizer.step()
    assert not torch.equal(before, conv)


@pytest.mark.parametrize(
    "field,value", [("block_size", 7), ("selector_top_k", 0), ("shift_label", True)]
)
def test_invalid_dflash2_configs_rejected(field, value):
    config = tiny_config(**{field: value})
    with pytest.raises(ValueError):
        prepare_dflash2_config(
            config, training_args("--dspark-loss-type", "tv-acceptance-beta")
        )


def test_dflash2_config_and_model_factory_use_native_architecture(
    tmp_path, monkeypatch
):
    from specforge.legacy.dspark_training import setup

    config = tiny_config()
    config.save_pretrained(tmp_path)
    args = training_args(
        "--draft-config-path",
        str(tmp_path / "config.json"),
        "--dspark-loss-type",
        "tv-acceptance-beta",
        "--attention-backend",
        "sdpa",
    )
    layers = []
    target = SimpleNamespace(set_capture_layers=lambda ids: layers.extend(ids))
    monkeypatch.setattr(setup, "print_on_rank0", lambda *args: None)
    monkeypatch.setattr(setup, "get_dflash_target_model", lambda **kwargs: target)
    _, draft = setup.build_models(args, torch.device("cpu"))
    assert type(draft) is DFlash2DraftModel
    assert draft.block_size == 8 and args.block_size == 8
    assert layers == [1]
    assert not hasattr(draft, "markov_head")


def test_dflash2_checkpoint_export_and_strict_warm_start(tmp_path, monkeypatch):
    from specforge.legacy.dspark_training import checkpoint

    model, _ = make_dflash2()
    args = training_args(
        "--dspark-loss-type", "tv-acceptance-beta", "--output-dir", str(tmp_path)
    )
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        checkpoint.FSDP, "state_dict_type", lambda *args: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        checkpoint,
        "save_fsdp_optimizer_state",
        lambda optimizer, model, path: torch.save({}, path),
    )
    checkpoint.save_checkpoint(args, 0, 3, model, model.draft_model, None)
    path = tmp_path / "epoch_0_step_3"
    assert json.loads((path / "config.json").read_text())["architectures"] == [
        "DFlash2DraftModel"
    ]
    assert not (path / "dspark.py").exists()
    state = torch.load(path / "training_state.pt", weights_only=True)
    assert isinstance(state["args"], dict)
    loaded = load_checkpoint_model(path, dtype=torch.float32)
    assert type(loaded) is DFlash2DraftModel
    for key, value in model.draft_model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    args.init_draft_model_path = str(path)
    prepare_warm_start(args)
    target = DFlash2DraftModel(loaded.config)
    initialize_draft_weights(target, args)
    for key, value in model.draft_model.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)
    # A different selector architecture must not silently reinitialize weights.
    incompatible = DFlash2DraftModel(tiny_config(selector_rank=8))
    with pytest.raises(RuntimeError, match="size mismatch"):
        initialize_draft_weights(incompatible, args)


def test_dspark_checkpoint_cannot_silently_override_requested_dflash2(tmp_path):
    config = tmp_path / "dflash2.json"
    config.write_text(json.dumps({"architectures": ["DFlash2DraftModel"]}))
    source = tmp_path / "dspark"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3DSparkModel"]})
    )
    (source / "model.safetensors").touch()
    args = training_args(
        "--draft-config-path", str(config), "--init-draft-model-path", str(source)
    )
    with pytest.raises(ValueError, match="not interchangeable"):
        prepare_warm_start(args)


def test_resume_rejects_changed_candidate_support_and_old_objective():
    model, _ = make_dflash2()
    state = {"dflash2_tv_beta": dflash2_beta_resume_contract(model.draft_model)}
    validate_dflash2_beta_resume(state, model.draft_model)
    model.draft_model.config.dflash_config["selector_top_k"] = 3
    with pytest.raises(ValueError, match="proposal semantics"):
        validate_dflash2_beta_resume(state, model.draft_model)
    with pytest.raises(ValueError, match="proposal semantics"):
        validate_dflash2_beta_resume({}, model.draft_model)


def test_dspark_selector_restriction_is_unchanged():
    from specforge.core.tv_acceptance import configure_tv_acceptance
    from .test_tv_acceptance_beta import beta_model

    model, _ = beta_model()
    model.draft_model.candidate_selector = nn.Linear(16, 16)
    with pytest.raises(ValueError, match="does not support candidate_selector"):
        configure_tv_acceptance(
            model, temperature=1, chunk_blocks=1, eos_token_ids=(), beta=True
        )


def test_native_masks_truncate_at_first_hole_and_empty_rank_participates():
    model, target = make_dflash2()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    anchors = torch.tensor([[0, 4]])
    mask = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 1, 1, 1]])
    with ProposalVerificationSession(target) as session:
        output = session.capture(ids, torch.ones_like(ids), mask)
        with patch.object(
            model,
            "_sample_anchor_positions",
            return_value=(anchors, torch.ones_like(anchors).bool()),
        ):
            result = model(
                ids,
                output.hidden_states,
                mask,
                last_hidden_states=output.last_hidden_states,
                tv_verification_session=session,
            )
    assert result[4].tolist() == [2, 2, 1, 1, 1, 0, 0]

    def all_reduce(tensor):
        if tensor.shape == (3,):
            tensor[2] = 1  # One block on another rank, zero on this rank.

    with ProposalVerificationSession(target) as session:
        output = session.capture(ids, torch.ones_like(ids), torch.zeros_like(ids))
        with (
            patch(
                "specforge.core.tv_acceptance.dist.is_initialized", return_value=True
            ),
            patch("specforge.core.tv_acceptance.dist.get_world_size", return_value=2),
            patch(
                "specforge.core.tv_acceptance.dist.all_reduce", side_effect=all_reduce
            ),
        ):
            result = model(
                ids,
                output.hidden_states,
                torch.zeros_like(ids),
                last_hidden_states=output.last_hidden_states,
                tv_verification_session=session,
            )
    result[0].backward()
    assert result[0] == 0
    for parameter in model.draft_model.parameters():
        assert parameter.grad is not None
        assert (
            torch.isfinite(parameter.grad).all() and parameter.grad.count_nonzero() == 0
        )


@pytest.mark.parametrize(
    "objective,block_size,anchor_mode,rollout_budget",
    [
        ("tv-acceptance-beta", 8, "random", 9),
        ("tv-acceptance-beta", 8, "on-policy", 9),
        ("tv-acceptance-beta", 8, "on-policy", 1),
        ("dflash2", 7, "random", 9),
        ("dpace", 7, "random", 9),
    ],
)
def test_shared_trainer_native_dflash2_fsdp_step_save_and_resume(
    tmp_path, monkeypatch, objective, block_size, anchor_mode, rollout_budget
):
    """Run the real trainer/checkpointer, with a real HF target for on-policy."""
    import copy
    import functools
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from transformers import Qwen3ForCausalLM

    from specforge.inference.target_engine.dflash_target_model import (
        HFDFlashTargetEngine,
    )
    from specforge.legacy.dspark_training import setup, trainer

    config = tiny_config()
    config.eos_token_id = 14
    target_path = tmp_path / "target"
    config.save_pretrained(target_path)
    draft_path = tmp_path / "draft-config.json"
    config.to_json_file(draft_path)
    if objective == "dpace":
        # Exercise the reported workflow: initialize from an existing native
        # block-8 checkpoint, train with block 7, then resume the new objective.
        DFlash2DraftModel(config).save_pretrained(tmp_path / "warm-start")

    class BF16Target(RecordingTarget):
        def set_capture_layers(self, layers):
            assert layers == [1]

        def proposal_verification(self):
            return ProposalVerificationSession(self)

        def generate_dflash_data(self, *args):
            output = super().generate_dflash_data(*args)
            output.hidden_states = output.hidden_states.bfloat16()
            output.last_hidden_states = output.last_hidden_states.bfloat16()
            return output

    target = BF16Target(16)
    captured_prefixes = []
    if anchor_mode == "on-policy":
        target_config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            head_dim=4,
            max_position_embeddings=64,
            vocab_size=16,
            layer_types=["full_attention"] * 2,
            # Keep both optimizer steps nonempty regardless of sampled tokens.
            eos_token_id=None,
        )
        target_config._attn_implementation = "sdpa"
        target_config.save_pretrained(target_path)
        target_template = (
            Qwen3ForCausalLM(target_config).bfloat16().eval().requires_grad_(False)
        )

    def build_target(**_kwargs):
        nonlocal target
        if anchor_mode == "on-policy":
            # Each trainer run receives fresh modules with identical frozen
            # weights, rather than modules already owned by an earlier FSDP.
            target = HFDFlashTargetEngine(copy.deepcopy(target_template))

            def record_prefill(_module, _args, kwargs):
                if kwargs.get("past_key_values") is None:
                    captured_prefixes.append(kwargs["input_ids"].detach().clone())

            target.model.model.register_forward_pre_hook(
                record_prefill, with_kwargs=True
            )
        return target

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    loss_mask = torch.ones_like(ids)
    if anchor_mode == "on-policy":
        loss_mask[:, :3] = 0
    batch = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        loss_mask=loss_mask,
        sample_id=torch.tensor([0]),
        source_id=torch.tensor([0]),
    )

    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)

    monkeypatch.setattr(setup, "get_dflash_target_model", build_target)
    monkeypatch.setattr(trainer, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(trainer, "get_dp_group", lambda: None)
    monkeypatch.setattr(trainer, "destroy_distributed", dist.destroy_process_group)
    monkeypatch.setattr(trainer, "_patch_flex_attention_triton_backend", lambda: None)
    monkeypatch.setattr(
        trainer, "FSDP", functools.partial(FSDP, device_id=torch.device("cpu"))
    )
    monkeypatch.setattr(
        trainer, "build_dataloader", lambda *args: (Loader([batch, batch]), None)
    )
    monkeypatch.setattr(
        trainer.AutoTokenizer,
        "from_pretrained",
        lambda *args: SimpleNamespace(eos_token_id=14, mask_token_id=15),
    )
    monkeypatch.setattr(
        trainer.TargetEmbeddingsAndHead,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            lm_head=(
                target.model.lm_head
                if anchor_mode == "on-policy"
                else nn.Linear(16, 16, bias=False).bfloat16()
            ),
            embed_tokens=(
                target.model.model.embed_tokens
                if anchor_mode == "on-policy"
                else nn.Embedding(16, 16).bfloat16()
            ),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "create_tracker",
        lambda *args: SimpleNamespace(
            log=lambda *args, **kwargs: None, close=lambda: None
        ),
    )

    for stop in (1, 2):
        rendezvous = tmp_path / f"rendezvous-{stop}"
        monkeypatch.setattr(
            trainer,
            "init_distributed",
            lambda **kwargs: dist.init_process_group(
                "gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1
            ),
        )
        args = training_args(
            "--target-model-path",
            str(target_path),
            "--draft-config-path",
            str(draft_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--dspark-loss-type",
            objective,
            "--tv-anchor-mode",
            anchor_mode,
            "--tv-rollout-max-tokens",
            str(rollout_budget),
            "--max-length",
            "24",
            "--target-model-backend",
            "hf",
            "--dpace-alpha",
            "0.3",
            "--attention-backend",
            "sdpa",
            "--block-size",
            str(block_size),
            "--num-anchors",
            "2",
            "--num-epochs",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "2",
            "--stop-after-steps",
            str(stop),
            "--log-interval",
            "1",
            "--save-interval",
            "10",
            *(
                ["--init-draft-model-path", str(tmp_path / "warm-start")]
                if objective == "dpace" and stop == 1 else []
            ),
            *(["--resume"] if stop == 2 else []),
        )
        try:
            trainer.run_training(args)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    final = tmp_path / "output" / "epoch_1_step_2"
    state = torch.load(final / "training_state.pt", weights_only=True)
    assert state["global_step"] == 2
    if objective == "dflash2":
        assert state["dflash2_native"]["policy"] == "native-ce-selector-v1"
        assert state["dflash2_native"]["loss_decay_gamma"] == 4.0
        assert "dflash2_tv_beta" not in state
    elif objective == "dpace":
        assert state["dflash2_native"]["policy"] == "native-dpace-selector-v1"
        assert state["dflash2_native"]["dpace_alpha"] == 0.3
        assert "dflash2_tv_beta" not in state
    else:
        assert state["dflash2_tv_beta"]["policy"] == "strict-topk-selector-beta-v1"
    if anchor_mode == "on-policy":
        assert len(captured_prefixes) == 2
        for prefix in captured_prefixes:
            torch.testing.assert_close(prefix, ids[:, :3])
        assert state["args"]["tv_anchor_mode"] == "on-policy"
        assert state["args"]["tv_rollout_max_tokens"] == rollout_budget
        assert state["args"]["num_anchors"] == 2
        assert state["args"]["max_length"] == 24
        first_state = torch.load(
            tmp_path / "output" / "epoch_0_step_1" / "training_state.pt",
            weights_only=True,
        )
        for checkpoint, expected_step in ((first_state, 1), (state, 2)):
            adam_states = checkpoint["optimizer_state_dict"]["state"]
            if rollout_budget == 1:
                assert not adam_states
            else:
                assert adam_states
                assert {int(value["step"]) for value in adam_states.values()} == {
                    expected_step
                }
        if rollout_budget == 1:
            assert first_state["scheduler_state_dict"] == state["scheduler_state_dict"]
        assert all(parameter.grad is None for parameter in target.model.parameters())
    restored = load_checkpoint_model(final)
    assert type(restored) is DFlash2DraftModel
    assert restored.block_size == block_size
