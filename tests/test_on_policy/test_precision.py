"""Probability precision across the real FSDP mixed-precision boundary."""

import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from specforge.on_policy.objective import block_loss, draft_distribution
from specforge.on_policy.replay import create_replay_draft
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from tests.test_on_policy.test_replay import tiny_model


def check_bf16_replay_precision(device, *, packed=False):
    """Shared CPU and two-GPU gate: real DSpark replay with BF16 FSDP."""
    model = tiny_model().to(device=device, dtype=torch.bfloat16)
    model.draft_model.config.architectures = ["DSparkDraftModel"]
    model.draft_model = create_replay_draft(model.draft_model.config).to(device=device)
    expected_inv_freq = model.draft_model.rotary_emb.inv_freq.clone()
    features = torch.randn(4, 32)  # Replay must explicitly cast these to BF16.
    target = torch.zeros(3, 16)
    target[:, :2] = torch.tensor([0.9, 0.1])
    rollout_q = torch.randn(3, 16).softmax(-1)
    block = {
        "context_length": 4,
        "anchor": 2,
        "proposal": [3, 4, 5],
        "valid_mask": [True, True, True],
        "sampling": {"temperature": 1.0},
    }
    if packed:
        block = [block, {**block, "anchor": 3, "proposal": [4, 5, 6]}]
        target = torch.stack([target, target])
        rollout_q = torch.stack([rollout_q, rollout_q])
    with torch.no_grad():
        expected_loss, expected_parity = model(block, features, target, rollout_q)

    def check_inputs(_module, inputs):
        # Check values as well as dtype: BF16 -> FP32 cannot restore precision.
        for actual, expected in zip(inputs[2:], (target, rollout_q)):
            torch.testing.assert_close(
                actual, expected.to(actual.device), atol=0, rtol=0
            )

    def check_activations(_module, _args, kwargs):
        torch.testing.assert_close(
            _module.rotary_emb.inv_freq, expected_inv_freq, atol=0, rtol=0
        )
        assert kwargs["target_hidden"].dtype == torch.bfloat16
        assert kwargs["noise_embedding"].dtype == torch.bfloat16

    handles = [
        model.register_forward_pre_hook(check_inputs),
        model.draft_model.register_forward_pre_hook(
            check_activations, with_kwargs=True
        ),
    ]
    try:
        backend = FSDPTrainingBackend(ParallelConfig(sharding_strategy="FULL_SHARD"))
        backend.prepare_model(
            model,
            optimizer_target=model.draft_model,
            cast_root_forward_inputs=False,
        )
        loss, parity = backend.module(block, features, target, rollout_q)
        assert loss.dtype == torch.float32
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(parity, expected_parity)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
        assert model.lm_head.weight.grad is None
        assert model.embed_tokens.weight.grad is None
    finally:
        for handle in handles:
            handle.remove()


class ProbabilityReplay(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 2))

    def forward(self, target):
        return block_loss(
            target, draft_distribution(self.logits, 1.0), torch.tensor([True])
        )


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        dist.init_process_group(
            "gloo", init_method=f"file://{directory.name}/rdzv", rank=0, world_size=1
        )
        self.addCleanup(dist.destroy_process_group)
        # Select CPU explicitly while keeping real FSDP wrapping/casting intact.
        cpu_fsdp = patch(
            "torch.distributed.fsdp.FullyShardedDataParallel",
            side_effect=lambda module, **kwargs: FSDP(
                module, device_id=torch.device("cpu"), **kwargs
            ),
        )
        cpu_fsdp.start()
        self.addCleanup(cpu_fsdp.stop)

    def test_default_fsdp_cast_still_reproduces_normalization_failure(self):
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(ProbabilityReplay())
        with self.assertRaisesRegex(ValueError, "invalid sampling distribution"):
            backend.module(torch.tensor([[0.9, 0.1]]))

    def test_bf16_fsdp_preserves_replay_probabilities_and_gradients(self):
        check_bf16_replay_precision(torch.device("cpu"))

    def test_packed_bf16_fsdp_preserves_fp32_probabilities_and_gradients(self):
        check_bf16_replay_precision(torch.device("cpu"), packed=True)
