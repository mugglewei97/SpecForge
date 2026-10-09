# coding=utf-8
"""DSpark draft model: DFlash backbone + EAGLE-style Markov and confidence heads.

DSpark shares SpecForge's DFlash block-diffusion drafter (dual-source KV
injection via :class:`DFlashDraftModel`, anchor sampling, MASK-token noise
stream) and adds two heads on top:

  - Markov head: a low-rank learned bigram bias added to the draft logits,
    conditioned on the (teacher-forced) previous token. Improves the per-token
    distribution without touching the backbone.
  - Confidence head (AcceptRatePredictor): predicts a per-draft-position
    acceptance probability, trained against the empirical draft-vs-target
    accept rate (used at inference time for adaptive block length).

Ported from TorchSpec PR #129 (``torchspec/models/draft/dspark.py``). The Markov
/ confidence modeling code is adapted from DeepSeek's DeepSpec
(``deepspec/modeling/dspark/{markov_head,common}.py``, MIT License).

SpecForge differences vs TorchSpec (load-bearing):
  - There is no ``DFlashConfig``; SpecForge's :class:`DFlashDraftModel` uses a
    plain ``Qwen3Config`` plus a ``config.dflash_config`` dict. So
    :class:`DSparkConfig` subclasses ``Qwen3Config`` and declares the DSpark
    fields as top-level attributes; DFlash-carried fields (``block_size``,
    ``num_target_layers``, ``dflash_config``) stay as before.
  - The draft model has no ``embed_tokens`` of its own (the embedding lives on
    the target and is passed into the online wrapper), and the context
    projection is ``self.fc`` (not ``context_proj``). The heads only depend on
    ``config.hidden_size`` / ``config.vocab_size``, so this does not matter for
    construction.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from .dflash import DFlashDraftModel
from .dflash2 import (
    CandidateSelector,
    DFlashGroupedConv,
    LocalTransitionAttention,
    Qwen3DFlash2DecoderLayer,
)


class DSparkConfig(Qwen3Config):
    """Configuration for the DSpark draft model.

    Extends ``Qwen3Config`` (SpecForge's DFlash draft is config-light and reads a
    plain ``Qwen3Config``). DSpark-specific fields are declared here; the
    DFlash-carried fields (``block_size``, ``num_target_layers``, and the nested
    ``dflash_config`` dict holding ``target_layer_ids`` / ``mask_token_id``) are
    consumed by the :class:`DFlashDraftModel` base ``__init__`` and must be
    present on the config object before constructing the model.
    """

    model_type = "dspark"

    def __init__(
        self,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        carh_gate_bias: float = 0.0,
        carh_predecessor_count: int = 1,
        carh_predecessor_context_mode: str = "none",
        carh_sampled_prefix_memory_rank: int = 0,
        selector_rank: int = 0,
        selector_top_k: int = 0,
        selector_runtime_enabled: bool = True,
        selector_margin_threshold: float = 0.0,
        recall_correction_rank: int = 0,
        recall_correction_gate_bias: float = -2.0,
        prefix_state_mixer_mode: str = "none",
        prefix_state_rank: int = 128,
        prefix_state_retention_bias: float = 2.0,
        prefix_state_update_bias: float = -1.0,
        prefix_state_gate_bias: float = -2.0,
        prefix_state_residual_scale: float = 0.1,
        parallel_refiner_rank: int = 0,
        parallel_refiner_steps: int = 0,
        parallel_refiner_gate_bias: float = -1.0,
        parallel_refiner_second_step_bias: float = -1.0,
        parallel_refiner_residual_scale: float = 0.1,
        parallel_refiner_runtime_enabled: bool = True,
        refiner_advantage_temperature_conditioned: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov
        self.carh_gate_bias = float(carh_gate_bias)
        self.carh_predecessor_count = int(carh_predecessor_count)
        self.carh_predecessor_context_mode = str(carh_predecessor_context_mode)
        self.carh_sampled_prefix_memory_rank = int(carh_sampled_prefix_memory_rank)
        self.selector_rank = int(selector_rank)
        self.selector_top_k = int(selector_top_k)
        # A selector may be retained as a training-time path teacher while
        # serving uses the distilled CARH logits directly.  Keep construction
        # separate from runtime activation so its weights remain loadable.
        self.selector_runtime_enabled = bool(selector_runtime_enabled)
        self.selector_margin_threshold = float(selector_margin_threshold)
        self.recall_correction_rank = int(recall_correction_rank)
        self.recall_correction_gate_bias = float(recall_correction_gate_bias)
        self.prefix_state_mixer_mode = str(prefix_state_mixer_mode)
        self.prefix_state_rank = int(prefix_state_rank)
        self.prefix_state_retention_bias = float(prefix_state_retention_bias)
        self.prefix_state_update_bias = float(prefix_state_update_bias)
        self.prefix_state_gate_bias = float(prefix_state_gate_bias)
        self.prefix_state_residual_scale = float(prefix_state_residual_scale)
        self.parallel_refiner_rank = int(parallel_refiner_rank)
        self.parallel_refiner_steps = int(parallel_refiner_steps)
        self.parallel_refiner_gate_bias = float(parallel_refiner_gate_bias)
        self.parallel_refiner_second_step_bias = float(
            parallel_refiner_second_step_bias
        )
        self.parallel_refiner_residual_scale = float(
            parallel_refiner_residual_scale
        )
        # The refiner can be retained as a training-only teacher.  Serving then
        # executes the distilled CARH path and pays no refiner latency.
        self.parallel_refiner_runtime_enabled = bool(
            parallel_refiner_runtime_enabled
        )
        self.refiner_advantage_temperature_conditioned = bool(
            refiner_advantage_temperature_conditioned
        )


class SurvivalConditionedPrefixStateMixer(nn.Module):
    """Strictly causal block-prefix state mixer.

    Each proposal position reads the state accumulated *before* that position,
    so the module cannot leak future draft states. The fixed-size recurrence is
    compatible with a future associative-scan kernel; the reference PyTorch
    implementation intentionally keeps the block-size loop explicit.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        state_rank: int,
        block_size: int,
        mode: str,
        retention_bias: float,
        update_bias: float,
        gate_bias: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if mode not in {"basic", "survival-conditioned"}:
            raise ValueError(
                "prefix state mixer mode must be basic or survival-conditioned"
            )
        if state_rank < 1 or block_size < 1 or residual_scale < 0:
            raise ValueError("invalid prefix state mixer rank/block_size/residual_scale")
        self.mode = mode
        self.state_rank = int(state_rank)
        self.block_size = int(block_size)
        self.residual_scale = float(residual_scale)
        self.input_proj = nn.Linear(hidden_size, state_rank, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 3 * state_rank)
        self.output_proj = nn.Linear(state_rank, hidden_size, bias=False)
        self.depth_embedding = nn.Embedding(block_size, 3 * state_rank)
        self.survival_proj = (
            nn.Linear(hidden_size, 1) if mode == "survival-conditioned" else None
        )
        with torch.no_grad():
            self.gate_proj.bias[:state_rank].fill_(retention_bias)
            self.gate_proj.bias[state_rank : 2 * state_rank].fill_(update_bias)
            self.gate_proj.bias[2 * state_rank :].fill_(gate_bias)
        nn.init.zeros_(self.depth_embedding.weight)
        nn.init.zeros_(self.output_proj.weight)
        if self.survival_proj is not None:
            nn.init.zeros_(self.survival_proj.weight)
            nn.init.constant_(self.survival_proj.bias, 2.0)
        self._last_survival_logits = None
        self._last_diagnostics = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        proposal_len = int(hidden_states.shape[-2])
        if proposal_len < 1 or proposal_len > self.block_size:
            raise ValueError(
                f"SPSM expected 1..{self.block_size} positions, got {proposal_len}"
            )
        original_shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, proposal_len, original_shape[-1])
        depth = self.depth_embedding.weight[:proposal_len].to(dtype=hidden.dtype)
        candidate = self.input_proj(hidden)
        retention_logits, update_logits, output_gate_logits = (
            self.gate_proj(hidden) + depth.unsqueeze(0)
        ).chunk(3, dim=-1)
        retention, update, output_gate = (
            torch.sigmoid(retention_logits),
            torch.sigmoid(update_logits),
            torch.sigmoid(output_gate_logits),
        )
        if self.survival_proj is not None:
            survival_logits = self.survival_proj(hidden).squeeze(-1)
            survival = torch.sigmoid(survival_logits).unsqueeze(-1)
            retention = retention * survival
            output_gate = output_gate * (2.0 - survival)
            self._last_survival_logits = survival_logits
        else:
            self._last_survival_logits = None

        state = torch.zeros_like(candidate[:, 0])
        prefix_states = []
        for depth_index in range(proposal_len):
            # Read-before-write gives strict prefix causality; depth zero is an
            # exact identity and seeds the state with the anchor-slot hidden.
            prefix_states.append(state)
            state = (
                retention[:, depth_index] * state
                + update[:, depth_index] * candidate[:, depth_index]
            )
        prefix_states = torch.stack(prefix_states, dim=1)
        residual = self.output_proj(output_gate * prefix_states)
        scaled_residual = self.residual_scale * residual
        with torch.no_grad():
            hidden_rms = hidden.float().square().mean().sqrt()
            residual_rms = scaled_residual.float().square().mean().sqrt()
            self._last_diagnostics = {
                "retention_gate_mean": retention.float().mean(),
                "update_gate_mean": update.float().mean(),
                "output_gate_mean": output_gate.float().mean(),
                "state_rms": prefix_states.float().square().mean().sqrt(),
                "residual_rms": residual_rms,
                "residual_to_hidden_rms": residual_rms
                / hidden_rms.clamp_min(1e-8),
                "output_proj_weight_rms": self.output_proj.weight.float()
                .square()
                .mean()
                .sqrt(),
            }
        return (hidden + scaled_residual).reshape(original_shape)


class VanillaMarkov(nn.Module):
    """Low-rank learned bigram bias added to the draft logits.

    Adapted from DeepSpec's ``deepspec/modeling/dspark/markov_head.py``.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        assert (
            self.markov_rank > 0
        ), f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)
        # Zero-init the projection so that at initialization the Markov bias
        # is zero.  Without this, the default N(0,1) embedding + Kaiming linear
        # produces a random bias with std ≈ 0.58 that corrupts the backbone
        # logits.  L1-on-corrected_logits gradients through softmax are too
        # weak to train the Markov head from that scale at typical learning
        # rates, causing the head to stay frozen at random init and destroy
        # acceptance rate during serving.
        nn.init.zeros_(self.markov_w2.weight)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        depth_idx: int = 0,
    ) -> torch.Tensor:
        del hidden_states, depth_idx
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del hidden_states
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_block_bias(token_ids=token_ids)

    def compute_block_bias(
        self,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del hidden_states
        return self.compute_step_bias(token_ids)


class GatedMarkovHead(VanillaMarkov):
    """Gate predecessor-token bias using the current backbone hidden state."""

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(int(hidden_size) + markov_rank, markov_rank)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        depth_idx: int = 0,
    ) -> torch.Tensor:
        del depth_idx
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states")
        prev_emb = self.get_prev_embeddings(token_ids)
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
        )
        return self.project_bias(gate.to(prev_emb.dtype) * prev_emb)

    def compute_block_bias(
        self,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_block_bias(
            token_ids=token_ids, hidden_states=hidden_states
        )


class RNNHead(VanillaMarkov):
    """Sequential block-local state over predecessor tokens and backbone hidden."""

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.joint_proj = nn.Linear(2 * markov_rank + int(hidden_size), 3 * markov_rank)

    def init_state(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self.get_prev_embeddings(token_ids))

    def step_with_state(
        self,
        state: torch.Tensor,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prev_emb = self.get_prev_embeddings(token_ids)
        z = torch.cat([state, prev_emb, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        return new_state, self.project_bias(torch.tanh(output_raw))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        depth_idx: int = 0,
    ) -> torch.Tensor:
        del depth_idx
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states")
        _, bias = self.step_with_state(
            self.init_state(token_ids), token_ids, hidden_states
        )
        return bias

    def compute_block_bias(
        self,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states")
        if token_ids.size(-1) == 0:
            return hidden_states.new_empty(*token_ids.shape, self.vocab_size)
        state = self.init_state(token_ids[..., 0])
        biases = []
        for depth in range(token_ids.size(-1)):
            state, bias = self.step_with_state(
                state, token_ids[..., depth], hidden_states[..., depth, :]
            )
            biases.append(bias)
        return torch.stack(biases, dim=-2)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_block_bias(
            token_ids=token_ids, hidden_states=hidden_states
        )


class SampledPrefixMemory(nn.Module):
    """Bounded, low-rank state of already sampled tokens in one proposal block.

    The output starts at zero, so adding this module to a trained CARH head
    leaves its initial logits unchanged. No state is persisted across blocks.
    """

    def __init__(self, markov_rank: int, state_rank: int) -> None:
        super().__init__()
        self.state_rank = int(state_rank)
        self.token_proj = nn.Linear(markov_rank, state_rank, bias=False)
        self.state_proj = nn.Linear(state_rank, state_rank, bias=False)
        self.update_gate = nn.Linear(markov_rank + state_rank, state_rank)
        self.output_proj = nn.Linear(state_rank, markov_rank, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, state: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(
            self.token_proj(token_embedding) + self.state_proj(state)
        )
        gate = torch.sigmoid(
            self.update_gate(torch.cat((token_embedding, state), dim=-1))
        )
        return state + gate * (candidate - state)


class ContextAwareCausalResidualHead(VanillaMarkov):
    """Context-aware causal correction conditioned on the sampled predecessor.

    Vanilla DSpark only projects the previous-token embedding. CARH fuses that
    exact causal signal with the current draft hidden state and draft depth,
    then applies a learned scalar safety gate before the zero-initialized vocab
    projection. The shared low-rank output keeps the sequential serving path
    substantially cheaper than another transformer forward.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        block_size: int,
        gate_bias: float = 0.0,
        predecessor_count: int = 1,
        predecessor_context_mode: str = "none",
        sampled_prefix_memory_rank: int = 0,
    ) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "carh"
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("CARH requires block_size > 0")
        self.hidden_proj = nn.Linear(int(hidden_size), self.markov_rank, bias=False)
        self.depth_embedding = nn.Embedding(self.block_size, self.markov_rank)
        self.fusion_norm = nn.LayerNorm(self.markov_rank)
        self.gate_proj = nn.Linear(self.markov_rank, 1)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))
        self._last_gate_mean: Optional[torch.Tensor] = None
        if predecessor_count not in (1, 2, 3):
            raise ValueError("CARH predecessor_count must be 1, 2 or 3")
        self.predecessor_count = int(predecessor_count)
        supported_context_modes = {
            "none", "innovation", "context", "innovation-position",
            "innovation-residual", "innovation-residual-position",
            "innovation-residual-norm-gate",
        }
        if predecessor_context_mode not in supported_context_modes:
            raise ValueError(
                "Unsupported CARH predecessor context mode: "
                f"{predecessor_context_mode}"
            )
        if predecessor_context_mode != "none" and self.predecessor_count != 1:
            raise ValueError("CARH predecessor context gate requires one predecessor")
        self.predecessor_context_mode = str(predecessor_context_mode)
        self.predecessor_context_proj = None
        self.predecessor_context_scale = None
        self.predecessor_context_position_scale = None
        self.innovation_down = None
        self.innovation_up = None
        self.innovation_position_gate = None
        self.innovation_norm_gate_threshold = None
        if self.predecessor_context_mode != "none":
            # An extra rank-space projection should not perturb the baseline
            # initialization stream of the backbone or existing CARH modules.
            with torch.random.fork_rng(devices=[]):
                self.predecessor_context_proj = nn.Linear(
                    self.markov_rank, self.markov_rank, bias=False
                )
            self.predecessor_context_scale = nn.Parameter(
                torch.zeros(self.markov_rank)
            )
            if self.predecessor_context_mode == "innovation-position":
                # Position-specific channel strengths; zero init preserves
                # the original CARH logits at initialization.
                self.predecessor_context_position_scale = nn.Parameter(
                    torch.zeros(self.block_size, self.markov_rank)
                )
            elif self.predecessor_context_mode in {
                "innovation-residual", "innovation-residual-position",
                "innovation-residual-norm-gate",
            }:
                # Learn a low-rank correction from the innovation vector itself,
                # rather than using it only as a multiplicative gate signal.
                with torch.random.fork_rng(devices=[]):
                    self.innovation_down = nn.Linear(
                        self.markov_rank, 128, bias=False
                    )
                    self.innovation_up = nn.Linear(
                        128, self.markov_rank, bias=False
                    )
                    nn.init.zeros_(self.innovation_up.weight)
                if self.predecessor_context_mode == "innovation-residual-position":
                    self.innovation_position_gate = nn.Parameter(
                        torch.full((self.block_size,), -2.0)
                    )
                elif self.predecessor_context_mode == "innovation-residual-norm-gate":
                    # A shared softplus threshold controls how strongly the
                    # branch trusts large predecessor/backbone mismatches.
                    self.innovation_norm_gate_threshold = nn.Parameter(
                        torch.zeros(1)
                    )
        self.second_prev_proj = None
        self.third_prev_proj = None
        if self.predecessor_count >= 2:
            # Preserve the initialization RNG stream of the original model.
            # This module is built after the backbone's HF post_init.
            with torch.random.fork_rng(devices=[]):
                self.second_prev_proj = nn.Linear(
                    self.markov_rank, self.markov_rank, bias=False
                )
                nn.init.zeros_(self.second_prev_proj.weight)
                if self.predecessor_count == 3:
                    self.third_prev_proj = nn.Linear(
                        self.markov_rank, self.markov_rank, bias=False
                    )
                    nn.init.zeros_(self.third_prev_proj.weight)
        self.sampled_prefix_memory_rank = int(sampled_prefix_memory_rank)
        if self.sampled_prefix_memory_rank < 0:
            raise ValueError("CARH sampled-prefix memory rank must be non-negative")
        if self.sampled_prefix_memory_rank and self.predecessor_count != 1:
            raise ValueError("CARH sampled-prefix memory requires predecessor_count=1")
        self.sampled_prefix_memory: Optional[SampledPrefixMemory] = None
        self.sampled_prefix_memory_scale = 1.0
        if self.sampled_prefix_memory_rank:
            # Preserve the baseline initialization RNG stream for matched A/B
            # scratch training; the extra branch initially changes no logits.
            with torch.random.fork_rng(devices=[]):
                self.sampled_prefix_memory = SampledPrefixMemory(
                    self.markov_rank, self.sampled_prefix_memory_rank
                )

    def init_sampled_prefix_state(self, reference_tensor: torch.Tensor) -> torch.Tensor:
        """Return a block-local zero state with the reference's batch shape."""
        if self.sampled_prefix_memory is None:
            raise ValueError("CARH sampled-prefix memory is disabled")
        return torch.zeros(
            (*reference_tensor.shape, self.sampled_prefix_memory_rank),
            device=reference_tensor.device,
            dtype=self.markov_w1.weight.dtype,
        )

    def advance_sampled_prefix_state(
        self, state: torch.Tensor, sampled_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Consume one realized proposal token (never an anchor or future ID)."""
        if self.sampled_prefix_memory is None:
            raise ValueError("CARH sampled-prefix memory is disabled")
        return self.sampled_prefix_memory(
            state, self.get_prev_embeddings(sampled_token_ids)
        )

    def _block_third_ids(self, token_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self.third_prev_proj is None:
            return None
        return torch.cat([torch.zeros_like(token_ids[..., :2]), token_ids[..., :-2]], dim=-1)

    def _block_second_ids(self, token_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self.second_prev_proj is None:
            return None
        # token_ids = [anchor, z1, z2, ...]. Shift only the last (depth) axis;
        # slot zero is a placeholder and is masked, never a cross-block token.
        return torch.cat([torch.zeros_like(token_ids[..., :1]), token_ids[..., :-1]], dim=-1)

    def _causal_latent_and_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        depth_ids: torch.Tensor,
        second_token_ids: Optional[torch.Tensor] = None,
        use_second_predecessor: bool = True,
        third_token_ids: Optional[torch.Tensor] = None,
        use_third_predecessor: bool = True,
        sampled_prefix_state: Optional[torch.Tensor] = None,
        hidden_context: Optional[torch.Tensor] = None,
        previous_hidden_context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("CARH requires current draft hidden_states")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        if self.second_prev_proj is not None and use_second_predecessor:
            if second_token_ids is None:
                raise ValueError("Dual CARH requires explicit second_token_ids for step calls")
            second = self.second_prev_proj(self.get_prev_embeddings(second_token_ids))
            prev_embeddings = prev_embeddings + second * depth_ids.gt(0).unsqueeze(-1)
        if self.third_prev_proj is not None and use_third_predecessor:
            if third_token_ids is None:
                raise ValueError("Triple CARH requires explicit third_token_ids for step calls")
            third = self.third_prev_proj(self.get_prev_embeddings(third_token_ids))
            prev_embeddings = prev_embeddings + third * depth_ids.gt(1).unsqueeze(-1)
        depth_embeddings = self.depth_embedding(depth_ids.long())
        if hidden_context is None:
            hidden_context = self.hidden_proj(hidden_states)
        fused = prev_embeddings + hidden_context + depth_embeddings
        if self.predecessor_context_mode != "none":
            if previous_hidden_context is None:
                previous_hidden_context = torch.zeros_like(hidden_context)
            predicted = self.predecessor_context_proj(previous_hidden_context)
            conditioning = (
                prev_embeddings - predicted
                if self.predecessor_context_mode.startswith("innovation")
                else predicted
            )
            if self.predecessor_context_mode in {
                "innovation-residual", "innovation-residual-position",
                "innovation-residual-norm-gate",
            }:
                residual = self.innovation_up(
                    torch.tanh(self.innovation_down(conditioning))
                )
                if self.innovation_norm_gate_threshold is not None:
                    mismatch = torch.linalg.vector_norm(
                        conditioning.float(), dim=-1, keepdim=True
                    )
                    reference = (
                        torch.linalg.vector_norm(
                            prev_embeddings.float(), dim=-1, keepdim=True
                        )
                        + torch.linalg.vector_norm(
                            predicted.float(), dim=-1, keepdim=True
                        )
                    ).clamp_min(1e-6)
                    mismatch_ratio = mismatch / reference
                    threshold = torch.nn.functional.softplus(
                        self.innovation_norm_gate_threshold
                    ).clamp_min(1e-4)
                    trust = mismatch_ratio / (mismatch_ratio + threshold)
                    residual = residual * trust.to(residual.dtype)
                if self.innovation_position_gate is not None:
                    residual = residual * torch.sigmoid(
                        self.innovation_position_gate[depth_ids.long()]
                    ).unsqueeze(-1)
            else:
                interaction = torch.tanh(hidden_context) * torch.tanh(conditioning)
                if self.predecessor_context_position_scale is not None:
                    channel_scale = torch.tanh(
                        self.predecessor_context_position_scale[depth_ids.long()]
                    )
                else:
                    channel_scale = torch.tanh(self.predecessor_context_scale)
                residual = channel_scale.to(interaction.dtype) * interaction
            fused = fused + residual * depth_ids.gt(0).unsqueeze(-1)
        if self.sampled_prefix_memory is not None and sampled_prefix_state is not None:
            memory_delta = self.sampled_prefix_memory.output_proj(
                sampled_prefix_state
            )
            fused = fused + (
                self.sampled_prefix_memory_scale
                * memory_delta
                * depth_ids.ge(2).unsqueeze(-1)
            )
        latent = torch.nn.functional.silu(self.fusion_norm(fused))
        gate = torch.sigmoid(self.gate_proj(latent)).to(latent.dtype)
        self._last_gate_mean = gate.detach().float().mean()
        return latent, gate

    def _causal_latent(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        depth_ids: torch.Tensor,
        second_token_ids: Optional[torch.Tensor] = None,
        third_token_ids: Optional[torch.Tensor] = None,
        sampled_prefix_state: Optional[torch.Tensor] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        previous_hidden_context = (
            self.hidden_proj(previous_hidden_states)
            if previous_hidden_states is not None
            and self.predecessor_context_mode != "none"
            else None
        )
        latent, gate = self._causal_latent_and_gate(
            token_ids, hidden_states, depth_ids, second_token_ids,
            third_token_ids=third_token_ids,
            sampled_prefix_state=sampled_prefix_state,
            previous_hidden_context=previous_hidden_context,
        )
        return gate * latent

    def compute_step_bias_gate_and_latent(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        depth_idx: int = 0,
        second_token_ids: Optional[torch.Tensor] = None,
        third_token_ids: Optional[torch.Tensor] = None,
        sampled_prefix_state: Optional[torch.Tensor] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the serving bias plus gate inputs for training-only calibration."""
        if (
            self.sampled_prefix_memory is not None
            and depth_idx >= 2
            and sampled_prefix_state is None
        ):
            raise ValueError("sampled_prefix_state is required at CARH depth >= 2")
        if (
            self.predecessor_context_mode != "none"
            and depth_idx > 0
            and previous_hidden_states is None
        ):
            raise ValueError("previous_hidden_states are required at CARH depth > 0")
        depth_ids = torch.full_like(token_ids, int(depth_idx), dtype=torch.long)
        if depth_idx == 0 and second_token_ids is None:
            second_token_ids = torch.zeros_like(token_ids)
        if depth_idx < 2 and third_token_ids is None:
            third_token_ids = torch.zeros_like(token_ids)
        latent, gate = self._causal_latent_and_gate(
            token_ids, hidden_states, depth_ids, second_token_ids,
            third_token_ids=third_token_ids,
            sampled_prefix_state=sampled_prefix_state,
            previous_hidden_context=(
                self.hidden_proj(previous_hidden_states)
                if previous_hidden_states is not None
                and self.predecessor_context_mode != "none"
                else None
            ),
        )
        return self.project_bias(gate * latent), gate.squeeze(-1), latent

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        depth_idx: int = 0,
        second_token_ids: Optional[torch.Tensor] = None,
        third_token_ids: Optional[torch.Tensor] = None,
        sampled_prefix_state: Optional[torch.Tensor] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            self.sampled_prefix_memory is not None
            and depth_idx >= 2
            and sampled_prefix_state is None
        ):
            raise ValueError("sampled_prefix_state is required at CARH depth >= 2")
        if (
            self.predecessor_context_mode != "none"
            and depth_idx > 0
            and previous_hidden_states is None
        ):
            raise ValueError("previous_hidden_states are required at CARH depth > 0")
        depth_ids = torch.full_like(token_ids, int(depth_idx), dtype=torch.long)
        if depth_idx == 0 and second_token_ids is None:
            second_token_ids = torch.zeros_like(token_ids)
        if depth_idx < 2 and third_token_ids is None:
            third_token_ids = torch.zeros_like(token_ids)
        return self.project_bias(
            self._causal_latent(
                token_ids, hidden_states, depth_ids, second_token_ids,
                third_token_ids, sampled_prefix_state, previous_hidden_states,
            )
        )

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        if hidden_states is None:
            raise ValueError("CARH block logits require hidden_states")
        return base_logits + self.compute_block_bias(
            token_ids=token_ids, hidden_states=hidden_states
        )

    def compute_block_bias(
        self,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bias, _ = self.compute_block_bias_and_latent(
            token_ids=token_ids, hidden_states=hidden_states
        )
        return bias

    def compute_block_bias_and_latent(
        self,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        use_second_predecessor: bool = True,
        use_third_predecessor: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the vocab bias and its gated low-rank CARH latent.

        Training-only preservation/no-op objectives regularize the latent
        directly so they do not need an FP32 copy of the full vocabulary bias.
        """
        if hidden_states is None:
            raise ValueError("CARH block bias requires hidden_states")
        depth_ids = torch.arange(
            token_ids.size(-1), device=token_ids.device, dtype=torch.long
        )
        depth_ids = depth_ids.view(
            *((1,) * (token_ids.ndim - 1)), token_ids.size(-1)
        ).expand_as(token_ids)
        sampled_prefix_states = None
        if self.sampled_prefix_memory is not None:
            if token_ids.size(-1) == 0:
                sampled_prefix_states = self.markov_w1.weight.new_empty(
                    (*token_ids.shape, self.sampled_prefix_memory_rank)
                )
            else:
                state = self.init_sampled_prefix_state(token_ids[..., 0])
                states = []
                for depth in range(token_ids.size(-1)):
                    if depth >= 2:
                        # token_ids=[anchor,z1,z2,...]; at depth d, consume z[d-2]
                        # while the immediate predecessor remains token_ids[d].
                        state = self.advance_sampled_prefix_state(
                            state, token_ids[..., depth - 1]
                        )
                    states.append(state)
                sampled_prefix_states = torch.stack(states, dim=-2)
        hidden_context = (
            self.hidden_proj(hidden_states)
            if self.predecessor_context_mode != "none"
            else None
        )
        previous_hidden_context = None
        if hidden_context is not None:
            previous_hidden_context = torch.cat(
                (torch.zeros_like(hidden_context[..., :1, :]), hidden_context[..., :-1, :]),
                dim=-2,
            )
        latent, gate = self._causal_latent_and_gate(
            token_ids, hidden_states, depth_ids, self._block_second_ids(token_ids),
            use_second_predecessor=use_second_predecessor,
            third_token_ids=self._block_third_ids(token_ids),
            use_third_predecessor=use_third_predecessor,
            sampled_prefix_state=sampled_prefix_states,
            hidden_context=hidden_context,
            previous_hidden_context=previous_hidden_context,
        )
        latent = gate * latent
        return self.project_bias(latent), latent


class AcceptRatePredictor(nn.Module):
    """Per-position acceptance-probability predictor (a single linear head).

    Adapted from DeepSpec's ``deepspec/modeling/dspark/common.py``.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class LowRankRecallCorrection(nn.Module):
    """Zero-initialized hidden residual for unary Top-k recall misses.

    CARH remains the ranking expert in logit space.  This module changes the
    representation *before* the shared target LM head, giving recall-miss
    states a separate path without adding a second vocabulary projection.
    Its predecessor input reuses CARH's low-rank token codebook.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        predecessor_rank: int,
        correction_rank: int,
        block_size: int,
        gate_bias: float,
    ) -> None:
        super().__init__()
        if correction_rank <= 0:
            raise ValueError("recall correction rank must be positive")
        self.block_size = int(block_size)
        self.hidden_proj = nn.Linear(hidden_size, correction_rank, bias=False)
        self.predecessor_proj = nn.Linear(
            predecessor_rank, correction_rank, bias=False
        )
        self.depth_embedding = nn.Embedding(self.block_size, correction_rank)
        self.norm = nn.LayerNorm(correction_rank)
        self.gate_proj = nn.Linear(correction_rank, 1)
        self.output_proj = nn.Linear(correction_rank, hidden_size, bias=False)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))
        nn.init.zeros_(self.output_proj.weight)
        self._last_gate_mean: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        predecessor_latent: torch.Tensor,
        depth_ids: torch.Tensor,
    ) -> torch.Tensor:
        fused = (
            self.hidden_proj(hidden_states)
            + self.predecessor_proj(predecessor_latent)
            + self.depth_embedding(depth_ids.long())
        )
        latent = F.silu(self.norm(fused))
        gate = torch.sigmoid(self.gate_proj(latent)).to(latent.dtype)
        self._last_gate_mean = gate.detach().float().mean()
        return hidden_states + gate * self.output_proj(latent)


class HazardAdaptiveParallelRefiner(nn.Module):
    """Causal-prefix parallel refinement gated by predicted rejection hazard.

    The input token at slot ``j`` is the causal predecessor available to draft
    position ``j`` (anchor for slot 0, then the current proposal prefix).  A
    strictly lower-prefix mixing mask prevents future proposal leakage while
    all positions are still evaluated by one batched matrix operation.  The
    hidden output projection is zero initialized, making checkpoint expansion
    an exact no-op before the new objective starts learning.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        predecessor_rank: int,
        refinement_rank: int,
        block_size: int,
        max_steps: int,
        gate_bias: float,
        second_step_bias: float,
        residual_scale: float,
        temperature_conditioned_advantage: bool = False,
    ) -> None:
        super().__init__()
        if refinement_rank < 1 or block_size < 1 or max_steps < 1:
            raise ValueError("parallel refiner rank/block_size/steps must be positive")
        if residual_scale < 0:
            raise ValueError("parallel refiner residual_scale must be non-negative")
        self.block_size = int(block_size)
        self.max_steps = int(max_steps)
        self.residual_scale = float(residual_scale)
        self.hidden_proj = nn.Linear(hidden_size, refinement_rank, bias=False)
        self.predecessor_proj = nn.Linear(
            predecessor_rank, refinement_rank, bias=False
        )
        self.depth_embedding = nn.Embedding(block_size, refinement_rank)
        self.iteration_embedding = nn.Embedding(max_steps, refinement_rank)
        # Row d mixes causal predecessor slots 0..d.  Starting at zero gives a
        # uniform prefix average; training can learn depth-specific locality.
        self.prefix_mixing_logits = nn.Parameter(
            torch.zeros(block_size, block_size)
        )
        self.norm = nn.LayerNorm(refinement_rank)
        self.hazard_proj = nn.Linear(refinement_rank, 1)
        self.advantage_class_proj = nn.Linear(refinement_rank, 1)
        self.advantage_value_proj = nn.Linear(refinement_rank, 1)
        self.advantage_temperature_proj = (
            nn.Linear(3, 2 * refinement_rank, bias=False)
            if temperature_conditioned_advantage
            else None
        )
        self.output_proj = nn.Linear(refinement_rank, hidden_size, bias=False)
        self.register_buffer(
            "causal_prefix_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            persistent=False,
        )
        nn.init.zeros_(self.depth_embedding.weight)
        nn.init.zeros_(self.iteration_embedding.weight)
        nn.init.zeros_(self.hazard_proj.weight)
        nn.init.constant_(self.hazard_proj.bias, float(gate_bias))
        nn.init.zeros_(self.advantage_class_proj.weight)
        nn.init.zeros_(self.advantage_class_proj.bias)
        nn.init.zeros_(self.advantage_value_proj.weight)
        nn.init.zeros_(self.advantage_value_proj.bias)
        if self.advantage_temperature_proj is not None:
            nn.init.zeros_(self.advantage_temperature_proj.weight)
        nn.init.zeros_(self.output_proj.weight)
        self.second_step_bias = float(second_step_bias)
        self._last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        predecessor_latent: torch.Tensor,
        iteration_idx: int,
        rollout_temperatures: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proposal_len = int(hidden_states.shape[-2])
        if not 1 <= proposal_len <= self.block_size:
            raise ValueError(
                f"parallel refiner expected 1..{self.block_size} positions, "
                f"got {proposal_len}"
            )
        if not 0 <= int(iteration_idx) < self.max_steps:
            raise ValueError("parallel refiner iteration is out of range")
        prefix_logits = self.prefix_mixing_logits[
            :proposal_len, :proposal_len
        ].masked_fill(
            ~self.causal_prefix_mask[:proposal_len, :proposal_len],
            torch.finfo(self.prefix_mixing_logits.dtype).min,
        )
        prefix_weights = torch.softmax(prefix_logits.float(), dim=-1).to(
            predecessor_latent.dtype
        )
        mixed_predecessor = torch.einsum(
            "ij,...jr->...ir", prefix_weights, predecessor_latent
        )
        depth = self.depth_embedding.weight[:proposal_len].to(hidden_states.dtype)
        iteration = self.iteration_embedding.weight[int(iteration_idx)].to(
            hidden_states.dtype
        )
        latent = F.silu(
            self.norm(
                self.hidden_proj(hidden_states)
                + self.predecessor_proj(mixed_predecessor)
                + depth
                + iteration
            )
        )
        hazard_logits = self.hazard_proj(latent).squeeze(-1)
        # These heads are training-only.  They predict whether the *actual*
        # paired-verifier continuation value improves, rather than treating a
        # high rejection hazard as evidence that a correction is useful.
        advantage_latent = latent
        if self.advantage_temperature_proj is not None:
            if rollout_temperatures is None:
                rollout_temperatures = torch.zeros(
                    latent.shape[:-2], device=latent.device, dtype=torch.float32
                )
            temperature = rollout_temperatures.to(
                device=latent.device, dtype=torch.float32
            ).clamp_min(0.0)
            temperature_features = torch.stack(
                [
                    temperature,
                    torch.log1p(temperature),
                    temperature.le(1e-6).float(),
                ],
                dim=-1,
            ).unsqueeze(-2)
            temperature_modulation = self.advantage_temperature_proj(
                temperature_features.to(self.advantage_temperature_proj.weight.dtype)
            ).to(advantage_latent.dtype)
            temperature_scale, temperature_shift = temperature_modulation.chunk(
                2, dim=-1
            )
            advantage_latent = (
                advantage_latent * (1.0 + torch.tanh(temperature_scale))
                + temperature_shift
            )
        self._last_advantage_class_logits = self.advantage_class_proj(
            advantage_latent
        ).squeeze(-1)
        self._last_advantage_value = self.advantage_value_proj(
            advantage_latent
        ).squeeze(-1)
        step_bias = self.second_step_bias * float(iteration_idx)
        gate = torch.sigmoid(hazard_logits + step_bias).to(latent.dtype)
        delta = (
            self.residual_scale
            * gate.unsqueeze(-1)
            * self.output_proj(latent)
        )
        with torch.no_grad():
            self._last_diagnostics = {
                "parallel_refiner_gate_mean": gate.float().mean(),
                "parallel_refiner_hazard_mean": torch.sigmoid(
                    hazard_logits.float()
                ).mean(),
                "parallel_refiner_prefix_entropy": -(
                    prefix_weights.float()
                    * prefix_weights.float().clamp_min(1e-12).log()
                ).sum(dim=-1).mean(),
                "parallel_refiner_advantage_probability": torch.sigmoid(
                    self._last_advantage_class_logits.float()
                ).mean(),
            }
        return delta, hazard_logits, gate


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    predecessor_count = int(getattr(config, "carh_predecessor_count", 1))
    memory_rank = int(getattr(config, "carh_sampled_prefix_memory_rank", 0))
    predecessor_context_mode = str(
        getattr(config, "carh_predecessor_context_mode", "none")
    )
    if predecessor_context_mode != "none":
        if (
            predecessor_context_mode not in {
                "innovation", "context", "innovation-position",
                "innovation-residual", "innovation-residual-position",
                "innovation-residual-norm-gate",
            }
            or markov_rank <= 0
            or str(getattr(config, "markov_head_type", "vanilla")).lower() != "carh"
            or predecessor_count != 1
        ):
            raise ValueError("CARH predecessor context gate requires an enabled pred1 CARH head")
        if memory_rank:
            raise ValueError("CARH predecessor context gate does not support sampled-prefix-memory")
        for key in ("selector_rank", "recall_correction_rank", "parallel_refiner_rank"):
            if int(getattr(config, key, 0)):
                raise ValueError(f"CARH predecessor context gate does not support {key}")
    if memory_rank < 0:
        raise ValueError("carh_sampled_prefix_memory_rank must be non-negative")
    if memory_rank:
        if (
            markov_rank <= 0
            or str(getattr(config, "markov_head_type", "vanilla")).lower() != "carh"
            or predecessor_count != 1
        ):
            raise ValueError("sampled-prefix memory requires an enabled pred1 CARH head")
        for key in ("selector_rank", "recall_correction_rank", "parallel_refiner_rank"):
            if int(getattr(config, key, 0)):
                raise ValueError(f"sampled-prefix CARH does not support {key}")
        if str(getattr(config, "prefix_state_mixer_mode", "none")) != "none":
            raise ValueError("sampled-prefix CARH does not support SPSM")
    if predecessor_count not in (1, 2, 3):
        raise ValueError("carh_predecessor_count must be 1, 2 or 3")
    if predecessor_count >= 2 and (
        str(getattr(config, "markov_head_type", "vanilla")).lower() != "carh"
        or markov_rank <= 0
    ):
        raise ValueError("Dual predecessor requires an enabled CARH head")
    if predecessor_count >= 2:
        for key in ("selector_rank", "recall_correction_rank", "parallel_refiner_rank"):
            if int(getattr(config, key, 0)):
                raise ValueError(f"Multi-predecessor CARH minimal experiment does not support {key}")
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None

    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    if markov_head_type == "vanilla":
        # Use draft_vocab_size when available (compact-vocab models like EAGLE3/EagleSpark),
        # otherwise fall back to vocab_size (full-vocab models like DSpark/DFlash).
        vocab_size = int(getattr(config, "draft_vocab_size", config.vocab_size))
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if markov_head_type in {"gated", "rnn"}:
        vocab_size = int(getattr(config, "draft_vocab_size", config.vocab_size))
        head_class = GatedMarkovHead if markov_head_type == "gated" else RNNHead
        return head_class(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            hidden_size=int(config.hidden_size),
        )
    if markov_head_type == "carh":
        vocab_size = int(getattr(config, "draft_vocab_size", config.vocab_size))
        return ContextAwareCausalResidualHead(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            hidden_size=int(config.hidden_size),
            block_size=int(config.block_size),
            gate_bias=float(getattr(config, "carh_gate_bias", 0.0)),
            predecessor_count=predecessor_count,
            predecessor_context_mode=predecessor_context_mode,
            sampled_prefix_memory_rank=memory_rank,
        )
    raise NotImplementedError(
        f"markov_head_type={markov_head_type!r} is not supported; "
        "expected vanilla, gated, rnn or carh."
    )


class BlockSummaryResidual(nn.Module):
    """Low-rank residual from the anchor or each position's own hidden state."""

    def __init__(
        self, hidden_size: int, rank: int, block_size: int, source: str = "anchor"
    ) -> None:
        super().__init__()
        if source not in {"anchor", "position"}:
            raise ValueError("block_summary_source must be anchor or position")
        self.block_size = int(block_size)
        self.source = source
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.position_gate = nn.Parameter(torch.zeros(block_size - 1))
        self.gate_mode = "position"
        self.gate_query = None
        self.gate_key = None
        self.gate_strength = None
        # This module is created after DFlashDraftModel.post_init().
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 4 or hidden_states.shape[-2] != self.block_size:
            raise ValueError("BlockSummaryResidual expects [batch, blocks, block_size, hidden]")
        if self.block_size == 1:
            return hidden_states
        source_hidden = (
            hidden_states[:, :, 0, :]
            if self.source == "anchor"
            else hidden_states[:, :, 1:, :]
        )
        summary = self.up(torch.nn.functional.silu(self.down(source_hidden)))
        if self.source == "anchor":
            summary = summary.unsqueeze(-2)
        gate_logits = self.position_gate.view(1, 1, self.block_size - 1, 1)
        if self.gate_mode == "compatibility":
            anchor = hidden_states[:, :, :1, :]
            current = hidden_states[:, :, 1:, :]
            query = self.gate_query(current)
            key = self.gate_key(anchor)
            similarity = (query * key).sum(dim=-1, keepdim=True) / (query.size(-1) ** 0.5)
            gate_logits = gate_logits + self.gate_strength.tanh() * similarity.tanh()
        update = summary * gate_logits.sigmoid()
        return torch.cat((hidden_states[:, :, :1, :], hidden_states[:, :, 1:, :] + update), dim=-2)


class DSparkDraftModel(DFlashDraftModel):
    """DSpark draft network: DFlash backbone + Markov / confidence heads.

    When ``dflash_config.conv_kernel_size`` and ``conv_group_size`` are set,
    the backbone reuses DFlash2's causal dynamic grouped convolution around
    every attention and MLP sublayer.  This is orthogonal to the optional
    candidate selector: CARH can therefore use the local convolution without
    paying for online path selection.
    """

    config_class = DSparkConfig
    _no_split_modules = ["Qwen3DFlashDecoderLayer", "Qwen3DFlash2DecoderLayer"]

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ) -> nn.Module:
        method_config = dict(getattr(config, "dflash_config", None) or {})
        taps = int(method_config.get("conv_kernel_size", 0) or 0)
        group_size = int(method_config.get("conv_group_size", 0) or 0)
        transition_rank = int(method_config.get("local_transition_rank", 0) or 0)
        if bool(taps) != bool(group_size):
            raise ValueError(
                "DSpark dynamic convolution requires dflash_config."
                "conv_kernel_size and conv_group_size together"
            )
        if taps == 0 and transition_rank == 0:
            return super()._build_decoder_layer(config, layer_idx)

        last_n_layers = int(method_config.get("conv_last_n_layers", 0) or 0)
        if last_n_layers < 0 or last_n_layers > int(config.num_hidden_layers):
            raise ValueError(
                "dflash_config.conv_last_n_layers must be in "
                f"[0, {config.num_hidden_layers}]"
            )
        if last_n_layers and layer_idx < int(config.num_hidden_layers) - last_n_layers:
            return super()._build_decoder_layer(config, layer_idx)

        conv_mode = str(method_config.get("conv_mode", "legacy"))
        apply_to = str(method_config.get("conv_apply_to", "attention-mlp"))
        if apply_to not in {"attention-output", "attention", "attention-mlp"}:
            raise ValueError(
                "dflash_config.conv_apply_to must be attention-output, "
                f"attention, or attention-mlp; got {apply_to!r}"
            )
        output_only = apply_to == "attention-output"

        def grouped_conv() -> Optional[DFlashGroupedConv]:
            if taps == 0:
                return None
            return DFlashGroupedConv(
                hidden_size=int(config.hidden_size),
                block_size=self.block_size,
                taps=taps,
                group_size=group_size,
                mode=conv_mode,
                kernel_conditioning=str(
                    method_config.get("conv_kernel_conditioning", "input")
                ),
                source_rank=int(method_config.get("conv_source_rank", 32)),
                apply_input=not output_only,
                apply_output=True,
                residual_scale=float(method_config.get("conv_residual_scale", 0.1)),
                gate_bias=float(method_config.get("conv_gate_bias", 2.0)),
                freeze_identity=bool(
                    method_config.get("conv_freeze_identity", False)
                ),
            )

        local_transition = None
        if transition_rank > 0:
            local_transition = LocalTransitionAttention(
                hidden_size=int(config.hidden_size),
                block_size=self.block_size,
                rank=transition_rank,
                num_heads=int(method_config.get("local_transition_heads", 4)),
                window=int(method_config.get("local_transition_window", 2)),
                residual_scale=float(
                    method_config.get("local_transition_residual_scale", 0.1)
                ),
                gate_bias=float(method_config.get("local_transition_gate_bias", -1.0)),
            )

        return Qwen3DFlash2DecoderLayer(
            config,
            layer_idx,
            attention_conv=grouped_conv(),
            mlp_conv=(grouped_conv() if apply_to == "attention-mlp" else None),
            local_transition_attention=local_transition,
            dynamic_conv_scale=float(
                method_config.get("local_transition_conv_scale", 1.0)
            ),
        )

    def __init__(self, config) -> None:
        super().__init__(config)

        summary_rank = int(getattr(config, "block_summary_rank", 0) or 0)
        summary_source = str(getattr(config, "block_summary_source", "anchor"))
        summary_gate_mode = str(getattr(config, "block_summary_gate_mode", "position"))
        if summary_rank < 0 or summary_rank > int(config.hidden_size):
            raise ValueError("block_summary_rank must be in [0, hidden_size]")
        if summary_source not in {"anchor", "position"}:
            raise ValueError("block_summary_source must be anchor or position")
        if summary_gate_mode not in {"position", "compatibility"}:
            raise ValueError("block_summary_gate_mode must be position or compatibility")
        if summary_gate_mode == "compatibility" and summary_source != "anchor":
            raise ValueError("compatibility gate requires anchor block summary")
        self.block_summary = None
        if summary_rank:
            # Keep subsequent initialization and sampling RNG aligned with
            # the matched no-summary control arm.
            with torch.random.fork_rng(devices=[]):
                self.block_summary = BlockSummaryResidual(
                    int(config.hidden_size), summary_rank, self.block_size,
                    source=summary_source,
                )
                if summary_gate_mode == "compatibility":
                    gate_rank = min(32, summary_rank)
                    self.block_summary.gate_mode = summary_gate_mode
                    self.block_summary.gate_query = nn.Linear(config.hidden_size, gate_rank, bias=False)
                    self.block_summary.gate_key = nn.Linear(config.hidden_size, gate_rank, bias=False)
                    # FSDP cannot shard zero-dimensional parameters.
                    self.block_summary.gate_strength = nn.Parameter(torch.zeros(1))

        method_config = dict(getattr(config, "dflash_config", None) or {})
        self.dynamic_conv_enabled = bool(
            int(method_config.get("conv_kernel_size", 0) or 0)
        )
        self.local_transition_enabled = bool(
            int(method_config.get("local_transition_rank", 0) or 0)
        )
        self.local_transition_conv_decay_start_ratio = float(
            method_config.get("local_transition_conv_decay_start_ratio", 0.10)
        )
        self.local_transition_conv_decay_end_ratio = float(
            method_config.get("local_transition_conv_decay_end_ratio", 0.70)
        )
        self.local_transition_final_conv_scale = float(
            method_config.get("local_transition_final_conv_scale", 0.0)
        )
        if not (
            0.0 <= self.local_transition_conv_decay_start_ratio
            < self.local_transition_conv_decay_end_ratio
            <= 1.0
        ):
            raise ValueError("invalid local transition convolution decay schedule")
        if not 0.0 <= self.local_transition_final_conv_scale <= 1.0:
            raise ValueError("local transition final convolution scale must be in [0, 1]")
        # Hugging Face post_init initializes Linear weights after the grouped
        # convolution constructor.  Re-zero only the dynamic deltas so the
        # exported model starts as the original CARH backbone.  The legacy
        # mode keeps base_kernel trainable; survival-gated experiments may
        # freeze it to preserve the exact identity path.
        if self.dynamic_conv_enabled:
            for layer in self.layers:
                for conv in (
                    getattr(layer, "attention_conv", None),
                    getattr(layer, "mlp_conv", None),
                ):
                    if conv is not None:
                        if bool(method_config.get("conv_identity_init", True)):
                            if conv.kernel_projection.weight is not None:
                                nn.init.zeros_(conv.kernel_projection.weight)
                        # HF post_init also reinitializes the optional source
                        # branch: restore its zero-residual initialization.
                        if conv.source_up is not None:
                            nn.init.zeros_(conv.source_up.weight)
        if self.local_transition_enabled:
            # Hugging Face post_init runs after child constructors; restore the
            # exact zero-residual initialization promised by LTA.
            for layer in self.layers:
                transition = getattr(layer, "local_transition_attention", None)
                if transition is not None:
                    transition.reset_output_projection()

        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )

        self.markov_head = build_markov_head(config)

        mixer_mode = str(getattr(config, "prefix_state_mixer_mode", "none"))
        if mixer_mode not in {"none", "basic", "survival-conditioned"}:
            raise ValueError(f"unsupported prefix_state_mixer_mode={mixer_mode!r}")
        if mixer_mode != "none" and self.dynamic_conv_enabled:
            raise ValueError(
                "SPSM replaces DynamicConv; set conv_kernel_size and "
                "conv_group_size to 0 when prefix_state_mixer_mode is enabled"
            )

        self.prefix_state_mixer: Optional[nn.Module] = None
        if mixer_mode != "none":
            self.prefix_state_mixer = SurvivalConditionedPrefixStateMixer(
                hidden_size=int(config.hidden_size),
                state_rank=int(getattr(config, "prefix_state_rank", 128)),
                block_size=self.block_size,
                mode=mixer_mode,
                retention_bias=float(
                    getattr(config, "prefix_state_retention_bias", 2.0)
                ),
                update_bias=float(
                    getattr(config, "prefix_state_update_bias", -1.0)
                ),
                gate_bias=float(getattr(config, "prefix_state_gate_bias", -2.0)),
                residual_scale=float(
                    getattr(config, "prefix_state_residual_scale", 0.1)
                ),
            )

        self.recall_correction: Optional[nn.Module] = None
        recall_rank = int(getattr(config, "recall_correction_rank", 0))
        if recall_rank > 0:
            if self.markov_head is None:
                raise ValueError("recall correction requires a Markov/CARH head")
            self.recall_correction = LowRankRecallCorrection(
                hidden_size=int(config.hidden_size),
                predecessor_rank=int(self.markov_head.markov_rank),
                correction_rank=recall_rank,
                block_size=self.block_size,
                gate_bias=float(
                    getattr(config, "recall_correction_gate_bias", -2.0)
                ),
            )
            # HF post_init has already run in DFlashDraftModel.__init__.  Keep
            # the new module an exact no-op when expanding a warm-start model.
            nn.init.zeros_(self.recall_correction.output_proj.weight)

        self.parallel_refiner: Optional[nn.Module] = None
        self.parallel_refiner_runtime_enabled = bool(
            getattr(config, "parallel_refiner_runtime_enabled", True)
        )
        refiner_rank = int(getattr(config, "parallel_refiner_rank", 0))
        refiner_steps = int(getattr(config, "parallel_refiner_steps", 0))
        if bool(refiner_rank) != bool(refiner_steps):
            raise ValueError(
                "parallel_refiner_rank and parallel_refiner_steps must be "
                "enabled together"
            )
        if refiner_steps not in {0, 1, 2}:
            raise ValueError("parallel_refiner_steps must be 0, 1, or 2")
        if refiner_rank > 0:
            if self.markov_head is None:
                raise ValueError("parallel refinement requires a Markov/CARH head")
            self.parallel_refiner = HazardAdaptiveParallelRefiner(
                hidden_size=int(config.hidden_size),
                predecessor_rank=int(self.markov_head.markov_rank),
                refinement_rank=refiner_rank,
                block_size=self.block_size,
                max_steps=refiner_steps,
                gate_bias=float(
                    getattr(config, "parallel_refiner_gate_bias", -1.0)
                ),
                second_step_bias=float(
                    getattr(config, "parallel_refiner_second_step_bias", -1.0)
                ),
                residual_scale=float(
                    getattr(config, "parallel_refiner_residual_scale", 0.1)
                ),
                temperature_conditioned_advantage=bool(
                    getattr(
                        config,
                        "refiner_advantage_temperature_conditioned",
                        False,
                    )
                ),
            )
            nn.init.zeros_(self.parallel_refiner.output_proj.weight)

        self.candidate_selector: Optional[nn.Module] = None
        self.selector_runtime_enabled = bool(
            getattr(config, "selector_runtime_enabled", True)
        )
        self.selector_margin_threshold = float(
            getattr(config, "selector_margin_threshold", 0.0)
        )
        if self.selector_margin_threshold < 0:
            raise ValueError("selector_margin_threshold must be non-negative")
        selector_rank = int(getattr(config, "selector_rank", 0))
        selector_top_k = int(getattr(config, "selector_top_k", 0))
        if bool(selector_rank) != bool(selector_top_k):
            raise ValueError(
                "DSpark selector_rank and selector_top_k must be enabled together"
            )
        if selector_rank > 0:
            self.candidate_selector = CandidateSelector(
                hidden_size=int(config.hidden_size),
                vocab_size=int(config.vocab_size),
                state_rank=selector_rank,
                top_k=selector_top_k,
                initializer_range=float(config.initializer_range),
            )
            # Zero preserves the legacy behavior (selector always active).
            # A positive threshold protects confident unary Top-1 decisions and
            # only enables conditional reranking when the Top-1/Top-2 margin is
            # smaller than the configured value.
            self.candidate_selector.margin_threshold = (
                self.selector_margin_threshold
            )

        self.confidence_head: Optional[nn.Module] = None
        if getattr(config, "enable_confidence_head", False):
            conf_input_dim = config.hidden_size
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "confidence_head_with_markov=True requires a Markov head "
                        "(markov_rank > 0)."
                    )
                conf_input_dim += self.markov_rank
            self.confidence_head = AcceptRatePredictor(conf_input_dim)

    def set_local_transition_progress(
        self, global_step: int, total_steps: int
    ) -> float:
        """Decay the old convolution while LTA learns its replacement."""
        if not self.local_transition_enabled:
            return 1.0
        progress = float(global_step) / max(float(total_steps), 1.0)
        start = self.local_transition_conv_decay_start_ratio
        end = self.local_transition_conv_decay_end_ratio
        if progress <= start:
            scale = 1.0
        elif progress >= end:
            scale = self.local_transition_final_conv_scale
        else:
            fraction = (progress - start) / (end - start)
            scale = 1.0 + fraction * (
                self.local_transition_final_conv_scale - 1.0
            )
        for layer in self.layers:
            if hasattr(layer, "set_dynamic_conv_scale"):
                layer.set_dynamic_conv_scale(float(scale))
        method_config = dict(getattr(self.config, "dflash_config", None) or {})
        method_config["local_transition_conv_scale"] = float(scale)
        self.config.dflash_config = method_config
        return float(scale)

    def survival_gate_logits(self):
        """Return the latest per-layer gate logits retained for supervision."""
        logits = []
        for layer in self.layers:
            for conv in (
                getattr(layer, "attention_conv", None),
                getattr(layer, "mlp_conv", None),
            ):
                value = getattr(conv, "_last_gate_logits", None)
                if value is not None:
                    logits.append(value)
        return logits

    def apply_recall_correction(
        self,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        depth_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.recall_correction is None:
            return hidden_states
        predecessor = self.markov_head.get_prev_embeddings(predecessor_ids)
        return self.recall_correction(hidden_states, predecessor, depth_ids)

    def apply_prefix_state_mixer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.prefix_state_mixer is None:
            return hidden_states
        return self.prefix_state_mixer(hidden_states)

    def apply_block_summary(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.block_summary is None:
            return hidden_states
        return self.block_summary(hidden_states)

    def prefix_state_survival_logits(self) -> Optional[torch.Tensor]:
        if self.prefix_state_mixer is None:
            return None
        return self.prefix_state_mixer._last_survival_logits

    def prefix_state_diagnostics(self) -> dict[str, torch.Tensor]:
        if self.prefix_state_mixer is None:
            return {}
        return self.prefix_state_mixer._last_diagnostics

    def apply_parallel_refiner(
        self,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        iteration_idx: int,
        rollout_temperatures: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.parallel_refiner is None:
            raise RuntimeError("parallel refiner is not enabled")
        predecessor = self.markov_head.get_prev_embeddings(predecessor_ids)
        return self.parallel_refiner(
            hidden_states,
            predecessor,
            int(iteration_idx),
            rollout_temperatures=rollout_temperatures,
        )

    def parallel_refiner_diagnostics(self) -> dict[str, torch.Tensor]:
        if self.parallel_refiner is None:
            return {}
        return self.parallel_refiner._last_diagnostics
