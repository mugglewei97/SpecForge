"""DFlash 2 draft architecture.

DFlash 2 keeps the one-pass DFlash backbone and adds two checkpoint-driven
components used by the serving implementation:

* a grouped, dynamic depthwise convolution around every attention and MLP;
* a low-rank candidate selector that re-ranks the target head's top-k tokens.

The module and parameter names intentionally match SGLang's
``DFlash2DraftModel`` so a normal Hugging Face export can be served directly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from .dflash import DFlashDraftModel, Qwen3DFlashDecoderLayer


from .neighbor_residual import NeighborResidual
from .light_conv_kernel import LightConvKernelProjection


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise convolution within each proposal block.

    By default one input projection produces both kernels. ``output`` uses
    the same projection's two halves on the respective sublayer input/output.
    ``source-aware`` adds a block-causal low-rank source to kernel generation,
    not to the hidden states. Both experiments retain unrestricted legacy taps.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
        mode: str = "legacy",
        apply_input: bool = True,
        apply_output: bool = True,
        residual_scale: float = 0.1,
        gate_bias: float = 2.0,
        freeze_identity: bool = False,
        kernel_conditioning: str = "input",
        source_rank: int = 32,
    ) -> None:
        super().__init__()
        if taps < 1:
            raise ValueError("DFlash2 conv_kernel_size must be >= 1")
        if taps > block_size:
            raise ValueError(
                "DFlash2 conv_kernel_size must not exceed block_size, got "
                f"conv_kernel_size={taps}, block_size={block_size}"
            )
        if group_size < 1 or hidden_size % group_size:
            raise ValueError(
                f"DFlash2 conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}"
            )

        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size
        if mode not in {"legacy", "survival-gated", "neighbor-bounded", "prefix-risk-bounded"}:
            raise ValueError(
                "DFlash2 convolution mode must be legacy or survival-gated, "
                f"got {mode!r}"
            )
        if not apply_input and not apply_output:
            raise ValueError("DFlash2 convolution must enable an input or output side")
        if mode == "survival-gated" and self.taps < 2:
            raise ValueError("survival-gated convolution requires at least two taps")
        if residual_scale < 0:
            raise ValueError("convolution residual_scale must be non-negative")
        self.mode = mode
        if kernel_conditioning not in {"input", "output", "source-aware", "grouped16", "grouped64", "low-rank64", "static"}:
            raise ValueError(
                f"Unsupported conv kernel_conditioning={kernel_conditioning!r}"
            )
        if kernel_conditioning != "input" and mode != "legacy":
            raise ValueError(
                "Alternative kernel conditioning requires legacy convolution"
            )
        if source_rank < 1:
            raise ValueError("conv source_rank must be positive")
        self.kernel_conditioning = kernel_conditioning
        self.source_rank = int(source_rank)
        self.apply_input = bool(apply_input)
        self.apply_output = bool(apply_output)
        self.residual_scale = float(residual_scale)
        self.gate_bias = float(gate_bias)
        self.neighbor = None
        if mode in {"neighbor-bounded", "prefix-risk-bounded"}:
            if taps != 2:
                raise ValueError("Neighbor residual requires exactly two taps")
            self.neighbor = NeighborResidual(
                int(hidden_size), group_size, block_size,
                risk=mode == "prefix-risk-bounded",
            )
        self._last_gate_logits = None

        # [input/output side, tap, channel], matching SGLang's loader contract.
        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(
            base_kernel, requires_grad=not bool(freeze_identity) and self.neighbor is None
        )
        if kernel_conditioning in {"grouped16", "grouped64", "low-rank64", "static"}:
            self.kernel_projection = LightConvKernelProjection(
                int(hidden_size), self.taps, self.group_size, kernel_conditioning
            )
        else:
            self.kernel_projection = nn.Linear(
                int(hidden_size), 2 * self.taps * self.num_groups, bias=False
            )
        self.source_down = self.source_up = None
        if kernel_conditioning == "source-aware":
            self.source_down = nn.Linear(int(hidden_size), self.source_rank, bias=False)
            self.source_up = nn.Linear(
                self.source_rank, 2 * self.taps * self.num_groups, bias=False
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the exact DFlash identity used by the reference model."""

        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            if self.kernel_projection.weight is not None:
                self.kernel_projection.weight.zero_()
            if self.source_up is not None:
                self.source_up.weight.zero_()

    def _input_coefficients(self, hidden_states: torch.Tensor) -> torch.Tensor:
        coefficients = self.kernel_projection(hidden_states)
        if self.source_down is not None:
            batch, length, _ = hidden_states.shape
            if length % self.block_size:
                raise ValueError("Source-aware convolution requires complete draft blocks")
            sources = self.source_down(hidden_states).reshape(
                batch, length // self.block_size, self.block_size, self.source_rank
            )
            # Shift in the small rank space, independently inside each block.
            previous = F.pad(sources[:, :, :-1], (0, 0, 1, 0))
            coefficients = coefficients + self.source_up(previous).reshape_as(
                coefficients
            )
        return coefficients.reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )

    def _side_coefficients(self, hidden_states: torch.Tensor, side: int) -> torch.Tensor:
        width = self.taps * self.num_groups
        return F.linear(
            hidden_states,
            self.kernel_projection.weight[side * width : (side + 1) * width],
        ).reshape(*hidden_states.shape[:-1], self.taps, self.num_groups)

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        *,
        side: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if sequence_length % self.block_size:
            raise ValueError(
                "DFlash2 convolution sequence length must be divisible by "
                f"block_size={self.block_size}, got {sequence_length}"
            )

        num_blocks = sequence_length // self.block_size
        blocks = hidden_states.reshape(
            batch_size,
            num_blocks,
            self.block_size,
            self.num_groups,
            self.group_size,
        )
        dynamic = delta.reshape(
            batch_size,
            num_blocks,
            self.block_size,
            self.taps,
            self.num_groups,
        )
        base = self.base_kernel[side].reshape(
            1,
            1,
            1,
            self.taps,
            self.num_groups,
            self.group_size,
        )
        if self.mode == "survival-gated":
            # Reuse tap zero as a per-group prefix-survival gate.  The actual
            # self tap is fixed to the identity, while only causal lag taps
            # contribute a bounded residual.  Keeping the projection shape
            # unchanged preserves the DFlash2 checkpoint/loading contract.
            gate_logits = dynamic[..., 0, :] + self.gate_bias
            gate = torch.sigmoid(gate_logits).unsqueeze(-1)
            self._last_gate_logits = gate_logits
            output = blocks
            for tap in range(1, self.taps):
                shifted = F.pad(
                    blocks[:, :, : self.block_size - tap],
                    (0, 0, 0, 0, tap, 0),
                )
                lag = torch.tanh(dynamic[..., tap, :]).unsqueeze(-1)
                output = output + self.residual_scale * gate * lag * shifted
            return output.reshape(batch_size, sequence_length, hidden_size)

        coefficients = base + dynamic.unsqueeze(-1)

        output = coefficients[:, :, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(
                blocks[:, :, : self.block_size - tap],
                (0, 0, 0, 0, tap, 0),
            )
            output = output + coefficients[:, :, :, tap] * shifted
        return output.reshape(batch_size, sequence_length, hidden_size)

    def prepare(self, hidden_states: torch.Tensor):
        if self.kernel_conditioning == "output":
            prepared = hidden_states
            if self.apply_input:
                prepared = self._convolve(
                    hidden_states, self._side_coefficients(hidden_states, 0), side=0
                )
            # The output kernel must be generated later, from actual output.
            return prepared, None
        if self.neighbor is not None:
            coefficients = self.kernel_projection(hidden_states).reshape(
                *hidden_states.shape[:-1], 2, self.taps, self.num_groups
            )
            gates = self.neighbor.coefficients(hidden_states)
            prepared = hidden_states
            if self.apply_input:
                prepared = self.neighbor.mix(hidden_states, coefficients[..., 0, :, :],
                                             gates[..., 0, :], self.residual_scale)
            return prepared, (coefficients[..., 1, :, :], gates[..., 1, :])
        coefficients = self._input_coefficients(hidden_states)
        prepared = hidden_states
        if self.apply_input:
            prepared = self._convolve(
                hidden_states, coefficients[..., 0, :, :], side=0
            )
        return prepared, coefficients[..., 1, :, :]

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.apply_output:
            return hidden_states
        if self.kernel_conditioning == "output":
            coefficients = self._side_coefficients(hidden_states, 1)
        if self.neighbor is not None:
            delta, gate = coefficients
            return self.neighbor.mix(hidden_states, delta, gate, self.residual_scale)
        return self._convolve(hidden_states, coefficients, side=1)


class LocalTransitionAttention(nn.Module):
    """Low-rank causal attention over the verified anchor and local transitions.

    Position zero of every DFlash block is the last verified token.  Every
    proposal position attends to that anchor, itself, and up to ``window``
    predecessors.  The source set is fixed and tiny, so all positions remain
    parallel and the operator is substantially cheaper than another full
    self-attention layer.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        block_size: int,
        rank: int,
        num_heads: int,
        window: int,
        residual_scale: float,
        gate_bias: float,
    ) -> None:
        super().__init__()
        if rank < 1 or num_heads < 1 or rank % num_heads:
            raise ValueError("local transition rank must be divisible by heads")
        if window < 1 or window >= block_size:
            raise ValueError("local transition window must be in [1, block_size)")
        if residual_scale < 0:
            raise ValueError("local transition residual scale must be non-negative")
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.num_heads = int(num_heads)
        self.head_dim = self.rank // self.num_heads
        self.window = int(window)
        self.residual_scale = float(residual_scale)
        self.num_sources = 2 + self.window  # anchor, self, lag-1 ... lag-window

        self.q_proj = nn.Linear(hidden_size, rank, bias=False)
        self.k_proj = nn.Linear(hidden_size, rank, bias=False)
        self.v_proj = nn.Linear(hidden_size, rank, bias=False)
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.output_proj = nn.Linear(rank, hidden_size, bias=False)
        self.source_bias = nn.Parameter(torch.zeros(num_heads, self.num_sources))
        self.depth_bias = nn.Parameter(
            torch.zeros(block_size, num_heads, self.num_sources)
        )
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))
        # Exact warm-start equivalence: LTA begins as a zero residual branch.
        nn.init.zeros_(self.output_proj.weight)

    def reset_output_projection(self) -> None:
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if sequence_length % self.block_size:
            raise ValueError(
                "local transition sequence length must be divisible by "
                f"block_size={self.block_size}, got {sequence_length}"
            )
        num_blocks = sequence_length // self.block_size
        blocks = hidden_states.reshape(
            batch_size, num_blocks, self.block_size, hidden_size
        )
        anchor = blocks[:, :, :1].expand(-1, -1, self.block_size, -1)
        sources = [anchor, blocks]
        valid = torch.ones(
            self.block_size,
            self.num_sources,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        for lag in range(1, self.window + 1):
            shifted = F.pad(blocks[:, :, : self.block_size - lag], (0, 0, lag, 0))
            sources.append(shifted)
            valid[:lag, 1 + lag] = False
        source_states = torch.stack(sources, dim=3)

        query = self.q_proj(blocks).reshape(
            batch_size, num_blocks, self.block_size, self.num_heads, self.head_dim
        )
        key = self.k_proj(source_states).reshape(
            batch_size,
            num_blocks,
            self.block_size,
            self.num_sources,
            self.num_heads,
            self.head_dim,
        )
        value = self.v_proj(source_states).reshape_as(key)
        scores = torch.einsum("bnqhd,bnqshd->bnqhs", query, key)
        scores = scores * (self.head_dim**-0.5)
        scores = scores + self.source_bias.view(1, 1, 1, self.num_heads, -1)
        # depth_bias is already [position, head, source], matching the final
        # three score dimensions [position, head, source].  Keeping that
        # layout matters when num_heads != num_sources (for example H8/W3).
        scores = scores + self.depth_bias.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(
            ~valid.view(1, 1, self.block_size, 1, self.num_sources),
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        mixed = torch.einsum("bnqhs,bnqshd->bnqhd", weights, value)
        gate = torch.sigmoid(self.gate_proj(blocks)).unsqueeze(-1)
        mixed = (mixed * gate).reshape(
            batch_size, num_blocks, self.block_size, self.rank
        )
        update = self.output_proj(mixed).reshape(
            batch_size, sequence_length, hidden_size
        )
        return hidden_states + self.residual_scale * update


class Qwen3DFlash2DecoderLayer(Qwen3DFlashDecoderLayer):
    """Qwen3 DFlash layer with DFlash2 convolutional sublayer wrappers."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        attention_conv: Optional[DFlashGroupedConv],
        mlp_conv: Optional[DFlashGroupedConv],
        local_transition_attention: Optional[LocalTransitionAttention] = None,
        dynamic_conv_scale: float = 1.0,
    ) -> None:
        super().__init__(config, layer_idx)
        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv
        self.local_transition_attention = local_transition_attention
        self.dynamic_conv_scale = float(dynamic_conv_scale)
        self.register_buffer(
            "_dynamic_conv_scale",
            torch.tensor(float(dynamic_conv_scale), dtype=torch.float32),
            persistent=False,
        )

    def set_dynamic_conv_scale(self, scale: float) -> None:
        self.dynamic_conv_scale = float(scale)
        self._dynamic_conv_scale.fill_(float(scale))

    def _prepare_conv(self, hidden_states, conv):
        if conv is None:
            return hidden_states, None
        prepared, kernel = conv.prepare(hidden_states)
        scale = self._dynamic_conv_scale.to(dtype=hidden_states.dtype)
        prepared = hidden_states + scale * (prepared - hidden_states)
        return prepared, kernel

    def _finish_conv(self, hidden_states, kernel, conv):
        # Output-conditioned kernels are intentionally deferred (kernel=None).
        if conv is None:
            return hidden_states
        finished = conv.finish(hidden_states, kernel)
        scale = self._dynamic_conv_scale.to(dtype=hidden_states.dtype)
        finished = hidden_states + scale * (finished - hidden_states)
        return finished

    def forward(self, **kwargs):
        # Explicit auxiliary return crosses checkpoint/FSDP boundaries. Do not
        # cache a graph-bearing activation on the module or pass labels to attn.
        semantic_codes = kwargs.pop("source_semantic_codes", None)
        semantic_weights = kwargs.pop("source_semantic_weights", None)
        target_hidden = kwargs.get("target_hidden")
        hidden_states = kwargs.get("hidden_states")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_kernel = None
        hidden_states, attention_kernel = self._prepare_conv(
            hidden_states, self.attention_conv
        )
        if self.local_transition_attention is not None:
            hidden_states = self.local_transition_attention(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            past_key_values=kwargs.get("past_key_value"),
            output_attentions=kwargs.get("output_attentions", False),
            use_cache=kwargs.get("use_cache", False),
            cache_position=kwargs.get("cache_position"),
            position_embeddings=kwargs.get("position_embeddings"),
            **{
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "target_hidden",
                    "hidden_states",
                    "attention_mask",
                    "position_ids",
                    "past_key_value",
                    "output_attentions",
                    "use_cache",
                    "cache_position",
                    "position_embeddings",
                }
            },
        )[0]
        hidden_states = self._finish_conv(
            hidden_states, attention_kernel, self.attention_conv
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        semantic_num = None
        if semantic_codes is not None:
            if self.mlp_conv is None or self.mlp_conv.source_down is None:
                raise ValueError("Semantic supervision requires source-aware MLP conv")
            # Exactly the unshifted source that generates the next slot's
            # coefficients. Re-evaluate the small projection inside this FSDP
            # block for training only; the inference path remains untouched.
            source = self.mlp_conv.source_down(hidden_states).float()
            error = 1.0 - F.cosine_similarity(
                source, semantic_codes.detach().float(), dim=-1, eps=1e-6
            )
            semantic_num = (error * semantic_weights.float()).sum()
        mlp_kernel = None
        hidden_states, mlp_kernel = self._prepare_conv(hidden_states, self.mlp_conv)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self._finish_conv(hidden_states, mlp_kernel, self.mlp_conv)
        output = residual + hidden_states
        return output if semantic_num is None else (output, semantic_num)


class CandidateSelector(nn.Module):
    """Low-rank transition scorer used to select a coherent token path."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        if state_rank < 1:
            raise ValueError("DFlash2 selector_rank must be >= 1")
        if top_k < 1 or top_k > vocab_size:
            raise ValueError(
                "DFlash2 selector_top_k must be in [1, vocab_size], got "
                f"{top_k} for vocab_size={vocab_size}"
            )
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), int(state_rank))
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), int(state_rank))
        )
        self.hidden_projection = nn.Linear(
            int(hidden_size),
            int(state_rank),
            bias=False,
        )
        self.initializer_range = float(initializer_range)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the selector as an exact unary no-op.

        The successor factor starts at zero, so candidate scores initially
        equal unary logits. It learns first; gradients reach the remaining
        bilinear factors once the residual becomes non-zero.
        """

        nn.init.normal_(self.predecessor_codebook, std=self.initializer_range)
        nn.init.zeros_(self.successor_codebook)
        nn.init.normal_(self.hidden_projection.weight, std=self.initializer_range)

    def score_candidates(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add low-rank predecessor transitions to a candidate set's logits."""

        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)
        transition = torch.einsum("...r,...kr->...k", context, successor)
        return unary_logits + transition

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build SGLang's K-by-K transition lattice for every proposal slot."""

        predecessor_ids = torch.cat(
            [
                anchor_token_ids[:, None, None].expand(-1, 1, self.top_k),
                candidate_ids[:, :-1],
            ],
            dim=1,
        )
        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)[:, :, None]
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc",
            context,
            successor,
        )

    def greedy_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Walk the candidate lattice for the local greedy inference helper."""

        predecessor_ids = anchor_token_ids
        path = []
        for position in range(candidate_ids.shape[1]):
            scores = self.score_candidates(
                candidate_ids=candidate_ids[:, position],
                unary_logits=unary_logits[:, position],
                hidden_states=hidden_states[:, position],
                predecessor_ids=predecessor_ids,
            )
            selected = scores.argmax(dim=-1, keepdim=True)
            predecessor_ids = candidate_ids[:, position].gather(1, selected)[:, 0]
            path.append(predecessor_ids)
        return torch.stack(path, dim=1)


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash backbone with local convolution and candidate path selection."""

    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]
    decoder_layer_class = Qwen3DFlash2DecoderLayer

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)

        # DFlashDraftModel.post_init() reinitializes every nn.Linear. Reset the
        # dynamic projections afterwards so DFlash2 starts as an exact DFlash
        # backbone, as required by the reference implementation.
        for layer in self.layers:
            if not isinstance(layer, Qwen3DFlash2DecoderLayer):
                raise TypeError(
                    "DFlash2DraftModel expected Qwen3DFlash2DecoderLayer, got "
                    f"{type(layer).__name__}"
                )
            layer.attention_conv.reset_parameters()
            layer.mlp_conv.reset_parameters()

        # The selector is constructed before post_init() through the draft-head
        # hook. Restore its serving-aligned unary no-op initialization.
        self.candidate_selector.reset_parameters()

    def _dflash2_config(self) -> dict:
        return dict(getattr(self.config, "dflash_config", None) or {})

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ) -> nn.Module:
        method_config = dict(getattr(config, "dflash_config", None) or {})
        taps = method_config.get("conv_kernel_size")
        group_size = method_config.get("conv_group_size")
        if not isinstance(taps, int) or isinstance(taps, bool):
            raise ValueError(
                "DFlash2DraftModel requires dflash_config.conv_kernel_size"
            )
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise ValueError("DFlash2DraftModel requires dflash_config.conv_group_size")

        def grouped_conv() -> DFlashGroupedConv:
            return DFlashGroupedConv(
                hidden_size=int(config.hidden_size),
                block_size=self.block_size,
                taps=taps,
                group_size=group_size,
                kernel_conditioning=str(
                    method_config.get("conv_kernel_conditioning", "input")
                ),
                source_rank=int(method_config.get("conv_source_rank", 32)),
            )

        return self.decoder_layer_class(
            config,
            layer_idx,
            attention_conv=grouped_conv(),
            mlp_conv=grouped_conv(),
        )

    def _init_draft_head(self, config: Qwen3Config, dflash_config: dict) -> None:
        selector_rank = dflash_config.get("selector_rank")
        selector_top_k = dflash_config.get("selector_top_k")
        if not isinstance(selector_rank, int) or isinstance(selector_rank, bool):
            raise ValueError("DFlash2DraftModel requires dflash_config.selector_rank")
        if not isinstance(selector_top_k, int) or isinstance(selector_top_k, bool):
            raise ValueError("DFlash2DraftModel requires dflash_config.selector_top_k")
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=selector_rank,
            top_k=selector_top_k,
            initializer_range=float(config.initializer_range),
        )

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the public DFlash2 unary-logit transform used by SGLang."""

        method_config = self._dflash2_config()
        transformed = logits.float() * float(
            method_config.get("output_multiplier", 1.0)
        )
        softcap = method_config.get("final_logit_softcapping")
        if softcap is not None:
            softcap = float(softcap)
            if softcap <= 0:
                raise ValueError("DFlash2 final_logit_softcapping must be > 0")
            transformed = torch.tanh(transformed / softcap) * softcap
        return transformed

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        hidden = draft_hidden[:, -self.block_size + 1 :, :]
        unary_logits = self.transform_unary_logits(target.lm_head(hidden))
        unary_topk, candidate_ids = unary_logits.topk(
            self.candidate_selector.top_k,
            dim=-1,
        )
        return self.candidate_selector.greedy_path(
            candidate_ids=candidate_ids,
            unary_logits=unary_topk,
            hidden_states=hidden,
            anchor_token_ids=block_output_ids[:, 0],
        )


__all__ = [
    "CandidateSelector",
    "DFlash2DraftModel",
    "DFlashGroupedConv",
    "LocalTransitionAttention",
    "Qwen3DFlash2DecoderLayer",
]
