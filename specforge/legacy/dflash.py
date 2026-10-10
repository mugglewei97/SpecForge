from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class MarkovScaffold(nn.Module):
    """Low-rank learned bigram bias for the train-time Markov Scaffold.

    Identical architecture to :class:`VanillaMarkov` but named
    ``markov_scaffold`` to avoid confusion with the serving-time
    ``markov_head`` (DSpark).  The parameter-name prefix allows
    :class:`BF16Optimizer` to assign a separate LR scale.

    At initialization the projection (``markov_w2``) is zero-filled so that
    the Markov bias starts at zero:
        ``markov_w2(markov_w1(token_ids)) == 0`` for any input.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)
        # Zero-init: at init, bias = w2(w1(x)) = 0, so corrected == base.
        nn.init.zeros_(self.markov_w2.weight)

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return additive bigram bias [same-shape-as-input, vocab_size]."""
        return self.markov_w2(self.markov_w1(token_ids.long()))


class CrossDepthMixerLayer(nn.Module):
    """Lightweight communication across parallel draft-depth slots.

    DFlash predicts every slot in a block in parallel.  This layer lets those
    latent slot representations exchange information without consuming any
    sampled token, so training and serving keep the same one-pass contract.
    Learnable residual gates start at ``gate_init``; zero therefore recovers
    the exact pre-mixer model at initialization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size, bias=False),
        )
        # FSDP cannot flatten scalar parameters.  Keep each gate as a
        # one-element vector; broadcasting preserves the exact same math.
        self.attn_gate = nn.Parameter(torch.tensor([float(gate_init)]))
        self.ffn_gate = nn.Parameter(torch.tensor([float(gate_init)]))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.attn_norm(hidden_states)
        mixed, _ = self.attn(normed, normed, normed, need_weights=False)
        hidden_states = hidden_states + torch.tanh(self.attn_gate) * mixed
        hidden_states = hidden_states + torch.tanh(self.ffn_gate) * self.ffn(
            self.ffn_norm(hidden_states)
        )
        return hidden_states


class CrossDepthMixer(nn.Module):
    """Apply depth mixing independently to every parallel draft block."""

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.block_size = int(block_size)
        self.layers = nn.ModuleList(
            [
                CrossDepthMixerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    gate_init=gate_init,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if sequence_length % self.block_size != 0:
            raise ValueError(
                "cross-depth mixer requires the query length to be divisible "
                f"by block_size={self.block_size}; got {sequence_length}"
            )
        num_blocks = sequence_length // self.block_size
        mixed = hidden_states.reshape(
            batch_size * num_blocks, self.block_size, hidden_size
        )
        for layer in self.layers:
            mixed = layer(mixed)
        return mixed.reshape(batch_size, sequence_length, hidden_size)


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    decoder_layer_class = Qwen3DFlashDecoderLayer

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        dflash_config = getattr(config, "dflash_config", {}) or {}
        block_size = getattr(config, "block_size", None)
        if block_size is None:
            block_size = dflash_config.get("block_size")
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise ValueError(
                "DFlash config must define an integer block_size either at "
                "config.block_size or config.dflash_config.block_size"
            )
        self.block_size = block_size
        self.layers = nn.ModuleList(
            [
                self._build_decoder_layer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.target_layer_fusion_mode = str(
            dflash_config.get("target_layer_fusion_mode", "none")
        )
        self.target_layer_fusion_source_dropout = float(
            dflash_config.get("target_layer_fusion_source_dropout", 0.0)
        )
        if not 0 <= self.target_layer_fusion_source_dropout <= 1:
            raise ValueError("target_layer_fusion_source_dropout must be in [0, 1]")
        if self.target_layer_fusion_source_dropout and (
            self.target_layer_fusion_mode != "shared-low-rank" or len(self.target_layer_ids) < 2
        ):
            raise ValueError("source dropout requires shared-low-rank and at least two sources")
        if self.target_layer_fusion_mode not in {"none", "per-draft-layer", "shared-low-rank", "shared-low-rank-delta", "static-channel", "shared-low-rank-grouped", "shared-low-rank-token"}:
            raise ValueError(
                "target_layer_fusion_mode must be none, per-draft-layer or shared-low-rank"
            )
        self.target_layer_fusion_residual_scale = float(
            dflash_config.get("target_layer_fusion_residual_scale", 0.1)
        )
        if self.target_layer_fusion_residual_scale < 0:
            raise ValueError("target layer fusion residual scale must be non-negative")
        if self.target_layer_fusion_mode == "static-channel":
            self.target_layer_fusion_channel = nn.Parameter(
                torch.zeros(config.num_hidden_layers, config.hidden_size)
            )
        if self.target_layer_fusion_mode in {"shared-low-rank", "shared-low-rank-delta", "shared-low-rank-grouped", "shared-low-rank-token"}:
            rank = int(dflash_config.get("target_layer_fusion_rank", 128))
            if not 1 <= rank <= config.hidden_size:
                raise ValueError("target_layer_fusion_rank must be in [1, hidden_size]")
            self.target_layer_fusion_weights = nn.Parameter(
                torch.zeros(config.num_hidden_layers, len(self.target_layer_ids))
            )
            self.target_layer_fusion_down = nn.Linear(config.hidden_size, rank, bias=False)
            self.target_layer_fusion_up = nn.Linear(rank, config.hidden_size, bias=False)
            if self.target_layer_fusion_mode == "shared-low-rank-grouped":
                if rank % 8:
                    raise ValueError("grouped fusion rank must be divisible by 8")
                self.target_layer_fusion_weights = nn.Parameter(
                    torch.zeros(config.num_hidden_layers, 8, len(self.target_layer_ids))
                )
            if self.target_layer_fusion_mode == "shared-low-rank-token":
                self.target_layer_fusion_scorer = nn.Linear(rank, 1, bias=False)
            if self.target_layer_fusion_mode == "shared-low-rank-delta":
                self.target_layer_fusion_delta = nn.ModuleList([
                    nn.Linear(rank, rank, bias=False) for _ in range(config.num_hidden_layers)
                ])
        elif self.target_layer_fusion_mode == "per-draft-layer":
            num_sources = len(self.target_layer_ids)
            self.target_layer_fusion_weights = nn.Parameter(
                torch.zeros(config.num_hidden_layers, num_sources)
            )
            self.target_layer_fusion_proj = nn.ModuleList(
                [
                    nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                    for _ in range(config.num_hidden_layers)
                ]
            )
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)

        cross_depth_mixer_layers = int(
            dflash_config.get("cross_depth_mixer_layers", 0)
        )
        cross_depth_mixer_heads = int(
            dflash_config.get("cross_depth_mixer_heads", 2)
        )
        cross_depth_mixer_dropout = float(
            dflash_config.get("cross_depth_mixer_dropout", 0.0)
        )
        cross_depth_mixer_gate_init = float(
            dflash_config.get("cross_depth_mixer_gate_init", 0.0)
        )
        if cross_depth_mixer_layers < 0:
            raise ValueError("cross_depth_mixer_layers must be non-negative")
        if cross_depth_mixer_heads <= 0:
            raise ValueError("cross_depth_mixer_heads must be positive")
        if config.hidden_size % cross_depth_mixer_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by cross_depth_mixer_heads"
            )
        if not 0.0 <= cross_depth_mixer_dropout < 1.0:
            raise ValueError("cross_depth_mixer_dropout must be in [0, 1)")
        self.cross_depth_mixer_layers = cross_depth_mixer_layers
        if cross_depth_mixer_layers > 0:
            self.cross_depth_mixer = CrossDepthMixer(
                hidden_size=config.hidden_size,
                block_size=self.block_size,
                num_layers=cross_depth_mixer_layers,
                num_heads=cross_depth_mixer_heads,
                dropout=cross_depth_mixer_dropout,
                gate_init=cross_depth_mixer_gate_init,
            )

        if self.projector_type == "domino":
            self.emb_dim = dflash_config["emb_dim"]
            self.gru_hidden_dim = dflash_config["gru_hidden_dim"]
            self.prefix_gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=self.gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            in_dim = config.hidden_size + self.gru_hidden_dim
            self.embed_proj = nn.Sequential(
                nn.Linear(in_dim, self.emb_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.emb_dim, config.vocab_size, bias=False),
            )
        elif self.projector_type == "dspark":
            # DSpark projector heads (Markov, confidence) are created by
            # DSparkDraftModel.__init__ after super().__init__() returns.
            pass
        elif self.projector_type is not None:
            raise ValueError(f"Unknown draft projector_type: {self.projector_type}")

        # Markov Scaffold (train-time corrective bias, NOT the serving Markov head).
        # When markov_scaffold_rank > 0, a low-rank bigram bias module is
        # created.  It is only used by the training wrapper (OnlineDFlashModel)
        # and does NOT participate in forward() or spec_generate().
        markov_scaffold_rank = int(dflash_config.get("markov_scaffold_rank", 0))
        self.markov_scaffold_rank = markov_scaffold_rank
        if markov_scaffold_rank > 0:
            self.markov_scaffold = MarkovScaffold(
                vocab_size=config.vocab_size,
                markov_rank=markov_scaffold_rank,
            )

        self._init_draft_head(config, dflash_config)

        self.post_init()

        if self.target_layer_fusion_mode == "per-draft-layer":
            # Zero residual initialization; the extra RMSNorm still changes
            # the shared path, so this is not strict warm-start equivalence.
            for projection in self.target_layer_fusion_proj:
                nn.init.zeros_(projection.weight)
        elif self.target_layer_fusion_mode in {"shared-low-rank", "shared-low-rank-delta", "shared-low-rank-grouped", "shared-low-rank-token"}:
            # Keep down random: zeroing both factors would block learning.
            nn.init.zeros_(self.target_layer_fusion_up.weight)
            if self.target_layer_fusion_mode == "shared-low-rank-token":
                nn.init.zeros_(self.target_layer_fusion_scorer.weight)
            if self.target_layer_fusion_mode == "shared-low-rank-delta":
                for projection in self.target_layer_fusion_delta:
                    nn.init.zeros_(projection.weight)

        # Re-apply zero-init *after* post_init(), because HuggingFace's
        # post_init() re-initialises all nn.Linear weights (including
        # markov_w2), clobbering the zero-fill done in MarkovScaffold.__init__.
        if hasattr(self, "markov_scaffold"):
            nn.init.zeros_(self.markov_scaffold.markov_w2.weight)

    def _build_decoder_layer(
        self, config: Qwen3Config, layer_idx: int
    ) -> nn.Module:
        """Build one backbone layer; DFlash architecture variants override it."""

        return self.decoder_layer_class(config, layer_idx)

    def _init_draft_head(self, config: Qwen3Config, dflash_config: dict) -> None:
        """Optional architecture-specific head construction hook."""

        del config, dflash_config

    def _fusion_source_mask(self, batch_size, device):
        # One decision and one excluded source per sequence, reused across layers.
        if not self.training or self.target_layer_fusion_source_dropout == 0:
            return None
        count = len(self.target_layer_ids)
        drop = torch.rand(batch_size, device=device) < self.target_layer_fusion_source_dropout
        index = torch.randint(count, (batch_size,), device=device)
        return drop[:, None] & (torch.arange(count, device=device)[None, :] == index[:, None])

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        semantic_enabled = kwargs.get("source_semantic_codes") is not None
        semantic_numerators = []
        raw_target_hidden = target_hidden
        target_hidden = self.hidden_norm(self.fc(raw_target_hidden))
        target_sources = None
        if self.target_layer_fusion_mode in {"per-draft-layer", "shared-low-rank", "shared-low-rank-delta", "shared-low-rank-grouped", "shared-low-rank-token"}:
            expected = len(self.target_layer_ids) * self.config.hidden_size
            if raw_target_hidden.size(-1) != expected:
                raise ValueError(
                    "per-draft-layer target fusion expected final dimension "
                    f"{expected}, got {raw_target_hidden.size(-1)}"
                )
            target_sources = raw_target_hidden.reshape(
                *raw_target_hidden.shape[:-1],
                len(self.target_layer_ids),
                self.config.hidden_size,
            )
            if self.target_layer_fusion_mode in {"shared-low-rank", "shared-low-rank-delta", "shared-low-rank-grouped", "shared-low-rank-token"}:
                # Project each source once, outside the draft-layer loop.
                target_sources = self.target_layer_fusion_down(target_sources)
        source_mask = self._fusion_source_mask(raw_target_hidden.shape[0], raw_target_hidden.device)
        source_scores = None
        if self.target_layer_fusion_mode == "shared-low-rank-token":
            # Pointwise context-only scoring, once per forward, shared across layers.
            source_scores = self.target_layer_fusion_scorer(torch.tanh(target_sources)).squeeze(-1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            layer_target_hidden = target_hidden
            if self.target_layer_fusion_mode == "static-channel":
                gain = 1 + 0.1 * torch.tanh(self.target_layer_fusion_channel[layer_idx])
                layer_target_hidden = target_hidden * gain.to(target_hidden.dtype)
            if target_sources is not None:
                logits = self.target_layer_fusion_weights[layer_idx]
                if self.target_layer_fusion_mode == "shared-low-rank-grouped":
                    weights = logits.softmax(-1).to(target_sources.dtype)
                    grouped = target_sources.reshape(*target_sources.shape[:-1], 8, target_sources.shape[-1] // 8)
                    fused_source = torch.einsum("...sgr,gs->...gr", grouped, weights).flatten(-2)
                elif source_scores is not None:
                    weights = (logits + source_scores).softmax(-1).to(target_sources.dtype)
                    fused_source = (target_sources * weights.unsqueeze(-1)).sum(-2)
                elif source_mask is None:
                    weights = torch.softmax(logits, dim=-1).to(target_sources.dtype)
                    fused_source = torch.einsum("...ld,l->...d", target_sources, weights)
                else:
                    # Mask logits before softmax: remaining sources sum to one.
                    weights = torch.softmax(logits[None, :].masked_fill(source_mask, -torch.inf), dim=-1)
                    fused_source = torch.einsum("b...ld,bl->b...d", target_sources, weights.to(target_sources.dtype))
                if self.target_layer_fusion_mode == "shared-low-rank-delta":
                    fused_source = fused_source + 0.1 * self.target_layer_fusion_delta[layer_idx](
                        torch.tanh(fused_source)
                    )
                if self.target_layer_fusion_mode in {"shared-low-rank", "shared-low-rank-delta", "shared-low-rank-grouped", "shared-low-rank-token"}:
                    residual = self.target_layer_fusion_up(fused_source)
                else:
                    residual = self.target_layer_fusion_proj[layer_idx](fused_source)
                layer_target_hidden = self.hidden_norm(
                    target_hidden
                    + self.target_layer_fusion_residual_scale * residual
                )
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=layer_target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if semantic_enabled:
                hidden_states, semantic_num = hidden_states
                semantic_numerators.append(semantic_num)
        if hasattr(self, "cross_depth_mixer"):
            hidden_states = self.cross_depth_mixer(hidden_states)
        output = self.norm(hidden_states)
        if semantic_enabled:
            # Average across layers, not sum: changing depth should not change
            # the semantic loss coefficient implicitly.
            return output, torch.stack(semantic_numerators).mean()
        return output

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids
