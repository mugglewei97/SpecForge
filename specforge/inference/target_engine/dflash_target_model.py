from abc import abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn

from specforge.sampling import processed_log_probs

from .base import TargetEngine
from .target_capture_policy import DFlashCapturePolicy, DFlashTargetOutput

# NOTE: the capture/load implementations live in
# ``target_capture_policy.DFlashCapturePolicy``, shared with the generic per-backend
# engines. The classes below keep the existing hierarchy and delegate.

_DFLASH = DFlashCapturePolicy()


class DFlashTargetEngine(TargetEngine):
    """DFlash target engine — the algorithm ABC over a frozen target backend.

    DFlash captures the concatenated hidden states of an arbitrary list of
    target layers (``set_capture_layers``) and trains on hard real-token labels,
    so — unlike EAGLE3 — there is no target distribution / vocab map. The generic
    :meth:`TargetEngine.capture` hook dispatches to ``generate_dflash_data``, so
    the extraction is byte-identical to the pre-Phase-B path.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetEngine":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training."""

    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> DFlashTargetOutput:
        """Generic extraction entry point (see :meth:`TargetEngine.capture`).

        Dispatches to the DFlash-specific ``generate_dflash_data``. DFlash takes
        no extra extraction kwargs, so any are ignored.
        """
        return self.generate_dflash_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        """Set which layers' hidden states to capture (TargetEngine hook)."""
        self.capture_layer_ids = layer_ids

    @abstractmethod
    def score_proposal_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the frozen target's greedy next-token id at every position."""

    @abstractmethod
    def score_proposal_distribution(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
        temperatures: Optional[torch.Tensor] = None,
        sampling_top_k: int = 0,
        sampling_top_p: float = 1.0,
        score_positions: Optional[torch.Tensor] = None,
    ):
        """Return greedy ids and a compact target Top-k distribution.

        The return value is ``(greedy_ids, topk_ids, topk_probs)`` with the
        sequence dimension preserved, or restricted to ``score_positions`` of
        shape ``[batch, depth]``. Keeping only Top-k probabilities bounds
        the memory retained by proposal-scored OPSC while still exposing the
        target modes that dominate rejection-sampling overlap.
        """


class SGLangDFlashTargetEngine(DFlashTargetEngine):

    backend = "sglang"

    def __init__(self, backend):  # backend: sglang_backend.SGLangCaptureBackend
        super().__init__()  # capture_layer_ids = None
        self._backend = backend

    @property
    def model_runner(self):
        """Kept for back-compat: the underlying sglang ModelRunner."""
        return self._backend.model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDFlashTargetEngine":
        # Lazy import so `import specforge` still works without the pinned sglang.
        from .sglang_backend import SGLangCaptureBackend

        backend = SGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **_DFLASH.spec.sglang_build_kwargs,
            **kwargs,
        )
        return cls(backend)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)  # records self.capture_layer_ids
        # Some target models expose set_eagle3_layers_to_capture; guard on it.
        self._backend.set_eagle3_capture_layers(layer_ids, if_supported=True)

    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        return _DFLASH.sglang_capture(
            self._backend, input_ids, attention_mask, loss_mask
        )

    def score_proposal_tokens(self, input_ids, attention_mask):
        return self._backend.score_next_token_ids(input_ids, attention_mask)

    def score_proposal_distribution(
        self,
        input_ids,
        attention_mask,
        top_k,
        temperatures=None,
        sampling_top_k=0,
        sampling_top_p=1.0,
        score_positions=None,
    ):
        return self._backend.score_next_token_distribution(
            input_ids,
            attention_mask,
            top_k,
            temperatures=temperatures,
            sampling_top_k=sampling_top_k,
            sampling_top_p=sampling_top_p,
            score_positions=score_positions,
        )


class HFDFlashTargetEngine(DFlashTargetEngine):

    backend = "hf"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetEngine":
        return cls(
            _DFLASH.hf_load(
                pretrained_model_name_or_path,
                torch_dtype,
                device,
                cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        )

    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        return _DFLASH.hf_capture(
            self.model, self.capture_layer_ids, input_ids, attention_mask, loss_mask
        )

    @torch.no_grad()
    def score_proposal_tokens(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        return outputs.logits.argmax(dim=-1)

    @torch.no_grad()
    def score_proposal_distribution(
        self,
        input_ids,
        attention_mask,
        top_k,
        temperatures=None,
        sampling_top_k=0,
        sampling_top_p=1.0,
        score_positions=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        logits = outputs.logits
        if score_positions is not None:
            if (
                score_positions.ndim != 2
                or score_positions.size(0) != logits.size(0)
                or score_positions.numel() == 0
            ):
                raise ValueError(
                    "score_positions must have shape [batch, depth] with depth > 0"
                )
            if (
                score_positions.min().item() < 0
                or score_positions.max().item() >= logits.size(1)
            ):
                raise ValueError("score_positions must lie within the target sequence")
            rows = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)
            positions = score_positions.to(device=logits.device, dtype=torch.long)
            logits = logits[rows, positions]
        if temperatures is None:
            temperatures = torch.ones(
                logits.size(0), device=logits.device, dtype=torch.float32
            )
        log_probs = processed_log_probs(
            logits,
            temperatures.to(device=logits.device, dtype=torch.float32),
            top_k=int(sampling_top_k),
            top_p=float(sampling_top_p),
        )
        k = min(int(top_k), logits.size(-1))
        topk_values, topk_ids = torch.topk(log_probs, k=k, dim=-1)
        topk_probs = torch.exp(topk_values)
        return topk_ids[..., 0], topk_ids, topk_probs


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetEngine:
    if backend == "sglang":
        return SGLangDFlashTargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


# --- Back-compat aliases (pre-Phase-B names) -------------------------------
DFlashTargetModel = DFlashTargetEngine
SGLangDFlashTargetModel = SGLangDFlashTargetEngine
HFDFlashTargetModel = HFDFlashTargetEngine
