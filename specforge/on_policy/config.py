"""Small, independent run contract; ordinary training recipes are unchanged."""

from typing import Literal

from pydantic import Field, model_validator

from specforge.config.schema import StrictConfigModel


class SamplingConfig(StrictConfigModel):
    temperature: float = Field(default=1.0, ge=1e-5, allow_inf_nan=False)
    top_k: int = -1
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    max_new_tokens: int = Field(default=256, gt=1)
    ignore_eos: bool = False
    stop_token_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sampling(self):
        # top_k=1 is a greedy request in SGLang, whose exact q has no gradient.
        if self.top_k != -1 and self.top_k < 2:
            raise ValueError(
                "on-policy TV requires stochastic sampling: top_k=-1 or >=2"
            )
        if any(token < 0 for token in self.stop_token_ids):
            raise ValueError("stop_token_ids must be nonnegative")
        return self


class ModelConfig(StrictConfigModel):
    target_model_path: str
    draft_checkpoint_path: str
    draft_model_config: str | None = None
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    cache_dir: str | None = None
    trust_remote_code: bool = False


class DataConfig(StrictConfigModel):
    train_data_path: str
    max_prompt_length: int = Field(default=2048, gt=0)
    # Optional combined prompt + generated sequence limit (legacy --max-length).
    max_length: int | None = Field(default=None, gt=2)
    chat_template: Literal["qwen"] | None = None
    chat_template_kwargs: dict = Field(default_factory=dict)

    @property
    def prompt_limit(self):
        # Leave room for the initial anchor and at least one candidate.
        return (
            min(self.max_prompt_length, self.max_length - 2)
            if self.max_length
            else self.max_prompt_length
        )


class TrainingConfig(StrictConfigModel):
    output_dir: str
    max_steps: int | None = Field(default=None, gt=0)
    num_epochs: int | None = Field(default=None, gt=0)
    # Global prompt budget, independent of rank/block/token count. Samples with
    # no candidate positions are excluded from the actual loss denominator.
    batch_size: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=1e-6, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    max_grad_norm: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    warmup_ratio: float = Field(default=0.0, ge=0, le=1)
    seed: int = 42
    dist_timeout: int = Field(default=60, gt=0)
    fsdp_sharding: Literal["FULL_SHARD", "SHARD_GRAD_OP"] = "FULL_SHARD"
    # Different CUDA kernels have rounding differences; fail closed on drift.
    replay_max_tv: float = Field(default=0.2, gt=0)
    max_empty_samples: int = Field(default=100, gt=0)
    attention_backend: Literal["sdpa", "eager", "flex_attention"] = "sdpa"
    log_interval: int = Field(default=1, gt=0)
    save_interval: int = Field(default=1, gt=0)
    # Always keep trajectory JSON (including discarded proposals). Large replay
    # tensors can be released only after training and all synchronization ACKs.
    retain_replay_tensors: bool = True

    @model_validator(mode="after")
    def training_budget(self):
        if self.max_steps is not None and self.num_epochs is not None:
            raise ValueError("set exactly one of training.max_steps or num_epochs")
        if self.max_steps is None and self.num_epochs is None:
            self.max_steps = 100
        return self


class RolloutConfig(StrictConfigModel):
    # Independent single-GPU engines; each loads the frozen target and draft.
    placement: Literal["dedicated", "colocated"] = "dedicated"
    cuda_devices: list[int] = Field(default_factory=lambda: [0], min_length=1)
    mem_fraction_static: float = Field(default=0.75, gt=0, lt=1)
    context_length: int = Field(default=4096, gt=0)
    attention_backend: Literal["flashinfer", "triton"] = "flashinfer"
    timeout_s: float = Field(default=1800, gt=0)

    @model_validator(mode="after")
    def unique_devices(self):
        if min(self.cuda_devices) < 0 or len(set(self.cuda_devices)) != len(
            self.cuda_devices
        ):
            raise ValueError(
                "rollout.cuda_devices must be distinct nonnegative ordinals"
            )
        return self


class OnPolicyConfig(StrictConfigModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    @model_validator(mode="after")
    def context_budget(self):
        if self.rollout.context_length < self.sequence_limit + 16:
            raise ValueError(
                "rollout.context_length must cover prompt + generation + 16 slots"
            )
        return self

    @property
    def sequence_limit(self):
        limit = self.data.max_prompt_length + self.sampling.max_new_tokens
        return min(limit, self.data.max_length) if self.data.max_length else limit

    def sampling_for_prompt(self, prompt_length):
        sampling = self.sampling.model_dump()
        if self.data.max_length is not None:
            sampling["max_new_tokens"] = min(
                sampling["max_new_tokens"], self.data.max_length - prompt_length
            )
        if sampling["max_new_tokens"] < 2:
            raise ValueError("prompt leaves no budget for speculative candidates")
        return sampling

    @classmethod
    def from_file(cls, path):
        import yaml

        with open(path, encoding="utf-8") as stream:
            return cls.model_validate(yaml.safe_load(stream))
