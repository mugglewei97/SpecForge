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
    chat_template_kwargs: dict = Field(default_factory=dict)


class TrainingConfig(StrictConfigModel):
    output_dir: str
    max_steps: int = Field(default=100, gt=0)
    # A global number of valid samples, independent of rank/block/token count.
    batch_size: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=1e-6, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    max_grad_norm: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    warmup_ratio: float = Field(default=0.0, ge=0, le=1)
    seed: int = 42
    dist_timeout: int = Field(default=60, gt=0)
    fsdp_sharding: Literal["FULL_SHARD", "SHARD_GRAD_OP"] = "FULL_SHARD"
    # Different CUDA kernels have rounding differences; fail closed on drift.
    replay_max_tv: float = Field(default=0.02, gt=0, lt=1)
    max_empty_samples: int = Field(default=100, gt=0)


class RolloutConfig(StrictConfigModel):
    # Independent single-GPU engines; each loads the frozen target and draft.
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
        if (
            self.rollout.context_length
            < self.data.max_prompt_length + self.sampling.max_new_tokens + 16
        ):
            raise ValueError(
                "rollout.context_length must cover prompt + generation + 16 slots"
            )
        return self

    @classmethod
    def from_file(cls, path):
        import yaml

        with open(path, encoding="utf-8") as stream:
            return cls.model_validate(yaml.safe_load(stream))
