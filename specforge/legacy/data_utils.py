# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from datasets import Dataset
from specforge.distributed import get_draft_sp_group, get_sp_ulysses_group


class DistributedWeightedSampler(Sampler[int]):
    """Deterministic distributed sampling with mutable per-example weights.

    Every data-parallel rank draws the same global index stream and consumes a
    disjoint strided shard.  ``update_weights`` may be called between epochs to
    turn acceptance diagnostics from epoch ``t`` into the curriculum for epoch
    ``t + 1``.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} must be in [0, {num_replicas})")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.weights = torch.ones(len(dataset), dtype=torch.double)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def update_weights(self, weights: torch.Tensor) -> None:
        weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        if weights.ndim != 1 or weights.numel() != len(self.dataset):
            raise ValueError(
                f"Expected {len(self.dataset)} replay weights, "
                f"got {tuple(weights.shape)}"
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Replay weights must be finite and non-negative")
        if weights.sum() <= 0:
            raise ValueError("At least one replay weight must be positive")
        self.weights = weights / weights.sum()

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        indices = global_indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_acceptance_replay_weights(
    signals: torch.Tensor,
    observed: torch.Tensor,
    masses: Sequence[float] = (0.60, 0.25, 0.10, 0.05),
    max_weight_ratio: float = 10.0,
) -> torch.Tensor:
    """Mix uniform, boundary, deep-failure and full-accept replay proposals.

    ``signals`` is ``[num_examples, 3]`` with non-negative scores for the
    boundary-hard, deep-hard and full-accept strata.  A stratum with no mass in
    the completed epoch falls back to uniform sampling, so the distribution is
    always well-defined.
    """

    if signals.ndim != 2 or signals.shape[1] != 3:
        raise ValueError("signals must have shape [num_examples, 3]")
    if observed.shape != signals.shape[:1]:
        raise ValueError("observed must have shape [num_examples]")
    if len(masses) != 4 or any(m < 0 for m in masses):
        raise ValueError("masses must contain four non-negative values")
    if abs(sum(masses) - 1.0) > 1e-6:
        raise ValueError("replay masses must sum to 1")
    if max_weight_ratio < 1:
        raise ValueError("max_weight_ratio must be at least 1")

    num_examples = signals.shape[0]
    if num_examples == 0:
        raise ValueError("Cannot build replay weights for an empty dataset")
    dtype = torch.float64
    uniform = torch.full((num_examples,), 1.0 / num_examples, dtype=dtype)
    probabilities = masses[0] * uniform
    valid = observed.to(device="cpu", dtype=torch.bool)
    scores = signals.detach().to(device="cpu", dtype=dtype).clamp_min(0)
    scores = scores * valid.unsqueeze(-1)
    for column, mass in enumerate(masses[1:]):
        proposal = scores[:, column]
        probabilities += mass * (
            proposal / proposal.sum() if proposal.sum() > 0 else uniform
        )

    cap = max_weight_ratio / num_examples
    if probabilities.max() <= cap:
        return probabilities / probabilities.sum()

    # Project by a single positive scale followed by clipping.  Bisection
    # preserves normalization while honoring the advertised hard cap.
    low, high = 0.0, 1.0
    while torch.clamp(probabilities * high, max=cap).sum() < 1.0:
        high *= 2.0
    for _ in range(64):
        midpoint = (low + high) / 2.0
        if torch.clamp(probabilities * midpoint, max=cap).sum() < 1.0:
            low = midpoint
        else:
            high = midpoint
    projected = torch.clamp(probabilities * high, max=cap)
    return projected / projected.sum()


def build_source_mixture_weights(
    source_sizes: Sequence[int], source_masses: Sequence[float]
) -> torch.Tensor:
    """Give each dataset source its requested total sampling probability."""
    if not source_sizes or len(source_sizes) != len(source_masses):
        raise ValueError("source sizes and masses must have the same non-zero length")
    if any(size <= 0 for size in source_sizes):
        raise ValueError("every source must contain at least one sample")
    if any(mass < 0 for mass in source_masses) or sum(source_masses) <= 0:
        raise ValueError("source masses must be non-negative with positive total mass")
    total_mass = float(sum(source_masses))
    return torch.cat(
        [
            torch.full(
                (size,),
                float(mass) / total_mass / size,
                dtype=torch.double,
            )
            for size, mass in zip(source_sizes, source_masses)
        ]
    )


class DataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def __init__(self):
        self.sp_degree = torch.distributed.get_world_size(get_draft_sp_group())
        self.ulysses_degree = torch.distributed.get_world_size(get_sp_ulysses_group())

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, S, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: A list of features, where each feature is a dictionary containing:
                - input_ids: torch.Tensor of shape (n,)
                - attention_mask: torch.Tensor of shape (n,)
                - loss_mask: torch.Tensor of shape (n,)

        Returns:
            A dictionary containing:
                - input_ids: torch.Tensor of shape (B, N)
                - attention_mask: torch.Tensor of shape (B, N)
                - loss_mask: torch.Tensor of shape (B, N)
        """
        max_length = max(item["input_ids"].shape[1] for item in features)

        # pad for sequence parrel
        max_length = (
            (max_length + self.sp_degree - 1) // self.sp_degree
        ) * self.sp_degree
        # position max len, ulysses do not need chuck position ids
        position_max_len = max_length * self.ulysses_degree

        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        if "position_ids" in features[0]:
            batch_position_ids = torch.cat(
                [
                    self.paddingtensor2D(item["position_ids"], position_max_len)
                    for item in features
                ]
            )
        else:
            batch_position_ids = None
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_state": None,
            "target": None,
        }
        if batch_position_ids is not None:
            batch["position_ids"] = batch_position_ids
        if "__sample_id" in features[0]:
            batch["sample_id"] = torch.tensor(
                [int(item["__sample_id"]) for item in features],
                dtype=torch.long,
            )
        if "__source_id" in features[0]:
            batch["source_id"] = torch.tensor(
                [int(item["__source_id"]) for item in features],
                dtype=torch.long,
            )
        if all("hidden_state" in item for item in features):
            assert all(
                "target" in item for item in features
            ), "target is required when hidden_state is provided"
            if self.sp_degree > 1:  # USP mode
                batch["hidden_state"] = torch.cat(
                    [item["hidden_state"] for item in features]
                )
            else:
                batch["hidden_state"] = torch.cat(
                    [
                        self.paddingtensor(item["hidden_state"], max_length)
                        for item in features
                    ]
                )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


class VlmDataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: A list of features, where each feature is a dictionary containing:
                - input_ids: torch.Tensor of shape (n,)
                - attention_mask: torch.Tensor of shape (n,)
                - loss_mask: torch.Tensor of shape (n,)
                - pixel_values: torch.Tensor of shape (grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
                - image_grid_thw: torch.Tensor of shape (3,)

        Returns:
            A dictionary containing:
                - input_ids: torch.Tensor of shape (B, N)
                - attention_mask: torch.Tensor of shape (B, N)
                - loss_mask: torch.Tensor of shape (B, N)
        """
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        batch_pixel_values = torch.cat(
            [item["pixel_values"] for item in features], dim=0
        )
        batch_image_grid_thw = torch.cat(
            [item["image_grid_thw"] for item in features], dim=0
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
            "hidden_state": None,
            "target": None,
        }
        if all("hidden_state" in item for item in features):
            assert all(
                "target" in item for item in features
            ), "target is required when hidden_state is provided"
            batch["hidden_state"] = torch.cat(
                [
                    self.paddingtensor(item["hidden_state"], max_length)
                    for item in features
                ]
            )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


class LengthBucketDistributedSampler(DistributedSampler):
    """Shuffle nearby-length global batches; TP peers share the same DP rank.

    Preserve DistributedSampler's count/padding and set_epoch contract. Sorting
    only within shuffled windows avoids an epoch-wide short-to-long curriculum.
    """
    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__(dataset, **kwargs)
        import pyarrow as pa
        import pyarrow.compute as pc

        column = dataset.data.column("input_ids")
        nested = column.type.value_type
        if pa.types.is_list(nested) or pa.types.is_large_list(nested) or pa.types.is_fixed_size_list(nested):
            column = pc.list_element(column, 0)  # stored [1, T] sequences
        # Arrow computes lengths without decoding/copying the token payload.
        lengths = pc.list_value_length(column)
        if dataset._indices is not None:
            lengths = pc.take(lengths, dataset._indices.column(0))
        self.lengths = lengths.to_pylist()
        self.global_batch_size = batch_size * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        ordered = []
        window = self.global_batch_size * 50
        for start in range(0, len(indices), window):
            bucket = sorted(indices[start:start + window], key=self.lengths.__getitem__, reverse=True)
            full = len(bucket) // self.global_batch_size
            for batch in torch.randperm(full, generator=generator).tolist():
                offset = batch * self.global_batch_size
                ordered.extend(bucket[offset:offset + self.global_batch_size])
            ordered.extend(bucket[full * self.global_batch_size:])
        return iter(ordered[self.rank:self.total_size:self.num_replicas])


def prepare_dp_dataloaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    pin_memory: Optional[bool] = False,
    shuffle: Optional[bool] = False,
    is_vlm: Optional[bool] = False,
    prefetch_factor: Optional[int] = 2,
    weighted_sampling: bool = False,
    length_bucket: bool = False,
    sampler_seed: int = 0,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Prepare dataloader for distributed data parallel training.

    Args:
        dataset: The dataset to load data from.
        batch_size: The batch size for each GPU.
        num_workers: The number of workers for data loading.
        process_group: The process group for distributed training.
        pin_memory: Whether to pin memory for data loading.
        shuffle: Whether to shuffle the dataset.
        is_vlm: Whether the dataset is a vision-language model dataset.
        **dataloader_kwargs: Additional keyword arguments for the DataLoader.

    Returns:
        A DataLoader for the dataset.
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    if length_bucket:
        if weighted_sampling or not shuffle or is_vlm:
            raise ValueError("Length bucketing requires shuffled, unweighted text data")
        sampler = LengthBucketDistributedSampler(
            dataset, batch_size=batch_size, num_replicas=world_size, rank=rank,
            shuffle=True, seed=sampler_seed,
        )
    elif weighted_sampling:
        sampler = DistributedWeightedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=sampler_seed,
        )
    else:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
    if is_vlm:
        datacollator_cls = VlmDataCollatorWithPadding
    else:
        datacollator_cls = DataCollatorWithPadding

    if num_workers == 0:
        prefetch_factor = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=datacollator_cls(),
        drop_last=True,
        **dataloader_kwargs,
    )
    return dataloader


def parse_harmony_message_content(content):
    """
    解析 content 字符串中的 Harmony 格式。
    如果匹配到 Harmony 格式，返回包含 channel 和 content 的列表；
    否则，返回原内容并标记为默认 channel。
    """
    # 匹配 <|channel|>xxx<|message|>yyy<|end|>
    pattern = r"<\|channel\|>(.*?)<\|message\|>(.*?)<\|end|>"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        # 如果没有匹配到 Harmony 标签，视作普通文本
        return [{"channel": "text", "content": content}]

    results = []
    for channel, msg_body in matches:
        results.append({"channel": channel.strip(), "content": msg_body.strip()})
    return results


def process_harmony_conversations(conversation):
    """
    处理传入的 list[list[dict]] 结构
    """
    new_conversation = []
    for msg in conversation:
        role = msg.get("role")
        original_content = msg.get("content", "")

        # 解析 content 中的 Harmony 结构
        segments = parse_harmony_message_content(original_content)

        # 为每个解析出的通道生成一个新的消息字典
        for seg in segments:
            new_msg = {
                "role": role,
                "channel": seg["channel"],  # 新增字段标识通道
                "content": seg["content"],
            }
            new_conversation.append(new_msg)

    return new_conversation
