"""Synchronous rollout -> sample-balanced FSDP replay -> full-weight barrier."""

import json
import logging
import os
import uuid
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .data import atomic_json, load_prompts, load_trace, trace_stem
from .schedule import planned_steps, prompt_batches

logger = logging.getLogger(__name__)


def _leader_call(action, rank, control_group=None):
    """Propagate control-plane failures before peers enter the next collective."""
    result, error = None, None
    if rank == 0:
        try:
            result = action()
        except Exception as exc:
            error = exc
    # A GPU/NCCL wait can compete with the SGLang process on the same device.
    # Colocated trainers wait on the CPU group throughout rollout and sync.
    device = (
        "cpu"
        if control_group is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    status = torch.tensor(int(error is None), device=device)
    dist.broadcast(status, src=0, group=control_group)
    if not status.item():
        if error is not None:
            raise error
        raise RuntimeError("on-policy coordinator failed; the batch cannot continue")
    return result


def replay_schedule(traces, rank, world_size):
    """Equal collective counts despite unequal numbers of blocks per sample.

    FSDP averages over ranks. Each real block therefore gets weight W/(N*M).
    Empty slots execute a real block with zero loss to join all collectives.
    """
    valid = [
        trace for trace in traces if any(any(b["valid_mask"]) for b in trace["blocks"])
    ]
    if not valid:
        raise ValueError("effective batch has no valid samples")
    per_rank = [[] for _ in range(world_size)]
    for sample_index, trace in enumerate(valid):
        blocks = [
            index
            for index, block in enumerate(trace["blocks"])
            if any(block["valid_mask"])
        ]
        for index in blocks:
            per_rank[sample_index % world_size].append(
                (trace, index, world_size / len(valid) / len(blocks))
            )
    slots = max(map(len, per_rank))
    exemplar = next(items[0] for items in per_rank if items)
    return per_rank[rank] + [(exemplar[0], exemplar[1], 0.0)] * (
        slots - len(per_rank[rank])
    )


def train_effective_batch(
    backend, traces, tensor_loader, rank, world_size, replay_max_tv
):
    schedule = replay_schedule(traces, rank, world_size)
    metrics = torch.zeros(2, device=next(backend.module.parameters()).device)
    for slot, (trace, index, weight) in enumerate(schedule):
        block = trace["blocks"][index]
        data = tensor_loader(trace)
        boundary = slot == len(schedule) - 1
        # no_sync must enclose FORWARD as well as backward (FSDP/DDP contract).
        context = nullcontext() if boundary else backend.module.no_sync()
        with context:
            loss, parity = backend.module(
                block,
                data["context_hidden"][: block["context_length"]],
                data[f"p_{index}"],
                data[f"q_{index}"],
            )
            error = parity.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(error, op=dist.ReduceOp.MAX)
            if not torch.isfinite(error) or error.item() > replay_max_tv:
                raise RuntimeError(
                    f"rollout/replay q TV={error.item():.6f} exceeds {replay_max_tv}; optimizer was not stepped"
                )
            backend.backward(loss * weight, is_boundary=True)
        metrics[0] += loss.detach() * weight / world_size
        metrics[1] = torch.maximum(metrics[1], parity)
    if dist.is_initialized():
        dist.all_reduce(metrics[0], op=dist.ReduceOp.SUM)
        dist.all_reduce(metrics[1], op=dist.ReduceOp.MAX)
    backend.step()  # exactly once, after the whole effective batch
    return {"loss": metrics[0].item(), "replay_max_tv": metrics[1].item()}


def _publish_weights(backend, draft_config, root, version, rank, control_group=None):
    from safetensors.torch import save_file

    # All FSDP ranks must participate, including ranks with no local samples.
    state = backend._module_state_dict()

    def publish():
        path = root / "weights" / f"{version:08d}"
        path.mkdir(parents=True, exist_ok=False)
        draft_state = {
            name.removeprefix("draft_model."): value.detach().cpu().contiguous()
            for name, value in state.items()
            if name.startswith("draft_model.")
        }
        if not draft_state:
            raise RuntimeError("FSDP export contained no draft parameters")
        save_file(draft_state, str(path / "model.safetensors"))
        draft_config.save_pretrained(path)

    _leader_call(publish, rank, control_group)


def _prepare_rollout_phase(control_group):
    if control_group is not None:
        # All ranks finish training/export and return unused allocator blocks
        # before the colocated engines allocate or update their serving weights.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier(group=control_group)


def _validate_draft(config):
    if config.architectures != ["DSparkDraftModel"]:
        raise ValueError("on-policy supports the registered dense DSparkDraftModel")
    if getattr(config, "is_causal", None) is not None:
        raise ValueError(
            "explicit is_causal overrides are not supported by DSpark replay"
        )
    if config.dflash_config.get("attention_mode", "gqa") not in {"gqa", "mha"}:
        raise ValueError("the initial on-policy gate supports GQA/MHA dense drafts")
    if config.dflash_config.get("markov_rank", 0) <= 0:
        raise ValueError("SGLang DSpark requires a Markov head")
    if config.dflash_config.get("mask_token_id") is None:
        raise ValueError("checkpoint must define dflash_config.mask_token_id")
    if getattr(config, "attention_dropout", 0) != 0:
        raise ValueError("on-policy rollout/replay requires attention_dropout=0")
    if getattr(config, "draft_vocab_size", config.vocab_size) != config.vocab_size:
        raise ValueError("reduced-vocabulary DSpark replay is not supported")


def retire_step_artifacts(root, version, training, request_ids):
    """Release only this run's generated artifacts after the all-worker ACK."""
    previous = version - 1
    if previous > 0 and previous % training.save_interval:
        path = root / "weights" / f"{previous:08d}"
        # COSEC: never follow a substituted directory outside this run's weights.
        if path.resolve().parent != root.resolve() / "weights":
            raise ValueError("weight directory escaped the run")
        for name in ("model.safetensors", "config.json"):
            (path / name).unlink()
        path.rmdir()
    if not training.retain_replay_tensors:
        for request_id in request_ids:
            path = trace_stem(root, request_id).with_suffix(".safetensors")
            # COSEC: delete only generated tensor files, never trajectory JSON.
            if path.resolve().parent != root.resolve() / "trajectories":
                raise ValueError("trajectory directory escaped the run")
            path.unlink()


def run(cfg):
    control_group = None
    if cfg.rollout.placement == "colocated":
        control_group = dist.new_group(
            backend="gloo", timeout=timedelta(minutes=cfg.training.dist_timeout)
        )
    try:
        return _run(cfg, control_group)
    finally:
        if control_group is not None:
            dist.destroy_process_group(control_group)


def _run(cfg, control_group):
    from transformers import AutoTokenizer

    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.model_loading import (
        load_draft_config_source,
        warm_start_draft_model,
    )

    from .replay import DSparkReplayModel
    from .rollout import RolloutPool

    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    root = Path(cfg.training.output_dir).expanduser().resolve()

    def initialize_output():
        # Existing runs are never silently overwritten; warm-start from an
        # exported version into a fresh output directory to continue training.
        root.mkdir(parents=True, exist_ok=True)
        if (root / "run.json").exists() or (root / "weights").exists():
            raise ValueError("on-policy output directory already contains a run")
        atomic_json(root / "run.json", cfg.model_dump())

    _leader_call(initialize_output, rank, control_group)
    torch.manual_seed(cfg.training.seed)
    draft_config = load_draft_config_source(
        cfg.model.draft_model_config or cfg.model.draft_checkpoint_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    _validate_draft(draft_config)
    if cfg.rollout.context_length < cfg.sequence_limit + draft_config.block_size + 7:
        raise ValueError(
            "rollout context must also cover the complete speculative block"
        )
    draft_config._attn_implementation = cfg.training.attention_backend
    draft = AutoDraftModel.from_config(draft_config, torch_dtype=torch.bfloat16)
    warm_start_draft_model(
        draft,
        cfg.model.draft_checkpoint_path,
        draft_config=draft_config,
        strategy="dspark",
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    parts = TargetEmbeddingsAndHead.from_pretrained(
        cfg.model.target_model_path,
        embed_key=cfg.model.embedding_key,
        lm_head_key=cfg.model.lm_head_key,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
        device=str(device),
        dtype=torch.bfloat16,
    )
    model = DSparkReplayModel(draft, parts).to(device)
    model.eval()  # gradients remain enabled; match the inference backbone

    def prepare_prompts():
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.target_model_path,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        prompts = load_prompts(
            cfg.data.train_data_path,
            tokenizer,
            cfg.data.prompt_limit,
            cfg.data.chat_template_kwargs,
            chat_template=cfg.data.chat_template,
        )
        # skip_tokenizer_init requires explicit EOS IDs on engine requests.
        if not cfg.sampling.ignore_eos:
            eos = getattr(parts.config, "eos_token_id", None)
            eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
            cfg.sampling.stop_token_ids = sorted(
                set(
                    cfg.sampling.stop_token_ids
                    + eos
                    + (
                        []
                        if tokenizer.eos_token_id is None
                        else [tokenizer.eos_token_id]
                    )
                )
            )
        atomic_json(
            root / "schedule.json",
            {
                "eligible_prompts": len(prompts),
                "total_steps": planned_steps(len(prompts), cfg.training),
                "global_batch_size": cfg.training.batch_size,
                "num_epochs": cfg.training.num_epochs,
            },
        )
        # Include resolved stop IDs so run.json describes actual engine sampling.
        atomic_json(root / "run.json", cfg.model_dump())
        return prompts

    prompts = _leader_call(prepare_prompts, rank, control_group)
    total_steps = json.loads((root / "schedule.json").read_text())["total_steps"]
    backend = FSDPTrainingBackend(
        ParallelConfig.from_distributed(sharding_strategy=cfg.training.fsdp_sharding),
        optimizer_factory=lambda module: BF16Optimizer(
            module,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            max_grad_norm=cfg.training.max_grad_norm,
            total_steps=total_steps,
            warmup_ratio=cfg.training.warmup_ratio,
        ),
    )
    # FSDP must preserve the captured FP32 p/q; BF16 rounding can break their
    # normalization and replay parity. Replay casts only the model features.
    backend.prepare_model(model, optimizer_target=draft, cast_root_forward_inputs=False)
    _publish_weights(backend, draft_config, root, 0, rank, control_group)
    _prepare_rollout_phase(control_group)
    pool = None
    try:
        pool = _leader_call(
            lambda: RolloutPool(cfg, root, draft.block_size), rank, control_group
        )
        batches = prompt_batches(prompts, cfg.training) if rank == 0 else None
        for step in range(total_steps):
            # Nothing for version step+1 may be generated during this batch.
            def collect_batch():
                epoch, batch_prompts = next(batches)
                traces, empty, all_ids = [], 0, []
                for offset in range(0, len(batch_prompts), len(pool.workers)):
                    wave = batch_prompts[offset : offset + len(pool.workers)]
                    requests = [
                        {
                            "request_id": uuid.uuid4().hex,
                            "input_ids": prompt["input_ids"],
                        }
                        for prompt in wave
                    ]
                    all_ids.extend(request["request_id"] for request in requests)
                    for result in pool.generate(requests):
                        trace, _ = load_trace(root, result["request_id"], step)
                        if any(any(block["valid_mask"]) for block in trace["blocks"]):
                            traces.append(trace)
                        else:
                            empty += 1
                            if empty >= cfg.training.max_empty_samples:
                                raise RuntimeError(
                                    "too many rollouts without a valid candidate block"
                                )
                if not traces:
                    raise RuntimeError(
                        "batch has no valid candidate blocks; no optimizer step"
                    )
                atomic_json(
                    root / "batches" / f"{step:08d}.json",
                    [trace["request_id"] for trace in traces],
                )
                atomic_json(
                    root / "batches" / f"{step:08d}-prompts.json",
                    {
                        "epoch": epoch,
                        "sample_ids": [prompt["sample_id"] for prompt in batch_prompts],
                        "request_ids": all_ids,
                    },
                )
                return epoch, all_ids

            collected = _leader_call(collect_batch, rank, control_group)
            request_ids = json.loads(
                (root / "batches" / f"{step:08d}.json").read_text()
            )
            traces = [
                load_trace(root, request_id, step, metadata_only=True)[0]
                for request_id in request_ids
            ]
            # Keep at most one sequence's features/logits resident on the host.
            cached_id, cached_tensors = None, None

            def tensor_loader(trace):
                nonlocal cached_id, cached_tensors
                if cached_id != trace["request_id"]:
                    _, cached_tensors = load_trace(root, trace["request_id"], step)
                    cached_id = trace["request_id"]
                return cached_tensors

            metrics = train_effective_batch(
                backend, traces, tensor_loader, rank, world, cfg.training.replay_max_tv
            )
            _publish_weights(backend, draft_config, root, step + 1, rank, control_group)
            _prepare_rollout_phase(control_group)

            def synchronize_and_log():
                pool.synchronize(step + 1)
                blocks = [
                    block
                    for trace in traces
                    for block in trace["blocks"]
                    if any(block["valid_mask"])
                ]
                metrics.update(
                    step=step + 1,
                    weight_version=pool.version,
                    epoch=collected[0],
                    full_weight_sync=True,
                    replay_tensors_retained=cfg.training.retain_replay_tensors,
                    samples=len(traces),
                    blocks=len(blocks),
                    actual_accepted_length=sum(
                        block["actual_accepted_count"] for block in blocks
                    )
                    / len(blocks),
                )
                atomic_json(root / "metrics" / f"{step + 1:08d}.json", metrics)
                retire_step_artifacts(root, step + 1, cfg.training, collected[1])
                if (
                    step + 1
                ) % cfg.training.log_interval == 0 or step + 1 == total_steps:
                    logger.info("on-policy %s", json.dumps(metrics))

            # Completion ACK from ALL rollout workers precedes the next batch.
            _leader_call(synchronize_and_log, rank, control_group)
        return total_steps
    finally:
        if pool is not None:
            pool.close()
