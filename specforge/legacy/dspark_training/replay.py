"""Strict rollout -> replay -> optimizer -> acknowledged draft sync loop."""

import itertools
import json
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from specforge.data.dspark_replay import ReplayCollator


def collective_phase(call, device):
    """Every rank observes a local Engine failure before entering more FSDP work."""
    result, error = None, None
    try:
        result = call()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(
            f"synchronous replay phase failed; stopping all ranks: {errors}"
        )
    return result


class SynchronousReplayController:
    """A version may be collected once and consumed by exactly one update."""

    def __init__(self, backend, sync_weights, *, initial_version=0):
        self.backend = backend
        self.sync_weights = sync_weights
        self.version = initial_version
        self.phase = "new"

    def synchronize(self):
        if self.phase not in ("new", "updated"):
            raise RuntimeError(f"cannot synchronize during {self.phase}")
        try:
            acknowledged = self.sync_weights(self.version)
            if acknowledged != self.version:
                raise RuntimeError(
                    "draft synchronization version acknowledgement mismatch"
                )
            self.phase = "ready"
        except BaseException:
            self.phase = "failed"
            raise

    def collect(self, prompts):
        if self.phase != "ready":
            raise RuntimeError("rollout cannot start before successful weight sync")
        self.phase = "rollout"
        try:
            records = self.backend.rollout(prompts, self.version)
            if any(record["draft_version"] != self.version for record in records):
                raise RuntimeError("rollout draft_version mismatch")
            self.phase = "replay"
            return records
        except BaseException:
            self.phase = "failed"
            raise

    def update(self, optimizer):
        if self.phase != "replay":
            raise RuntimeError(
                "each rollout batch must feed exactly one optimizer update"
            )
        try:
            optimizer.step()
            self.version += 1
            self.phase = "updated"
            self.synchronize()
        except BaseException:
            self.phase = "failed"
            raise


def full_draft_state(model):
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        state = model.state_dict()
    return {
        name.removeprefix("draft_model."): value.detach().cpu().contiguous().clone()
        for name, value in state.items()
        if name.startswith("draft_model.")
    }


def prompt_rows(data, *, prefix):
    if isinstance(data, list):
        return [
            (f"{prefix}-{i}-{r['sample_id']}", list(r["input_ids"]))
            for i, r in enumerate(data)
        ]
    result = []
    for i, (ids, attention, loss) in enumerate(
        zip(data["input_ids"], data["attention_mask"], data["loss_mask"])
    ):
        supervised = ((loss > 0.5) & attention.bool()).nonzero().flatten()
        if not supervised.numel() or int(supervised[0]) < 1:
            raise ValueError(
                "chat prompt requires at least one masked prefix token before the response"
            )
        prompt = ids[: int(supervised[0])].tolist()
        result.append((f"{prefix}-{i}-{int(data['sample_id'][i])}", prompt))
    return result


def split_replay_batch(data, size):
    count = data["input_ids"].size(0)
    for start in range(0, count, size):
        stop = min(count, start + size)
        yield {
            key: (
                value[start:stop]
                if isinstance(value, list)
                or (isinstance(value, torch.Tensor) and value.ndim > 0)
                else value
            )
            for key, value in data.items()
        }


def run_replay_training(
    args,
    dataloader,
    backend,
    model,
    draft,
    optimizer,
    device,
    tracker,
    total_steps,
    needs_target_hidden,
    *,
    forward_batch,
    save_checkpoint,
):
    rank = dist.get_rank()
    # Parameter names under FSDP contain implementation wrappers; state_dict
    # removes them. Include every draft parameter, with a separate trainable manifest.
    trainable_names = {
        name.replace("_fsdp_wrapped_module.", "")
        for name, p in draft.named_parameters()
        if p.requires_grad
    }

    def sync_weights(version):
        state = full_draft_state(model)
        return collective_phase(
            lambda: backend.sync(state, draft.config, trainable_names, version), device
        )

    controller = SynchronousReplayController(backend, sync_weights)
    controller.synchronize()
    run_ids = [uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(run_ids, src=0)
    run_id = run_ids[0]
    trace_dir = Path(args.output_dir).resolve() / "rollout_traces" / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    if getattr(backend, "directory", None) is not None:
        (trace_dir / f"sglang-source-rank-{rank}.json").write_text(
            (backend.directory / "source-audit.json").read_text()
        )
    step = 0
    for epoch in range(args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        iterator = iter(dataloader)
        while True:
            # Prefetching/tokenizing prompts is harmless; generation occurs only
            # here, after the previous step's all-rank synchronization barrier.
            batches = list(itertools.islice(iterator, args.accumulation_steps))
            if not batches:
                break
            prompts = []
            for index, batch in enumerate(batches):
                prompts.extend(
                    prompt_rows(
                        batch, prefix=f"{run_id}-v{controller.version}-r{rank}-b{index}"
                    )
                )
            records = collective_phase(lambda: controller.collect(prompts), device)
            args.draft_version = controller.version

            def collate():
                # Save raw traces before budget subsampling; acceptance lengths
                # remain telemetry and never determine the replay window.
                path = trace_dir / f"step-{step:08d}-rank-{rank:05d}.jsonl"
                with path.open("x") as stream:
                    for record in records:
                        stream.write(json.dumps(record) + "\n")
                return ReplayCollator(
                    draft_version=controller.version,
                    target_version=args.rollout_target_version,
                    num_anchors=args.num_anchors,
                    eos_token_ids=model.tv_eos_token_ids,
                )(records)

            data = collective_phase(collate, device)
            count = data["block_keep_mask"].sum().to(device=device, dtype=torch.float32)
            dist.all_reduce(count)
            if count.item() <= 0:
                raise ValueError("rollout batch has no valid replay blocks on any rank")
            data["tv_global_block_count"] = count
            step += 1
            model.set_training_progress(step, total_steps)
            draft.train()
            micro_size = args.micro_batch_size or args.batch_size
            loss_sum = torch.zeros((), device=device)
            proxy_sum = torch.zeros_like(loss_sum)
            overlap_sum = torch.zeros_like(loss_sum)
            correct_sum = torch.zeros_like(loss_sum)
            position_sum = torch.zeros_like(loss_sum)
            for index, micro in enumerate(split_replay_batch(data, micro_size)):
                result = forward_batch(
                    micro,
                    args,
                    device,
                    backend,
                    model,
                    needs_target_hidden,
                    step,
                    micro_index=index,
                )
                loss = result[0]
                # Already normalized by the global effective-batch block count.
                loss.backward()
                loss_sum += loss.detach()
                metrics = result[5]
                positions = result[4].sum()
                proxy_sum += (
                    metrics["tv_acceptance_length_proxy"] * metrics["tv_valid_blocks"]
                )
                overlap_sum += metrics["tv_mean_overlap"] * positions
                correct_sum += result[1] * positions
                position_sum += positions
            controller.update(optimizer)
            args.draft_version = controller.version
            dist.all_reduce(loss_sum)
            loss_sum /= dist.get_world_size()
            if step % args.log_interval == 0:
                tracker.log(
                    {
                        "train/loss": loss_sum.item(),
                        "train/tv_valid_blocks": count.item(),
                        "train/tv_acceptance_loss": loss_sum.item(),
                        "train/tv_acceptance_length_proxy": (proxy_sum / count).item(),
                        "train/tv_mean_overlap": (
                            overlap_sum / position_sum.clamp_min(1)
                        ).item(),
                        "train/tv_mean_valid_length": (position_sum / count).item(),
                        "train/accuracy": (
                            correct_sum / position_sum.clamp_min(1)
                        ).item(),
                        "train/draft_version": controller.version,
                        "train/lr": optimizer.get_learning_rate(),
                    },
                    step=step,
                )
                if rank == 0:
                    print(
                        f"Replay step {step}: loss={loss_sum.item():.6f}, blocks={int(count)}, synchronized draft V{controller.version}",
                        flush=True,
                    )
            if step % args.save_interval == 0:
                save_checkpoint(args, epoch, step, model, draft, optimizer)
            if (args.max_steps is not None and step >= args.max_steps) or (
                args.stop_after_steps is not None and step >= args.stop_after_steps
            ):
                save_checkpoint(args, epoch, step, model, draft, optimizer)
                return
    save_checkpoint(args, args.num_epochs, step, model, draft, optimizer)
