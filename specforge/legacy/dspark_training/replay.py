"""Strict rollout -> replay -> optimizer -> acknowledged draft sync loop."""

import concurrent.futures
import itertools
import json
import time
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


def full_draft_state(model, *, offload_to_cpu=True):
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=False),
    ):
        state = model.state_dict()
    return {
        # FULL_STATE_DICT already owns full tensors independent of FSDP shards.
        name.removeprefix("draft_model."): value.detach().contiguous()
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


def _write_trace(path, records):
    """Write rollout trace records to a JSONL file (called from background thread)."""
    with path.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def rollout_counts(records):
    """Actual output counts, with accepted proposals clipped to the stop boundary."""
    rounds = [meta for record in records for meta in record["round_metadata"]]
    return [
        sum(len(r["input_ids"]) - r["prompt_length"] for r in records),
        len(rounds),
        sum(min(r["accepted_draft_tokens"], r["committed_tokens"]) for r in rounds),
        sum(r.get("verified_draft_tokens", r["draft_length"]) for r in rounds),
        sum(r["committed_tokens"] for r in rounds),
    ]


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
    sync_times = {}

    def synchronize_device():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    def sync_weights(version):
        started = time.perf_counter()
        state = full_draft_state(
            model,
            offload_to_cpu=getattr(backend, "tensor_transport", "files") != "cuda-ipc",
        )
        synchronize_device()
        sync_times["export"] = time.perf_counter() - started
        started = time.perf_counter()
        acknowledged = collective_phase(
            lambda: backend.sync(state, draft.config, trainable_names, version), device
        )
        sync_times["sync"] = time.perf_counter() - started
        return acknowledged

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

    # Background trace writer: one dedicated thread per rank so trace I/O
    # overlaps with target capture + forward/backward.  Each step confirms
    # completion (via future.result()) before optimizer.step(), so errors
    # propagate back and no step's traces are lost.
    trace_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="trace-writer"
    )
    trace_future = None  # the in-flight write, if any

    def _await_trace():
        """Wait for the previous step's background trace write and propagate errors."""
        nonlocal trace_future
        if trace_future is not None:
            try:
                trace_future.result()
            except BaseException:
                # Re-raise in the main thread so collective_phase can sync
                # the failure across all ranks.
                raise
            finally:
                trace_future = None

    try:
        step = 0
        for epoch in range(args.num_epochs):
            dataloader.sampler.set_epoch(epoch)
            iterator = iter(dataloader)
            while True:
                step_t0 = time.perf_counter()
                if torch.device(device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                # Prefetching/tokenizing prompts is harmless; generation occurs only
                # here, after the previous step's all-rank synchronization barrier.
                batches = list(itertools.islice(iterator, args.accumulation_steps))
                if not batches:
                    break
                prompts = []
                for index, batch in enumerate(batches):
                    prompts.extend(
                        prompt_rows(
                            batch,
                            prefix=f"{run_id}-v{controller.version}-r{rank}-b{index}",
                        )
                    )
                started = time.perf_counter()
                records = collective_phase(lambda: controller.collect(prompts), device)
                rollout_seconds = time.perf_counter() - started
                args.draft_version = controller.version

                # --- Submit trace write to background thread ---
                # Serialize records immediately so the background thread owns a
                # snapshot and never races with later mutations of `records`.
                path = trace_dir / f"step-{step:08d}-rank-{rank:05d}.jsonl"
                trace_future = trace_executor.submit(_write_trace, path, records)

                # --- Collate without blocking on trace I/O ---
                def collate():
                    return ReplayCollator(
                        draft_version=controller.version,
                        target_version=args.rollout_target_version,
                        num_anchors=args.num_anchors,
                        eos_token_ids=model.tv_eos_token_ids,
                        prefix_mode=getattr(args, "tv_prefix_mode", "teacher-forced"),
                        block_size=getattr(args, "block_size", None),
                    )(records)

                data = collective_phase(collate, device)
                count = (
                    data["block_keep_mask"].sum().to(device=device, dtype=torch.float32)
                )
                dist.all_reduce(count)
                if count.item() <= 0:
                    raise ValueError(
                        "rollout batch has no valid replay blocks on any rank"
                    )
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
                tv_position_sum = None
                tv_position_count = None
                capture_before = getattr(backend, "capture_seconds", 0.0)
                started = time.perf_counter()
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
                        metrics["tv_acceptance_length_proxy"]
                        * metrics["tv_valid_blocks"]
                    )
                    overlap_sum += metrics["tv_mean_overlap"] * positions
                    correct_sum += result[1] * positions
                    position_sum += positions
                    if tv_position_sum is None:
                        tv_position_sum = torch.zeros_like(result[2])
                        tv_position_count = torch.zeros_like(result[4])
                    tv_position_sum += result[2].detach() * result[4]
                    tv_position_count += result[4]

                synchronize_device()
                capture_seconds = (
                    getattr(backend, "capture_seconds", 0.0) - capture_before
                )
                train_seconds = time.perf_counter() - started - capture_seconds

                # --- Confirm trace write before optimizer.step() ---
                collective_phase(_await_trace, device)
                started = time.perf_counter()
                controller.update(optimizer)
                optimizer_seconds = (
                    time.perf_counter()
                    - started
                    - sync_times["export"]
                    - sync_times["sync"]
                )
                args.draft_version = controller.version
                dist.all_reduce(loss_sum)
                loss_sum /= dist.get_world_size()
                if step % args.log_interval == 0:
                    # Every rank participates. Stage maxima can come from
                    # different ranks and need not add up to total step time.
                    times = torch.tensor(
                        [
                            time.perf_counter() - step_t0,
                            rollout_seconds,
                            capture_seconds,
                            train_seconds,
                            optimizer_seconds,
                            sync_times["export"],
                            sync_times["sync"],
                        ],
                        dtype=torch.float64,
                        device=device,
                    )
                    counts = torch.tensor(
                        rollout_counts(records), dtype=torch.float64, device=device
                    )
                    dist.all_reduce(times, op=dist.ReduceOp.MAX)
                    dist.all_reduce(counts)
                    (
                        elapsed,
                        rollout_s,
                        capture_s,
                        train_s,
                        optimizer_s,
                        export_s,
                        sync_s,
                    ) = times.tolist()
                    output_tokens, rounds, accepted, verified, committed = (
                        counts.tolist()
                    )
                    perf_metrics = {
                        "perf/step_seconds": elapsed,
                        "perf/rollout_seconds": rollout_s,
                        "perf/target_capture_seconds": capture_s,
                        "perf/forward_backward_seconds": train_s,
                        "perf/optimizer_seconds": optimizer_s,
                        "perf/weight_export_seconds": export_s,
                        "perf/weight_sync_seconds": sync_s,
                        "perf/rollout_tokens_per_second": output_tokens
                        / max(rollout_s, 1e-9),
                        "perf/output_tokens_per_step_second": output_tokens
                        / max(elapsed, 1e-9),
                        "rollout/accepted_draft_tokens_per_round": accepted
                        / max(rounds, 1),
                        "rollout/draft_acceptance_rate": accepted / max(verified, 1),
                        "rollout/committed_tokens_per_round": committed
                        / max(rounds, 1),
                    }
                    peak = torch.tensor(
                        torch.cuda.max_memory_allocated(device)
                        if torch.device(device).type == "cuda"
                        else 0,
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                    perf_metrics["perf/trainer_peak_allocated_bytes"] = peak.item()
                    position_metrics = {
                        f"train/tv_position_{index + 1}": value
                        for index, value in enumerate(
                            (tv_position_sum / tv_position_count.clamp_min(1)).tolist()
                        )
                    }
                    tracker.log(
                        {
                            **perf_metrics,
                            **position_metrics,
                            "train/loss": loss_sum.item(),
                            "train/tv_valid_blocks": count.item(),
                            "train/tv_acceptance_loss": loss_sum.item(),
                            "train/tv_acceptance_length_proxy": (
                                proxy_sum / count
                            ).item(),
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
                        acc_len_proxy = (proxy_sum / count).item()
                        mean_overlap = (overlap_sum / position_sum.clamp_min(1)).item()
                        mean_valid_len = (position_sum / count).item()
                        accuracy = (correct_sum / position_sum.clamp_min(1)).item()
                        print(
                            f"Replay step {step}: loss={loss_sum.item():.6f}, blocks={int(count)}, "
                            f"acc_len={acc_len_proxy:.3f}, overlap={mean_overlap:.3f}, "
                            f"valid_len={mean_valid_len:.2f}, acc={accuracy:.4f}, "
                            f"real_accepted={accepted / max(rounds, 1):.3f}, "
                            f"draft V{controller.version}, time={elapsed:.1f}s "
                            f"(rollout={rollout_s:.1f}, capture={capture_s:.1f}, "
                            f"train={train_s:.1f}, export={export_s:.1f}, sync={sync_s:.1f})",
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
    finally:
        # Drain the last in-flight trace write (if any) and shut down the
        # executor cleanly.  Waits up to 30 s; if the write hangs the
        # process is likely exiting anyway.
        if trace_future is not None:
            try:
                trace_future.result(timeout=30)
            except BaseException:
                pass
        trace_executor.shutdown(wait=False)
