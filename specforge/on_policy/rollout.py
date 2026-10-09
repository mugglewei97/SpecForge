"""Isolated SGLang engines and the all-worker weight-version barrier."""

import json
import multiprocessing as mp
import os
import signal
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _idle_rpc(engine, timeout, **kwargs):
    """A finished HTTP/engine response can precede overlap-queue cleanup."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return engine.collective_rpc("specforge_on_policy", **kwargs)
        except AssertionError as exc:
            # Retry only the pre-mutation idle check. A load/cache error must
            # never be retried as if a partial update were a fresh version.
            if (
                "on-policy RPC requires an idle engine" not in str(exc)
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(0.05)


def _engine_process(connection, configuration, root, worker_id, device, block_size):
    engine = None

    def terminate(_signum, _frame):
        raise SystemExit("rollout worker terminated")

    signal.signal(signal.SIGTERM, terminate)
    try:
        # Engine children must not inherit the trainer's torchrun rendezvous.
        for key in tuple(os.environ):
            if key in {
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_WORLD_SIZE",
            } or key.startswith("TORCHELASTIC_"):
                os.environ.pop(key, None)
        os.environ.update(
            {
                "CUDA_VISIBLE_DEVICES": str(device),
                "SPECFORGE_ON_POLICY_ROOT": root,
                "SPECFORGE_ON_POLICY_WORKER": str(worker_id),
                "SPECFORGE_ON_POLICY_COLOCATED": (
                    "1"
                    if configuration["rollout"].get("placement") == "colocated"
                    else "0"
                ),
                "SGLANG_RAGGED_VERIFY_MODE": "static",
                "SGLANG_DSPARK_FOLDED_PROPOSAL": "0",
                "SGLANG_DSPARK_FAST_SAMPLING": "0",
                "SGLANG_SIMULATE_ACC_LEN": "0",
            }
        )
        from importlib.metadata import version

        if version("sglang") != "0.5.18":
            raise RuntimeError("on-policy hooks require patched sglang==0.5.18")
        from sglang import Engine
        from sglang.srt.managers.scheduler import Scheduler

        if not hasattr(Scheduler, "specforge_on_policy"):
            raise RuntimeError("apply patches/sglang/v0.5.18/on-policy.patch first")
        model, rollout = configuration["model"], configuration["rollout"]
        from .config import OnPolicyConfig

        cfg = OnPolicyConfig.model_validate(configuration)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        engine = Engine(
            port=port,
            model_path=model["target_model_path"],
            speculative_draft_model_path=str(Path(root) / "weights" / "00000000"),
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=block_size + 1,
            dtype="bfloat16",
            tp_size=1,
            trust_remote_code=model["trust_remote_code"],
            download_dir=model["cache_dir"],
            skip_tokenizer_init=True,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            max_running_requests=1,
            chunked_prefill_size=-1,
            context_length=rollout["context_length"],
            mem_fraction_static=rollout["mem_fraction_static"],
            attention_backend=rollout["attention_backend"],
            random_seed=configuration["training"]["seed"] + worker_id,
        )
        _idle_rpc(engine, rollout["timeout_s"], operation="sync", version=0)
        connection.send_bytes(json.dumps({"ok": True, "version": 0}).encode())
        while True:
            request = json.loads(connection.recv_bytes())
            operation = request["operation"]
            if operation == "close":
                break
            if operation == "sync":
                _idle_rpc(
                    engine,
                    rollout["timeout_s"],
                    operation="sync",
                    version=request["version"],
                )
                result = {"ok": True, "version": request["version"]}
            elif operation == "rollout":
                response = engine.generate(
                    input_ids=request["input_ids"],
                    sampling_params=cfg.sampling_for_prompt(len(request["input_ids"])),
                    rid=request["request_id"],
                )
                if (
                    "output_ids" not in response
                    or response.get("meta_info", {})
                    .get("finish_reason", {})
                    .get("type")
                    == "abort"
                ):
                    raise RuntimeError(
                        f"SGLang did not complete the rollout: {response.get('meta_info')}"
                    )
                _idle_rpc(
                    engine,
                    rollout["timeout_s"],
                    operation="drain",
                    version=request["version"],
                    request_id=request["request_id"],
                    output_ids=response["output_ids"],
                )
                result = {"ok": True, "request_id": request["request_id"]}
            else:
                raise ValueError("unknown engine operation")
            connection.send_bytes(json.dumps(result).encode())
    except BaseException:
        try:
            connection.send_bytes(
                json.dumps({"ok": False, "error": traceback.format_exc()}).encode()
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if engine is not None:
            engine.shutdown()
        connection.close()


class RolloutPool:
    def __init__(self, cfg, root, block_size):
        self.root = Path(root)
        self.timeout = cfg.rollout.timeout_s
        self.version = -1
        self.workers = []
        self.poisoned = False
        context = mp.get_context("spawn")
        try:
            for worker_id, device in enumerate(cfg.rollout.cuda_devices):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_engine_process,
                    args=(
                        child,
                        cfg.model_dump(),
                        str(self.root),
                        worker_id,
                        device,
                        block_size,
                    ),
                )
                process.start()
                child.close()
                self.workers.append((parent, process))
            for worker in self.workers:
                if self._receive(worker).get("version") != 0:
                    raise RuntimeError("initial draft synchronization failed")
            self._check_acks(0)
            self.version = 0
        except BaseException:
            self.close()
            raise

    def _receive(self, worker):
        connection, process = worker
        if not connection.poll(self.timeout):
            self.poisoned = True
            raise TimeoutError(f"SGLang worker {process.pid} exceeded {self.timeout}s")
        result = json.loads(connection.recv_bytes())
        if not result.get("ok"):
            self.poisoned = True
            raise RuntimeError(result.get("error", "SGLang worker failed"))
        return result

    def _check_acks(self, version):
        for index in range(len(self.workers)):
            ack = json.loads((self.root / "acks" / f"worker-{index}.json").read_text())
            if (
                ack["weight_version"] != version
                or not ack["cache_invalidated"]
                or ack["keys"] < 1
            ):
                raise RuntimeError("missing full-weight/cache acknowledgement")

    def generate(self, requests):
        if self.poisoned:
            raise RuntimeError("rollout pool failed; no further batches are allowed")
        if len(requests) > len(self.workers):
            raise ValueError("submit at most one request per rollout worker")
        try:
            for worker, request in zip(self.workers, requests):
                worker[0].send_bytes(
                    json.dumps(
                        dict(request, operation="rollout", version=self.version)
                    ).encode()
                )
            with ThreadPoolExecutor(max_workers=len(requests)) as pool:
                results = list(
                    pool.map(self._receive, self.workers[: len(requests)])
                )
            if any(
                result["request_id"] != request["request_id"]
                for result, request in zip(results, requests)
            ):
                raise RuntimeError("rollout worker returned a different request")
            return results
        except BaseException:
            self.poisoned = True
            raise

    def synchronize(self, version):
        if self.poisoned or version != self.version + 1:
            raise RuntimeError("invalid on-policy synchronization boundary")
        # Do not expose the new version until every worker loaded every weight
        # and invalidated its inference caches. A partial failure is terminal.
        try:
            for connection, _ in self.workers:
                connection.send_bytes(
                    json.dumps({"operation": "sync", "version": version}).encode()
                )
            with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
                responses = list(pool.map(self._receive, self.workers))
            if any(response.get("version") != version for response in responses):
                raise RuntimeError("rollout worker acknowledged a wrong version")
            self._check_acks(version)
            self.version = version
        except BaseException:
            self.poisoned = True
            raise

    def close(self):
        import psutil

        for connection, process in self.workers:
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants = []
            if process.is_alive():
                try:
                    connection.send_bytes(b'{"operation":"close"}')
                except (OSError, BrokenPipeError):
                    pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            # Engine shutdown normally reaps these. Bound cleanup after a
            # scheduler crash too, using only descendants of our owned worker.
            for child in descendants:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(descendants, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            connection.close()
