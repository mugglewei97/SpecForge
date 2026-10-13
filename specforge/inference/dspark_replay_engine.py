"""One owned SGLang Engine per FSDP rank, on the same GPU.

The Engine owns one target copy used for both real decoding and replay capture.
Safe local files carry full tensors; no output text is tokenized on replay.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from specforge.data.dspark_replay import validate_trace
from specforge.inference.dspark_replay_patch import inspect_patch, installed_root


class SGLangReplayEngine:
    def __init__(self, args):
        self.args = args
        self.engine = None
        self.version = None
        self.capture_layer_ids = None
        self.failed = False
        self.directory = None

    def set_capture_layers(self, layer_ids):
        self.capture_layer_ids = list(layer_ids)

    def _rpc(self, action, **payload):
        if self.failed:
            raise RuntimeError("SGLang replay Engine failed; refusing further work")
        try:
            response_path = self.directory / "response.json"
            response_path.unlink(missing_ok=True)
            (self.directory / "request.json").write_text(
                json.dumps(dict(action=action, **payload))
            )
            self.engine.collective_rpc("specforge_dspark_replay")
            return json.loads(response_path.read_text())
        except BaseException:
            self.failed = True
            raise

    def sync(self, state_dict, config, trainable_names, version):
        try:
            if self.failed:
                raise RuntimeError("cannot synchronize a failed replay Engine")
            if self.version is not None and version != self.version + 1:
                raise RuntimeError(
                    "draft synchronization must advance exactly one version"
                )
            if self.engine is None:
                report = inspect_patch(installed_root(), require=True)
                self.directory = Path(
                    tempfile.mkdtemp(prefix="specforge-dspark-replay-")
                )
                self.block_size = int(config.block_size)
                (self.directory / "source-audit.json").write_text(
                    json.dumps(report, indent=2)
                )
                initial = self.directory / "initial"
                initial.mkdir()
                cfg = config.to_dict()
                cfg["architectures"] = ["Qwen3DSparkModel"]
                cfg.pop("auto_map", None)
                (initial / "config.json").write_text(json.dumps(cfg))
                save_file(state_dict, str(initial / "model.safetensors"))
                from sglang import Engine

                # Child schedulers inherit the rank-private root during spawn.
                previous = os.environ.get("SPECFORGE_DSPARK_REPLAY_DIR")
                os.environ["SPECFORGE_DSPARK_REPLAY_DIR"] = str(self.directory)
                try:
                    self.engine = Engine(
                        model_path=self.args.target_model_path,
                        speculative_algorithm="DSPARK",
                        speculative_draft_model_path=str(initial),
                        speculative_num_draft_tokens=int(config.block_size) + 1,
                        tp_size=1,
                        dp_size=1,
                        base_gpu_id=torch.cuda.current_device(),
                        dtype="bfloat16",
                        trust_remote_code=self.args.trust_remote_code,
                        mem_fraction_static=self.args.sglang_mem_fraction_static,
                        attention_backend=self.args.sglang_attention_backend,
                        enable_nccl_nvls=self.args.sglang_enable_nccl_nvls,
                        enable_symm_mem=self.args.sglang_enable_symm_mem,
                        disable_cuda_graph=True,
                        disable_overlap_schedule=True,
                        disable_radix_cache=True,
                        chunked_prefill_size=-1,
                        enable_torch_compile=False,
                        context_length=self.args.sglang_context_length
                        or self.args.max_length,
                        max_running_requests=max(
                            self.args.batch_size * self.args.accumulation_steps, 1
                        ),
                        watchdog_timeout=self.args.rollout_timeout,
                        skip_tokenizer_init=True,
                        enable_return_hidden_states=True,
                        log_level="warning",
                    )
                    import zmq

                    self.engine.send_to_rpc.setsockopt(
                        zmq.RCVTIMEO, self.args.rollout_timeout * 1000
                    )
                finally:
                    if previous is None:
                        os.environ.pop("SPECFORGE_DSPARK_REPLAY_DIR", None)
                    else:
                        os.environ["SPECFORGE_DSPARK_REPLAY_DIR"] = previous
            save_file(state_dict, str(self.directory / "weights.safetensors"))
            result = self._rpc(
                "sync",
                draft_version=version,
                target_version=self.args.rollout_target_version,
                trainable_names=sorted(trainable_names),
                capture_layer_ids=self.capture_layer_ids,
            )
            if (
                result.get("protocol") != 1
                or result.get("draft_version") != version
                or not result.get("cache_flushed")
                or set(result.get("loaded_names", [])) != set(state_dict)
                or set(result.get("trainable_names", [])) != set(trainable_names)
            ):
                raise RuntimeError(
                    "SGLang draft synchronization acknowledgement is incomplete"
                )
            self.version = version
            return version
        except BaseException:
            self.failed = True
            raise

    def rollout(self, prompts, version):
        if self.version != version:
            raise RuntimeError("rollout draft_version mismatch")
        requests = []
        for sample_id, tokens in prompts:
            remaining = self.args.max_length - len(tokens)
            if not tokens or remaining <= 0:
                raise ValueError("rollout prompt must leave space within max-length")
            sampling = dict(
                temperature=self.args.rollout_temperature,
                top_p=self.args.rollout_top_p,
                top_k=self.args.rollout_top_k,
                max_new_tokens=min(self.args.rollout_max_new_tokens, remaining),
            )
            requests.append(
                dict(
                    sample_id=sample_id,
                    input_ids=tokens,
                    sampling_config={
                        **sampling,
                        "speculative": {
                            "algorithm": "DSPARK",
                            "num_draft_tokens": self.block_size + 1,
                        },
                    },
                )
            )
        self._rpc("begin", draft_version=version, requests=requests)
        try:
            output = self.engine.generate(
                input_ids=[r["input_ids"] for r in requests],
                rid=[r["sample_id"] for r in requests],
                sampling_params=[
                    {
                        k: v
                        for k, v in r["sampling_config"].items()
                        if k != "speculative"
                    }
                    for r in requests
                ],
            )
            records = self._rpc("finish")["records"]
            if len(records) != len(requests) or len(output) != len(requests):
                raise RuntimeError("incomplete SGLang rollout batch")
            by_id = {r["sample_id"]: r for r in records}
            records = [by_id[r["sample_id"]] for r in requests]
            for record, result in zip(records, output):
                validate_trace(
                    record,
                    draft_version=version,
                    target_version=self.args.rollout_target_version,
                )
                if (
                    "output_ids" not in result
                    or result["output_ids"]
                    != record["input_ids"][record["prompt_length"] :]
                ):
                    raise RuntimeError("SGLang raw output IDs differ from verify trace")
            return records
        except BaseException:
            self.failed = True
            raise

    @torch.no_grad()
    def generate_dflash_data(self, input_ids, attention_mask, loss_mask):
        result = self._rpc(
            "capture",
            draft_version=self.version,
            input_ids=input_ids.cpu().tolist(),
            attention_mask=attention_mask.cpu().tolist(),
            loss_mask=loss_mask.cpu().tolist(),
        )
        if result["draft_version"] != self.version:
            self.failed = True
            raise RuntimeError("target capture version acknowledgement mismatch")
        tensors = load_file(str(self.directory / "capture.safetensors"))
        (self.directory / "capture.safetensors").unlink()
        return SimpleNamespace(**tensors)

    def close(self):
        if self.engine is not None:
            self.engine.shutdown()
        if self.directory is not None:
            shutil.rmtree(self.directory)
