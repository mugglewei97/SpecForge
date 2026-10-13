"""Inspect/apply the small, source-checked SGLang DSpark replay patch.

Audited against release/v0.5.18 commit 71de97b264. Fail closed on unknown
worker layouts instead of assuming a generic weight-update API is sufficient.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

WORKER = "srt/speculative/dspark_components/dspark_worker_v2.py"
SCHEDULER = "srt/managers/scheduler.py"
MARKER = "# SpecForge DSpark replay protocol 1"
AUDITED_SOURCES = {
    WORKER: "c2e1333e75390f7e4785d59e3bddc3131eaec72e818b975020f6c1bb2452b560",
    "srt/speculative/dspark_components/dspark_verify.py": "534b8d81f26cad3fb066c131954bc897acdd4318910308701d791594c40a088e",
    "srt/models/dspark.py": "194759ffba28cc1e1d7d8ba28cb2f0134ec9f935eae34392bca2178c748b19ce",
    "srt/models/dflash.py": "b4c867543d80845bb621cb9bf93a0f9d8b09a6753f2a4d32e5e36d2122babc30",
    "srt/model_executor/model_runner_components/spec_aux_hidden_state.py": "db31c1d76466c6f99e5acdf9483a2a6beb4e6d8a6d96a07c17646257a61a7e9f",
}
EDITS = {
    WORKER: [
        (
            "        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:\n",
            "        " + MARKER + "\n"
            "        from specforge.inference.dspark_replay_server import register_requests\n"
            "        register_requests(self, batch)\n"
            "        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:\n",
        ),
        (
            "        if batch.return_logprob:\n            compute_spec_logprobs(\n",
            "        from specforge.inference.dspark_replay_server import record_round\n"
            "        record_round(self, batch, prefix_lens, draft_tokens, accept, layout)\n"
            "        if batch.return_logprob:\n            compute_spec_logprobs(\n",
        ),
    ],
    SCHEDULER: [
        (
            "    def flush_cache(self, empty_cache: bool = True):\n",
            "    " + MARKER + "\n"
            "    def specforge_dspark_replay(self):\n"
            "        from specforge.inference.dspark_replay_server import dispatch\n"
            "        dispatch(self)\n\n"
            "    def flush_cache(self, empty_cache: bool = True):\n",
        ),
    ],
}


def installed_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("SGLang is not installed in this Python environment")
    return Path(spec.origin).resolve().parent


def inspect_patch(root, *, apply=False, require=False):
    root = Path(root).resolve(strict=True)
    changes, report = {}, {}
    for relative, expected in AUDITED_SOURCES.items():
        text = (root / relative).read_text()
        for old, new in EDITS.get(relative, []):
            text = text.replace(new, old)
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"unaudited SGLang implementation: {relative} sha256={actual}; inspect this server revision before adapting the replay patch"
            )
    for relative, edits in EDITS.items():
        path = root / relative
        text = path.read_text()
        # Require all markers, not just one: partial installs cannot be used.
        states = [new in text for old, new in edits]
        if any(states) and not all(states):
            raise RuntimeError(f"partially installed replay patch: {path}")
        patched = all(states)
        if not patched:
            for old, new in edits:
                if text.count(old) != 1:
                    raise RuntimeError(f"unsupported SGLang source layout: {path}")
                text = text.replace(old, new, 1)
            changes[path] = text
        report[relative] = {
            "patched": patched,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    # Inspect the actual serving model and cache implementation too. A matching
    # package version by itself does not establish these contracts.
    for relative, symbols in {
        "srt/models/dspark.py": (
            "class Qwen3DSparkModel",
            "_fused_kv_write_cache",
            "_stacked_ctx_kv_cache",
        ),
        "srt/managers/scheduler.py": ("def is_fully_idle", "def handle_rpc_request"),
        "srt/entrypoints/engine.py": ("def collective_rpc", "def generate"),
    }.items():
        text = (root / relative).read_text()
        if any(symbol not in text for symbol in symbols):
            raise RuntimeError(f"unsupported SGLang replay capability: {relative}")
    if require and changes:
        raise RuntimeError(
            "SGLang has no complete per-round replay patch; run python -m specforge.inference.dspark_replay_patch --apply"
        )
    if apply:
        # Validate every file before any mutation. Each replacement is atomic.
        for path, text in changes.items():
            temporary = path.with_suffix(path.suffix + ".replay-tmp")
            if temporary.exists() or temporary.is_symlink():
                raise RuntimeError(f"patch temporary path already exists: {temporary}")
            temporary.write_text(text)
            temporary.replace(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sglang-root", type=Path, help="package directory containing srt/"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_patch(args.sglang_root or installed_root(), apply=args.apply),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
