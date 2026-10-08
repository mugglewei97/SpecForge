import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.on_policy.config import OnPolicyConfig, SamplingConfig
from specforge.on_policy.data import (
    atomic_json,
    load_prompts,
    load_trace,
    prompt_messages,
    trace_stem,
)


class DataTests(unittest.TestCase):
    def test_qwen_system_prompt_and_parallel_preprocessing(self):
        def template(messages, **kwargs):
            assert "enable_thinking" not in kwargs
            assert all(m["content"] != "held out" for m in messages)
            # Two different sequences expose whether a dataset system was lost.
            return (
                [1, 2]
                if messages[0]["content"] == "You are a helpful assistant."
                else [3]
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            conversation = [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "held out"},
            ]
            path.write_text(
                json.dumps(
                    [
                        {"messages": conversation},
                        {
                            "messages": [
                                {"role": "system", "content": "custom"},
                                *conversation,
                            ]
                        },
                    ]
                )
            )
            tokenizer = SimpleNamespace(apply_chat_template=template)
            outputs = []
            for workers in ("1", "2"):
                with patch.dict("os.environ", SPECFORGE_DATA_NUM_PROC=workers):
                    outputs.append(
                        load_prompts(path, tokenizer, 10, chat_template="qwen")
                    )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual([p["input_ids"] for p in outputs[0]], [[1, 2], [3]])

    def test_only_held_out_assistant_removed(self):
        messages = [
            {"role": r, "content": str(i)}
            for i, r in enumerate(["system", "user", "assistant", "user", "assistant"])
        ]
        self.assertEqual(prompt_messages({"messages": messages}), messages[:-1])
        self.assertEqual(len(messages), 5)
        self.assertEqual(
            prompt_messages(
                {
                    "conversations": [
                        {"from": "human", "value": "q"},
                        {"from": "gpt", "value": "a"},
                    ]
                }
            ),
            [{"role": "user", "content": "q"}],
        )

    def test_ambiguous_prompt_rejected(self):
        for messages in ([], [{"role": "user"}], [{"role": "assistant"}]):
            with self.assertRaises(ValueError):
                prompt_messages({"messages": messages})

    def test_tokenizer_receives_generation_prompt_without_answer(self):
        seen = []

        def template(messages, **kwargs):
            seen.append((messages, kwargs))
            return [1, 2, 3]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "q"},
                            {"role": "assistant", "content": "SECRET ANSWER"},
                        ]
                    }
                )
                + "\n"
            )
            prompts = load_prompts(
                path, SimpleNamespace(apply_chat_template=template), 3
            )
        self.assertEqual(prompts[0]["input_ids"], [1, 2, 3])
        self.assertEqual(
            seen[0][1],
            {"tokenize": True, "add_generation_prompt": True, "return_dict": False},
        )
        self.assertNotIn("SECRET ANSWER", str(seen))

    def test_trace_version_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            rid = "../../unexpected"
            stem = trace_stem(directory, rid)
            self.assertEqual(stem.parent, Path(directory) / "trajectories")
            atomic_json(
                stem.with_suffix(".json"), {"request_id": rid, "weight_version": 2}
            )
            save_file({"p": torch.ones(1)}, str(stem.with_suffix(".safetensors")))
            self.assertEqual(
                load_trace(directory, rid, 2, metadata_only=True)[0]["weight_version"],
                2,
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                load_trace(directory, rid, 1)

    def test_config_fails_closed_on_sampling(self):
        for kwargs in (
            {"temperature": 0},
            {"top_k": 1},
            {"top_k": 0},
            {"top_p": 0},
            {"temperature": float("nan")},
            {"frequency_penalty": 1},
        ):
            with self.assertRaises(ValueError):
                SamplingConfig(**kwargs)
        cfg = OnPolicyConfig.from_file("examples/on_policy/qwen3-4b-dspark-tv.yaml")
        self.assertEqual(cfg.training.batch_size, 4)

    def test_real_tokenizer_return_contract(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(
            WordLevel({"[UNK]": 0, "q": 1, "assistant": 2}, unk_token="[UNK]")
        )
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        tokenizer.chat_template = "{% for m in messages %}{{ m.content }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "messages": [
                                {"role": "user", "content": "q"},
                                {"role": "assistant", "content": "held out"},
                            ]
                        }
                    ]
                )
            )
            prompts = load_prompts(path, tokenizer, 10)
        self.assertEqual(prompts[0]["input_ids"], [1, 2])
