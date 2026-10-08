import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from specforge.on_policy.config import OnPolicyConfig
from specforge.on_policy.rollout import RolloutPool, _engine_process
from specforge.on_policy.trainer import _prepare_rollout_phase, run


class ColocationTests(unittest.TestCase):
    def test_worker_binds_one_physical_gpu_and_drops_trainer_rendezvous(self):
        cfg = OnPolicyConfig.from_file("examples/on_policy/qwen3-8b-dspark-tv.yaml")
        observed = {}
        engine = Mock()

        def create_engine(**kwargs):
            observed.update(kwargs)
            observed["environment"] = dict(os.environ)
            return engine

        connection = Mock()
        connection.recv_bytes.return_value = b'{"operation": "close"}'
        with (
            patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                    "RANK": "0",
                    "WORLD_SIZE": "8",
                    "LOCAL_RANK": "0",
                    "LOCAL_WORLD_SIZE": "8",
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "12345",
                    "TORCHELASTIC_RUN_ID": "trainer",
                },
            ),
            patch.dict(
                "sys.modules",
                {
                    "sglang": SimpleNamespace(Engine=create_engine),
                    "sglang.srt.managers.scheduler": SimpleNamespace(
                        Scheduler=SimpleNamespace(specforge_on_policy=Mock())
                    ),
                },
            ),
            patch("importlib.metadata.version", return_value="0.5.18"),
            patch("signal.signal"),
        ):
            _engine_process(connection, cfg.model_dump(), "/unused", 5, 5, 7)
        self.assertEqual(observed["tp_size"], 1)
        self.assertEqual(observed["mem_fraction_static"], 0.3)
        environment = observed["environment"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(environment["SPECFORGE_ON_POLICY_WORKER"], "5")
        self.assertEqual(environment["SPECFORGE_ON_POLICY_COLOCATED"], "1")
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RUN_ID",
        ):
            self.assertNotIn(key, environment)
        self.assertEqual(
            json.loads(connection.send_bytes.call_args.args[0]),
            {"ok": True, "version": 0},
        )
        engine.shutdown.assert_called_once()

    def test_every_engine_must_ack_before_version_advances(self):
        pool = RolloutPool.__new__(RolloutPool)
        pool.version, pool.poisoned = 0, False
        connections = [Mock() for _ in range(8)]
        pool.workers = [(connection, None) for connection in connections]
        pool._check_acks = Mock()

        def receive(_worker):
            self.assertEqual(pool.version, 0)
            for connection in connections:
                self.assertEqual(
                    json.loads(connection.send_bytes.call_args.args[0]),
                    {"operation": "sync", "version": 1},
                )
            return {"version": 1}

        pool._receive = Mock(side_effect=receive)
        pool.synchronize(1)
        self.assertEqual(pool._receive.call_count, 8)
        pool._check_acks.assert_called_once_with(1)
        self.assertEqual(pool.version, 1)

    def test_phase_boundary_frees_unused_cuda_before_cpu_barrier(self):
        events, group = [], object()
        with (
            patch(
                "torch.cuda.synchronize",
                side_effect=lambda: events.append("synchronize"),
            ),
            patch(
                "torch.cuda.empty_cache", side_effect=lambda: events.append("release")
            ),
            patch(
                "torch.distributed.barrier",
                side_effect=lambda **kw: events.append(kw["group"]),
            ),
        ):
            _prepare_rollout_phase(None)
            self.assertEqual(events, [])
            _prepare_rollout_phase(group)
        self.assertEqual(events, ["synchronize", "release", group])

    def test_cpu_group_is_cleaned_up_on_failure(self):
        cfg = OnPolicyConfig.from_file("examples/on_policy/qwen3-8b-dspark-tv.yaml")
        group = object()
        with (
            patch("torch.distributed.new_group", return_value=group) as create,
            patch("torch.distributed.destroy_process_group") as destroy,
            patch(
                "specforge.on_policy.trainer._run", side_effect=ValueError("failed")
            ) as execute,
        ):
            with self.assertRaisesRegex(ValueError, "failed"):
                run(cfg)
        self.assertEqual(create.call_args.kwargs["backend"], "gloo")
        execute.assert_called_once_with(cfg, group)
        destroy.assert_called_once_with(group)


if __name__ == "__main__":
    unittest.main()
