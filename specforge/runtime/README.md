# SpecForge runtime

`specforge.runtime` contains the transport substrate shared by every
`specforge train` topology. Training and inference compute live in
`specforge.training` and `specforge.inference`; the removed
`specforge.runtime.{training,inference}` import trees are not retained.

The runtime has three responsibilities:

- `contracts.py` defines the metadata-only `PromptTask` and `SampleRef` records
  plus the tensor-carrying `TrainBatch` boundary.
- `control_plane/` owns prompt scheduling, online ref staging, the single
  online-consumer ledger, and optimizer-boundary DP acknowledgement.
- `data_plane/` owns feature stores, fixed offline refs, consume-once online
  channels, rank inboxes, and `FeatureDataLoader`.

All feature-based `specforge train` launchers converge on the same
`Trainer -> FeatureDataLoader -> TrainerController -> TrainerCore` lifecycle.
Offline refs remain a fixed list and never enter an online queue or ledger.
Online capture always runs in an external patched SGLang server and uses
`RefDistributor -> per-rank InboxChannel -> StreamingRefQueue`, including for a
single consumer rank. There is no colocated target-model or local-rollout path.

The opt-in `specforge train-on-policy` command has a separate synchronous
trajectory lifecycle in `specforge.on_policy`, reusing the FSDP backend and
checkpoint loading without entering the feature stream. Its isolated SGLang
engines synchronize at every optimizer boundary. See the
[DSpark on-policy recipe](../../docs/recipes/dspark-on-policy-tv.md).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the supported topology matrix and
cross-plane flow, [control_plane/DESIGN.md](control_plane/DESIGN.md) for ledger
and acknowledgement ownership, and [data_plane/DESIGN.md](data_plane/DESIGN.md)
for transport and cleanup semantics.

The dependency-light architecture gates can be run with:

```bash
python -m unittest \
  tests.test_runtime.test_package_architecture \
  tests.test_runtime.test_ref_distributor \
  tests.test_runtime.test_streaming_ref_channel
```
