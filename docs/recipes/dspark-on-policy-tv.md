# DSpark on-policy TV post-training

This opt-in mode loads an existing DSpark checkpoint, performs complete native
SGLang speculative decoding, and optimizes the fixed-trajectory acceptance-length
proxy. Existing `specforge train` recipes and their losses are unchanged. The
new command is `specforge train-on-policy`; the torchrun module is
`specforge.on_policy`. Both read the same dedicated typed recipe.

## Supported first validation target

Use SGLang **0.5.18** with the patch below, a dense `DSparkDraftModel` with
GQA/MHA and vanilla/gated/RNN Markov head, BF16, one trainer node, and one or
more independent single-GPU rollout engines. Qwen3-4B and Qwen3-8B recipes are
provided for H200. FSDP training may use dedicated GPUs or share each GPU with
one rollout engine. Each rollout GPU must fit the complete frozen target plus
draft, alongside the trainer in colocated mode. Target TP, MoE/MLA drafts,
vocabulary remapping, custom logit processors, penalties, string stops, quantized models,
and full optimizer-state resume are not implemented in this mode. Unknown
recipe fields fail validation instead of being silently ignored.

Rollout uses eager static full-width verification and disables prefix caching
and CUDA graphs. This deliberately provides an inspectable first correctness
gate; it is not a serving throughput benchmark. Full and sliding draft
attention are reconstructed with their respective inference masks.

## Install and launch on H200

Install this SpecForge checkout in the same Python environment as SGLang. Apply
the patch to the **v0.5.18 source checkout actually imported by that environment**:

```bash
git -C /path/to/sglang checkout v0.5.18
git -C /path/to/sglang apply --check /path/to/SpecForge/patches/sglang/v0.5.18/on-policy.patch
git -C /path/to/sglang apply /path/to/SpecForge/patches/sglang/v0.5.18/on-policy.patch
pip install -e /path/to/sglang/python --no-deps
pip install -e /path/to/SpecForge --no-deps
```

The patch is additive to the existing `spec-capture.patch`. Its hooks are
inactive unless `SPECFORGE_ON_POLICY_ROOT` is set by the new launcher. It adds
an engine-local scheduler RPC, not an HTTP service.

Copy `examples/on_policy/qwen3-4b-dspark-tv.yaml` and set the target, checkpoint,
dataset and new output directory. `draft_checkpoint_path` accepts the existing
SpecForge checkpoint format or an HF draft directory. Supply
`draft_model_config` for a checkpoint without `config.json`. For a custom
checkpoint, use its matching architecture config, not an unrelated stock one.

For a first smoke run set `training.max_steps: 2`, keep
`training.batch_size: 4`, and use a small representative dataset. With rollout
on physical GPU 0 and two FSDP ranks on GPUs 1 and 2:

```bash
specforge train-on-policy --config /path/to/run.yaml --plan
CUDA_VISIBLE_DEVICES=1,2 SPECFORGE_RUN_H200_GATE=1 \
  python -m unittest tests.test_on_policy.test_fsdp_cuda -v
CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
  -m specforge.on_policy --config /path/to/run.yaml
python -m specforge.on_policy.audit --run-dir /checkpoints/dspark-on-policy-tv
```

A direct one-trainer-GPU launch also works:

```bash
CUDA_VISIBLE_DEVICES=1 specforge train-on-policy --config /path/to/run.yaml
```

`rollout.cuda_devices` contains physical GPU ordinals. Trainer devices must be
explicit. `rollout.placement: dedicated` (the default) requires disjoint devices.
`rollout.placement: colocated` requires one TP=1 engine per trainer GPU, with
`rollout.cuda_devices` matching `CUDA_VISIBLE_DEVICES` in order. Multiple rollout
workers each load their own target and draft; every worker receives every
weight version, including workers idle in a
small batch. Run directories are never overwritten. Published
`weights/00000000` is the initial model; `weights/00000001`, etc. are complete
HF draft checkpoints suitable for the existing serving/export workflow. A
new run may warm-start from any of these; it starts a fresh optimizer.

## Converted Qwen3-8B eight-GPU server job

`examples/on_policy/qwen3-8b-dspark-tv.yaml` preserves the supplied job's target
and dataset paths, draft architecture config, output directory, learning rate
`6e-4`, six epochs, warmup ratio `0.04`, gradient norm `1.0`, total sequence limit
3072, Qwen template, block size 7, trainer FlexAttention, log interval 50, save
interval 2000 and SGLang memory fraction `0.3`. It warm-starts from:

```
/mnt/amed-s1/common/ckpt/gaochang/EagleModel/outputs/qwen3-8b-dspark-120w-mix/epoch_6_step_228768
```

After installing the SGLang patch above, run on the server:

```bash
cd /ossfs/workspace/SpecForge
bash scripts/train_dspark_on_policy_8gpu.sh --plan
bash scripts/train_dspark_on_policy_8gpu.sh
```

The launcher restores eight FSDP ranks on physical GPUs 0–7. Each of those
same GPUs also hosts one TP=1 SGLang rollout engine (eight engines total), with
`mem_fraction_static: 0.3` per engine. Rank 0 coordinates the engine subprocesses.
The original per-GPU batch size 4 on eight ranks is **global batch size 32** in
this entrypoint. Full batches assign four samples per FSDP rank; replay
accumulates their block gradients before the single optimizer step.
Rollout and replay execute in separate phases. Trainers use a CPU/Gloo group
to wait for rollout and all-worker weight synchronization, keeping NCCL waits
off the shared GPUs. Both processes release unused CUDA allocator blocks at
phase boundaries; live model weights and the SGLang KV pool remain resident.
The launcher keeps `SPECFORGE_DATA_NUM_PROC=32` for CPU prompt tokenization and
`FLASHINFER_DISABLE_VERSION_CHECK=1`.

In epoch mode, each length-eligible prompt is visited exactly once per epoch,
with a deterministic shuffle for each epoch. The final batch of each epoch may
be smaller. The scheduler horizon is `6 * ceil(eligible_prompts / 32)`; its
warmup uses the original 0.04 ratio. Immediate-EOS rollouts have no candidate
loss and are excluded from that batch's mean. An entirely empty batch fails
before an optimizer update. `schedule.json`, batch prompt IDs and metric sample
counts expose these boundaries. Set either `num_epochs` or `max_steps`, never
both. The old step-budget recipe remains supported.

`data.max_length: 3072` bounds prompt **plus** generated tokens. Per-request
generation is capped to `3072 - prompt_length`; prompts longer than 3070 are
filtered because they leave no anchor/candidate budget. Engine context 3088
includes scratch space for full speculative verification; it does not raise
the 3072 visible-token limit. The Qwen default system message is inserted only
when the data has no system message. The tokenizer's thinking default is kept.

The new stochastic sampling settings are temperature 1, no top-k/top-p
filtering, and EOS stopping. The original supervised job did not specify these
settings. `loss_decay_gamma=4.0` and `num_anchors=512` have no equivalent in this
objective: equal block means replace decay weighting, and native rollout
blocks replace sampled offline anchors. They are intentionally absent.

Full draft weights are still published and acknowledged **every step**.
`save_interval: 2000` controls retention: initial, every 2000th and latest/final
weights remain; superseded intermediate sync snapshots are removed only after
the next all-worker ACK. All trajectory JSON remains, including complete
rejected proposals. This long-run recipe sets `retain_replay_tensors: false`
to release full-vocabulary p/q and target features after the batch's completed
update and synchronization. Set it to true for a short full-replay audit.
The existing output directory may contain older offline checkpoints, but an
existing on-policy `run.json` or `weights/` makes the launcher refuse reuse.

## Objective and trajectory contract

For each block, `a_i = 1 - 0.5 * sum_v |p_i(v)-q_i(v)|` and
`loss = 1 - mean_k prod_{i<=k} a_i`. Products and TV are computed in FP32.
The sampled tokens are fixed: this is the requested proxy gradient, not an
unbiased gradient of true expected acceptance. No CE/confidence auxiliary
loss or decay weighting is added. The existing confidence head is frozen and
still fully synchronized/exported.

Input records are JSON arrays or JSONL with `messages` or ShareGPT
`conversations`. A record must end with an assistant turn; that turn is removed
before applying the target tokenizer's generation chat template. Earlier
assistant turns remain. Overlong prompts are filtered, not silently truncated.
The Qwen3-4B smoke recipe disables Qwen thinking explicitly. The converted
Qwen3-8B recipe preserves the tokenizer default from the original Qwen job.

SGLang performs the real proposal, target verification, accept/reject walk and
residual correction/bonus sampling. The hooks preserve the complete proposal,
including its discarded suffix, committed context, anchor, acceptance count,
correction/bonus token, masks, exact effective sampling parameters and weight
version. The full generated sequence is saved separately. Native target
verification already evaluates **every candidate prefix**, including rejected
branches. Its frozen p and target auxiliary features are retained for replay;
there is no second target model in the trainer and no substitute HF target
forward. Replay rebuilds draft embeddings, positions, masks and Markov previous
tokens, then computes q through FSDP with gradients.

In SGLang 0.5.18 DSpark, q applies **temperature only**; p applies temperature
then target top-k/top-p according to the native rejection-sampling helper.
Applying top-k/top-p to q would be incorrect for this engine. Greedy requests
(temperature zero or top-k 1) are rejected because their exact q has no useful
gradient. Additional unsupported sampling transforms cannot be configured.

The first proposal EOS remains valid; later positions are masked. Generation
budget truncation is measured from the actual anchor, never from the accepted
count. Thus rejection does not truncate loss. Blocks speculatively launched
after the final sequence ended are retained with an all-false mask. All-empty
rollouts are excluded from the effective batch's loss denominator. An entirely
empty batch stops before an optimizer update.

Each sample's valid blocks have equal weight, and valid samples have equal
weight. For world size W the per-block backward weight is `W/(N*M_n)`, because
FSDP averages gradients across ranks. Ranks with fewer blocks execute zero-loss
padding forwards, so all ranks enter the same collectives. `no_sync` encloses
forward and backward; one optimizer step follows the complete effective batch.
Parameters stay fixed throughout collection and replay.

Before each backward, replay q is compared to the recorded rollout q. A maximum
per-position TV above `training.replay_max_tv` aborts without stepping that
batch. The default 0.02 is an initial BF16 cross-kernel gate, not a measured
H200 tolerance; inspect the observed values and tighten it after validation.

After the optimizer step, all ranks gather the complete draft. Every rollout
worker loads every serving parameter, checks the packed tensor values, clears
the stacked/fused weight caches, flushes the request/KV pools and acknowledges
the version. Only after all acknowledgements does the next batch start. A
partial synchronization failure poisons the pool and stops training.

## Inspect results and measure actual acceptance

The run saves JSON metadata and safetensors under `trajectories/`, batch request
IDs under `batches/`, complete drafts under `weights/`, worker acknowledgements
under `acks/`, and per-step metrics under `metrics/`. The audit requires two
completed synchronized versions by default and at least one real rejection.
`actual_accepted_length` counts accepted draft candidates, excluding the
anchor/correction/bonus; proxy loss is reported separately.

Full-vocabulary p/q are saved to make replay auditable. Storage can be large:
two FP32 distributions cost `8*K*V` bytes per block, in addition to target
features. With `retain_replay_tensors: false`, this tensor storage is bounded
to the pending effective batch; proposal metadata is still retained. Set
`save_interval` to bound full-weight snapshot retention. Audit reports whether
it checked full replay tensors or only retained trajectory metadata. Keep the
first validation run short and size the output filesystem before extending it.

To compare initial and trained acceptance, copy the recipe and point
`data.train_data_path` to a **held-out** set. Keep target, sampling, block size
and prompt template fixed:

```bash
python -m specforge.on_policy.evaluate \
  --config /path/to/held-out.yaml \
  --before /checkpoints/dspark-on-policy-tv/weights/00000000 \
  --after /checkpoints/dspark-on-policy-tv/weights/00000002 \
  --output-dir /checkpoints/dspark-tv-evaluation --max-samples 32
```

`comparison.json` contains both measured lengths, their difference and per-sample
counts. Use more prompts and multiple seeds for a quality conclusion. A lower
proxy loss alone does not establish a higher actual acceptance length.

## Local checks

```bash
python -m unittest discover -s tests/test_on_policy -t . -v
```

These tests cover analytic loss/gradients, EOS/padding, real tiny DSpark replay
for all three Markov heads, a three-rank Gloo gradient-equivalence case with an
empty rank, discarded-proposal preservation, synchronization and failure gates.
CUDA/SGLang kernel parity and improvement in actual acceptance must be checked
on H200; passing CPU checks does not establish either.
