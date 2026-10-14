# DSpark TV 接收长度代理目标

默认 `--tv-prefix-mode teacher-forced` 使用最终序列前缀。真实 proposal 前缀训练使用
`--anchor-mode replay --tv-prefix-mode proposal`，见文末 proposal 模式说明。
默认长度估计器为 `--tv-length-estimator overlap-product`。在同步 proposal replay 中，
可选 `--tv-length-estimator proposal-survival`，保留具体 proposal token 与后续前缀的
接受关联，并修正 block 内采样分布的梯度；公式与适用条件见文末。

使用恢复的 `scripts/train_dspark.py` 入口：

```bash
bash examples/run_qwen3_8b_dspark_tv_acceptance.sh 8 flex_attention sglang
```

脚本沿用 Qwen3-8B、70w 数据、6 epochs、batch size 4、学习率 `6e-4`、
block size 7 和 512 anchors，输出到独立的 `qwen3-8b-dspark-tv-acceptance` 目录。
也可以在原训练命令后增加：

```bash
--dspark-loss-type tv-acceptance \
--tv-temperature 1.0 \
--tv-objective-chunk-blocks 8
```

## 从已有 DSpark checkpoint 后训练

指定具体 checkpoint 目录（包含 `config.json` 和模型权重），以及新的输出目录：

```bash
bash examples/run_qwen3_8b_dspark_tv_acceptance.sh 8 flex_attention sglang \
    --init-draft-model-path /path/to/dspark/epoch_5_step_20000 \
    --output-dir /path/to/dspark-tv-posttrain
```

也可直接在 `torchrun ... scripts/train_dspark.py` 命令后加入这两个参数，
并保留 `--dspark-loss-type tv-acceptance`。
启动脚本允许在末尾追加训练参数，覆盖示例默认的 `--learning-rate`、
`--num-epochs`、`--max-steps` 等设置。

- 从 checkpoint 加载 draft 权重和结构配置，覆盖通用 `--draft-config-path`。
  保留其中的 block size、target capture layers 和 mask token；显式
  `--mask-token-id` 仍可覆盖。target 模型需与该 draft 训练时使用的模型匹配。
- 支持旧的 `model_type=qwen3` 和原生 `model_type=dspark` 导出配置，
  模型权重支持 safetensors / PyTorch 权重文件及其分片。
  缺失或不匹配的源模型权重会报错，避免随机补权重后继续后训练。
- 只加载模型，不读取旧的 `training_state.pt`。optimizer、scheduler、epoch、
  global step 从新训练开始；后训练的学习率和训练时长由当前命令决定。
- 从 CE/L1 训练的 vanilla Markov DSpark 切换到 TV loss 使用此方式。
  已启动的 TV 后训练任务中断后，用该任务的输出目录和 `--resume` 恢复；
  此时移除 `--init-draft-model-path`。两者互斥。

## 数学与前缀语义

在 target 生成的训练序列上随机选取 anchors。每个 block 使用该序列的
ground-truth 前缀，不采样新的 draft token。使用完整词表分布
`p = softmax(target_logits / T)`、`q = softmax(draft_logits / T)`，
`a_i = 1 - 0.5 * sum_v |p_iv - q_iv|`，计算

```text
E_proxy = sum_{k=1..K} prod_{i=1..k} a_i
L_block = 1 - E_proxy / K
loss = sum_valid_blocks L_block / number_of_valid_blocks
```

实现借鉴 [D-PACE](https://arxiv.org/abs/2605.18810) 的前缀连乘信用分配思想，
以 TV 分布重叠替换 gold-token confidence，并直接反传整个连乘表达式。
它没有 detached position weights、额外 PACE smoothing 或 log-loss 替换。
这是训练数据路径上的接收长度代理目标，不声称是真实推理期望接收长度
的无偏估计。保留原有 block 连乘公式，不改成逐 token TV 的简单平均。

target 对原始 batch 只 forward 一次，所有 anchors 复用其 hidden states。
对于 anchor `s`，block 槽位 `j=0..K-1` 的对齐关系为：

- 预测 token：`input_ids[s+j+1]`。
- Markov head 前驱：ground-truth `input_ids[s+j]`。
- target 分布：原始序列的 `last_hidden_states[s+j]` 经冻结的 LM head 得到。

draft 的预测与 ground truth 不一致时也不替换前驱；差异由 TV loss 惩罚。
target 的最终 hidden states 已经经过模型的 final norm，不重复归一化。
代码按 block 分块投影词表 logits，避免一次保留所有位置的完整词表分布。

首个无效 mask、padding 或序列尾部截断 block；ground-truth EOS 本身计入，
EOS 后续位置不计入。`K` 是截断后的有效预测位置数，空 block 不计入均值。
不包含 target 的额外 bonus token。FP16/BF16 的 softmax、TV 和连乘使用 FP32。

## 实现范围与开销

- 默认 `--dspark-loss-type ce-l1` 保留原有 loss。新模式只优化上述代理目标；
  `--loss-decay-gamma`、CE/L1/confidence 权重和 PACE 权重不参与新 loss。
  confidence head 冻结，target 始终不接收梯度。
- 当前支持给定配置的 vanilla Markov head（或关闭 Markov），`T > 0`，
  全词表 softmax；不包含实验性 CARH/selector/refiner。
- 当前要求 `--tp-size 1`、`--accumulation-steps 1`、`--micro-batch-size 0`。
  此限制适用于默认 `--anchor-mode random`；replay 模式支持按整个 optimizer step
  的有效 block 总数归一化的累积和 microbatch（见下文）。
  支持 torchrun/FSDP 数据并行，按所有 rank 的有效 block 总数归一化。
  全部 rank 都没有有效 block 时明确报错。
- target capture 在 checkpoint 之外执行一次，反向只重算确定性的 logits/loss。
  `--tv-objective-chunk-blocks` 控制词表张量峰值，显存允许时可测试 32/64
  以减少小批量调用。它不改变代理目标，可以在同一模式下 resume 时调整。
- batch 4、512 anchors 仍只需要原始 4 条训练序列的一次 target 批量 forward，
  不再产生额外 2048 个候选验证请求。draft 的 block 计算和完整词表 TV
  仍有开销，实际吞吐需在服务器测量。
- `--tv-temperature` 是 target/draft 的共同 softmax 温度；旧参数
  `--tv-sampling-temperature` 作为别名保留，但不再进行采样。
  `--tv-verification-batch-size` 为兼容旧命令继续接受，启动时提示已忽略。

日志：`tv_acceptance_loss`、`tv_acceptance_length_proxy`、`tv_mean_overlap`、
`tv_valid_blocks`、`tv_mean_valid_length`。代理长度不是 SGLang 实测接收长度。
原有 `accuracy` 日志在此模式表示 ground-truth 前缀上 draft/target 的 argmax 一致率；
position loss 是对应位置的 TV。

改变 objective 或 TV 温度不能静默恢复 optimizer 状态。新 checkpoint 的
训练参数保存 `tv_prefix_mode=teacher-forced`；旧的 sampled-prefix TV
checkpoint（没有这个标记）不能直接 `--resume` 到新语义。
从旧 CE/L1 或 sampled-prefix TV checkpoint 切换时使用
`--init-draft-model-path`，开始新的训练目录和 optimizer/scheduler。

## 源码兼容与验证

旧入口所需缺失模块从用户指定的 `AQ/SpecForge` 仓库复制。
与当前统一训练接口冲突的 draft model、optimizer 和 dataloader 实现保存在
`specforge/legacy/`，入口显式导入它们；当前统一训练模块没有被覆盖。
checkpoint 导出复制这套兼容模型及其相对导入依赖。
SGLang in-process backend 也来自 AQ；远程需要其对应的 SGLang 环境。
本次没有更改项目依赖版本，也没有在个人电脑启动 GPU 训练。

训练入口和实现按职责拆分，原命令与参数保持兼容：

- `scripts/train_dspark.py`：解析参数并启动训练；`--help` 不加载 GPU 训练依赖。
- `specforge/legacy/dspark_training/`：参数声明、配置合并、模型/数据构建、
  checkpoint 导出与训练循环；`objective.py` 集中处理 TV 参数、resume 检查和启用逻辑。
- `specforge/core/tv_acceptance.py`：ground-truth 前驱与 target hidden 对齐、
  TV 公式、checkpoint 分块和全局归约。

CPU 测试：

```bash
python -m pytest -q tests/test_utils/test_tv_acceptance.py tests/test_scripts/test_train_dspark.py
```

覆盖公式、梯度有限差分、空 block、mask 截断、EOS、温度、低精度，
多 batch/anchor 的 target hidden 对齐、真实前驱、checkpoint 梯度一致性，
以及真实小型 DSpark 的前向反向。训练 batch 测试检查 target 只调用一次、
反向不重跑 target、前后向都不采样 token。远程 8 卡 FSDP/SGLang 吞吐和显存
仍需服务器 smoke test。

## SGLang 真实 rollout anchors 的严格同步 replay

`--anchor-mode random` 是默认值，保留上述训练数据随机 anchors 路径。
`--anchor-mode replay` 在同一训练入口、同一 FSDP optimizer 中启用：

```
初始完整 draft V0 → SGLang 加载、逐张量回读确认
  → 用 V0 生成整个 optimizer step 的 rollout batch
  → 原始 tokens + 逐轮 anchors + 版本写入 JSONL
  → 按 microbatch 一次 target capture / sequence → 并行 blocks → backward
  → optimizer.step → 完整 draft V1 → SGLang 回读确认 + 所有 rank 同步
  → 才开始 V1 的下一批 rollout
```

部署仍是 8 卡 colocated：每个 torchrun/FSDP rank 所在 GPU 上运行一个
TP=1、DP=1 的 owned SGLang Engine。该 Engine 中的**同一份 target**负责
speculative rollout 和 replay capture，不额外加载第二份 capture target。
为确保 first version 的同步和缓存语义，关闭 SGLang overlap、CUDA graph
和 torch.compile。FSDP attention backend 仍可选 `sdpa`。
实现不启动 Ray，也不调用 `pkill`。

### 安装 SGLang 0.5.18 replay 补丁

开发与本地环境均已确认使用 SGLang `0.5.18`。本地审阅的源码基线是
`release/v0.5.18` commit `71de97b264`，其中
`DSparkWorkerV2` 没有训练所需的逐轮 trace；已有 spec-capture prefill 补丁不满足要求。
补丁工具核对 draft/verify/model 的源码 SHA-256，以及 scheduler/Engine 接口，
检查全部文件后才修改 worker 和 scheduler 两个源码文件。其他实现会明确失败，不能只改版本号
或跳过校验就启动训练。下列命令在训练环境检查实际安装源码并安装补丁。

```bash
cd /ossfs/workspace/SpecForge
# 使用训练环境的 Python；确保该 checkout 可被 SGLang 子进程 import。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m specforge.inference.dspark_replay_patch         # 只检查/报告
python -m specforge.inference.dspark_replay_patch --apply # 匹配基线后安装；可重复执行
```

新增 scheduler RPC 仅供训练创建的本机 Engine 使用，不开放 HTTP 接口。
没有假设通用热更新接口会更新 DSpark：RPC 只复制完整 draft state，检查参数全集、
shape/dtype、所有 trainable 参数清单（包括 Markov），回读验证，并清空 KV/radix
状态和 `_fused_kv_write_cache`、`_stacked_ctx_kv_cache`。
QKV/MLP 的融合权重按已核实的 TP=1 布局组装；无法映射的参数直接报错。
任何同步或版本确认失败都会停止所有训练 rank；已失败的 Engine 不允许继续生成。

### 训练命令

```bash
bash examples/run_qwen3_8b_dspark_tv_replay.sh 8 sdpa \
  --rollout-target-version YOUR_IMMUTABLE_TARGET_CHECKPOINT_ID \
  --accumulation-steps 2 --micro-batch-size 1 \
  --max-steps 2 --log-interval 1
```

脚本沿用给定服务器的模型、数据、初始化 checkpoint 和 `mem-fraction-static=0.3`。
该 replay 示例现在默认启用 `--tv-prefix-mode proposal --replay-tensor-transport cuda-ipc`；
在命令末尾追加 `--tv-prefix-mode teacher-forced` 可运行前缀对照组。
先跑两个 optimizer steps，检查日志中同步版本依次为 V1、V2；再去掉 `--max-steps`。
`--batch-size 4 --accumulation-steps 2` 表示每 rank 每个更新采集 8 条 rollout。
整批采集完成后才训练；不会提前生成下一步的数据。末尾不足一个累积窗口也完成一次更新。
replay 模式中 `--max-steps`、日志/保存间隔和 checkpoint step 使用 optimizer step 单位。
首版从 `--init-draft-model-path` 新建 optimizer；暂不支持 replay `--resume`。

默认 `--rollout-prompt-format chat` 复用现有 prompt 预处理，以第一段 response
监督区间之前的 tokens 作为 prompt，原数据 completion 不参与 replay。
也可用 `--rollout-prompt-format tokens` 直接读取 JSONL：

```json
{"sample_id":"prompt-1","input_ids":[1,23,45],"prompt_length":3}
```

生成后的序列一律直接读取 token IDs，不重新 tokenize，也不再次应用 chat template。
`rollout_traces/<run-id>/step-XXXXXXXX-rank-XXXXX.jsonl` 保存完整协议字段，
以及各 rank 的 SGLang 源码检查报告。`RolloutTraceDataset` 可读取这些记录用于审计；
训练控制器不将历史 trace 文件作为可跨更新重复使用的训练集。

### 位置、计数和 loss

- `anchor_positions` 是完整序列中的 0-based 绝对位置，在实际 draft/verify 路径
  直接读取 `prefix_lens`。SGLang prefill 先生成 root，因此第一轮通常以第一个 response
  token 为 anchor；该 root 已提交，不计入该轮新提交数。不能用接受率均值推导 anchors。
- 每轮 `draft_length` 记录实际 proposal 宽度，`verified_draft_tokens` 记录 verification
  窗口，`accepted_draft_tokens` 记录 verifier 接受的 draft 数。
  `verifier_committed_tokens` 是 verifier 的 draft+correction/bonus 输出长度；
  `committed_tokens` 是按真实 EOS/stop/max-length 截断后的最终提交长度。
  `root_tokens=0`；实际提交的 correction/bonus 另有计数，停止位置之后的 token 不计入。
- 在 `teacher-forced` 模式，anchor `s`、槽位 `j`：目标 `input_ids[s+j+1]`，Markov predecessor
  `input_ids[s+j]`，target final hidden `last_hidden_states[s+j]`。
  **接受长度仅作 telemetry，绝不截断训练 block**；仍计算最多 K 个最终序列位置。
- 在 `teacher-forced` 模式，response mask 从 `prompt_length` 构造，可叠加 trace 的 `loss_mask` 和右侧 padding
  `attention_mask`。EOS 本身有效，之后无效。anchor 本身无需属于 response。
- anchors 使用 padding + keep mask，超过预算时从所有有效 anchors 中均匀无放回抽样。
  重叠窗口仍用原有 block attention 隔离。所有 microbatches 共用整步所有 rank 的
  有效 block 总数归一化，不再平均各 microbatch 的 block 均值。
- target capture 位于 checkpoint 外；target hidden 与 anchors 停止梯度，backward
  只重算 draft/head/loss。真实 rollout 的多轮 target verification 不属于 replay 的
  “每条完整序列一次 target forward”计数。

CPU 回归：

```bash
python -m pytest -q tests/test_utils/test_tv_acceptance.py \
  tests/test_scripts/test_train_dspark.py tests/test_utils/test_dspark_replay.py \
  tests/test_runtime/test_dspark_replay_server.py
```

包括真实小型 DSpark 的重叠并行/逐块梯度对比、接受长度不截断窗口、跨 microbatch
精确归一化、末尾不足累积窗口、target capture 次数、同步故障、Markov 权重覆盖、
派生缓存失效、trace correction/bonus/stop 对齐和补丁幂等性。
CPU 验证不替代服务器的 CUDA/SGLang/FSDP 两步 smoke test；当前没有服务器连接信息，
8 卡的峰值显存、吞吐与 GPU 端到端行为尚未验证。

### 用 CUDA IPC 减少权重和 hidden 传输

原训练命令可追加 `--replay-tensor-transport cuda-ipc`。CLI 默认仍为 `files`，replay 示例使用 IPC，
可用相同 checkpoint、数据、batch、anchors 和 seed 做对照。此选项只适用于
Linux CUDA 下每个 FSDP rank 与其 owned SGLang Engine 共用同一 GPU 的部署。
不改变 TV 公式、训练 block、优化器更新频率或采样参数。

| 数据路径 | `files` | `cuda-ipc` |
| --- | --- | --- |
| 每步完整 draft 导出 | FSDP 汇集并 offload 到 CPU | FSDP 汇集到 GPU |
| draft 同步 | 写/读 `weights.safetensors`，复制到 GPU | JSON 传 CUDA IPC handle，直接在 GPU 复制并回读验证 |
| target capture 输出 | GPU → CPU → safetensors → CPU → GPU | Engine GPU → 同卡训练进程 GPU clone |

IPC 仍使用很小的 `request.json` / `response.json` 作为控制消息。
Engine 启动时的初始 checkpoint、正式训练 checkpoint 和 rollout trace 仍写文件。
同步继续使用专用 DSpark 映射及版本确认，不调用可能转发到 target 的通用权重更新接口。
此实现直接共享完整 state 的 storage，尚未改成逐桶 FSDP 汇集；仍有 full-state
all-gather 和额外 GPU 权重副本。capture 接收方 clone 完并等待 CUDA 完成后才 ACK
释放发送方 buffer，使 checkpoint backward 持有的是训练进程自己的张量。
这些副本会增加峰值显存；出现显存不足时可显式退回 `files`，不会静默回退。

先在训练环境验证真实进程间传输：

```bash
python -m pytest -q tests/test_runtime/test_dspark_replay_ipc.py
```

再以原训练命令分别运行短测（分别使用新的输出目录）：

```bash
# 在原命令末尾追加，保留原本的学习率、mem-fraction-static 等参数。
--replay-tensor-transport files --max-steps 20 --log-interval 1 \
  --output-dir /path/to/replay-benchmark-files

--replay-tensor-transport cuda-ipc --max-steps 20 --log-interval 1 \
  --output-dir /path/to/replay-benchmark-ipc
```

先做单卡两步 smoke，再在实际 8 卡配置比较；检查权重版本递增、loss 正常和实际
接受长度没有异常。排除启动及前几步，比较后续 step 耗时与吞吐；相同 seed 不保证
两次随机 rollout 的 token 序列完全一致，应比较多步结果。通过 CUDA 测试只确认
底层共享可用，不能替代 SGLang + FSDP 训练测试。

日志新增：

- `perf/step_seconds`：当前 optimizer step 总耗时，不包含启动和 checkpoint 保存。
- `perf/rollout_seconds`、`target_capture_seconds`、`forward_backward_seconds`、
  `optimizer_seconds`、`weight_export_seconds`、`weight_sync_seconds`（均带 `perf/` 前缀）：
  分阶段耗时，取所有 rank 最大值。capture 包含 target 前向和传输；
  forward/backward 不含本 rank 的 capture 调用耗时，但可能包含等待其他 rank。
  不同阶段的最大值可能来自不同 rank，因此不必加和为 step 总耗时。
- `perf/rollout_tokens_per_second`：所有 rank 的新生成 token 数 / 最慢 rank rollout 秒数。
  `perf/output_tokens_per_step_second` 则除以完整训练 step 秒数，用于评估整体训练速度。
- `rollout/accepted_draft_tokens_per_round`：真实 rollout 每轮接受且最终提交的 draft token 数；
  `draft_acceptance_rate` 为其总数 / verified draft token 总数；
  `committed_tokens_per_round` 包含 correction/bonus（均带 `rollout/` 前缀）。
  三者汇总全部 rank，并按实际 stop 截断，区别于 teacher-forced TV 的代理长度。
  这些是训练 prompt 上的在线统计，最终效果仍需固定的独立评测集确认。

### 真实 proposal 前缀模式

```bash
# 已安装 protocol 1 的训练环境也需重新执行；工具支持校验后升级到 protocol 2。
python -m specforge.inference.dspark_replay_patch --apply
bash examples/run_qwen3_8b_dspark_tv_replay.sh 1 sdpa --max-steps 2 --log-interval 1 \
  --output-dir /path/to/proposal-tv-smoke
```

`--tv-prefix-mode proposal` 复用 rollout 的完整原始 proposal `d1…dK`：

- Backbone 仍使用 root/mask 和原有 block attention；proposal 不进入双向注意力输入。
  Vanilla Markov 前驱改为 `[root, d1, …, d(K-1)]`。
- 在 verification 的 target logits processor 输入处复制 **final-norm hidden**，
  `[root, d1, …, dK]` 的前 K 行分别预测 `d1…dK`，最后一行 bonus 不参与 loss。
  不使用多层 aux hidden 或 selected-token log-prob 重建 teacher 分布。
- 有效长度取实际 verification 覆盖、剩余生成预算和 proposal 自身首个 EOS 的交集。
  包含 EOS 本身；拒绝位置、correction token、最终提交序列长度均不裁剪 proposal suffix。
  接受长度仅用于统计，TV 仍比较完整词表。
- 每个请求在 verification 中做均匀 reservoir sampling，最多保存 `--num-anchors` 个
  block 的 hidden。trace v2 保存每轮 proposal、valid mask、预算、round ID，以及选中的
  `selected_round_ids`；sample ID、round ID、anchor 和模型版本贯穿 capture/replay。
  Collator 与模型前向不再抽样，teacher hidden 使用完全相同的 block 顺序。
- 完整最终序列仍 capture 一次，为 draft 提供 context aux hidden；不逐 block 重跑 target。
  Teacher 分布由缓存 hidden 经冻结 LM head 分块重建，沿用原有 TV 连乘和全局 block 归一化。
  `files` 与 `cuda-ipc` 均支持，IPC clone 完成并同步后才释放发送方 allocation。
  server 的 selected cache 按 microbatch 消费；未消费完成禁止同步下一版本。

首版要求固定 K、static verification、vanilla Markov、相同且不小于 `1e-5` 的 rollout/TV 温度
（更低温度会被 SGLang 改写）、
`top_p=1 / top_k=-1`。拒绝 grammar、额外 logits processor/penalty、target logits
scale/softcap、compact verification，以及会把标签注入 backbone 的 source semantics。
旧 trace 或缺失 proposal/verification hidden 会报错；JSONL 用于审计，verification hidden
只在当前训练更新期间保留，不能仅用 JSONL 离线重放 proposal TV。

BF16 缓存按每 rank `B × A × K × H × 2` 字节计，B=4、A=512、K=7、H=4096
约 112 MiB；IPC clone、单轮临时 hidden 和训练激活另计。日志增加
`train/tv_position_1…K` 与 `perf/trainer_peak_allocated_bytes`（所有 rank 最大值，
仅训练进程的 PyTorch allocator，不包含独立 SGLang 进程或整卡占用）。

```bash
python -m pytest -q tests/test_utils/test_proposal_tv.py \
  tests/test_runtime/test_dspark_proposal_capture.py \
  tests/test_utils/test_tv_acceptance.py tests/test_utils/test_dspark_replay.py \
  tests/test_runtime/test_dspark_replay_server.py tests/test_runtime/test_dspark_replay_ipc.py \
  tests/test_scripts/test_train_dspark.py
```

CPU 测试覆盖同前缀 loss/梯度等价、拒绝后前驱、EOS/预算/覆盖、teacher 停止梯度、
final hidden 行对齐、reservoir 上限、身份校验、IPC 生命周期及旧补丁升级。
真实 SGLang GPU 上还需比较缓存重建 logits 与显式 proposal forward、同版本 draft
rollout/replay 分数，并做单卡/多卡两步 smoke。收益评估用独立 prompts、相同 GPU 时间，
比较实际接受长度和吞吐；默认 `overlap-product` 在固定 proposal 上反传仍是代理目标，
不是期望接受长度的无偏梯度。

### 按具体 proposal token 的生存概率估计接受长度

在上述 proposal replay 命令末尾追加：

```bash
--tv-length-estimator proposal-survival \
--output-dir /path/to/proposal-survival-posttrain
```

已有命令默认继续使用 `overlap-product`。新估计器要求 `tv-acceptance + replay + proposal`，
沿用完整词表、相同温度、逐步同步、每批 rollout 只更新一次的约束，不支持 dropout。
模型权重用 `--init-draft-model-path` 初始化，新建 optimizer/scheduler；切换估计器不能
静默恢复旧 optimizer 状态。无需新增 target forward 或修改 SGLang trace 补丁。

记 `d_i` 为原始 proposal，`p_i`、`q_i` 为对应 proposal 前缀上的 target/draft 分布，
`a_i = sum_v min(p_i(v), q_i(v))`，`q_old` 为采集 proposal 的 draft 分布。采用：

```text
w_i(theta) = min(p_i(d_i), q_theta_i(d_i)) / stopgrad(q_old_i(d_i))
R_hat(theta) = sum_{k=1..H} valid_k * [prod_{i<k} w_i(theta)] * a_k(theta)
L_block = 1 - R_hat / H
```

`H` 为固定的模型 block size（例如 7）。`valid_k` 按实际 verification 覆盖、剩余预算、
此前是否出现 proposal EOS 确定；当前位置 EOS 本身仍计入。空 block 不计入均值。
外层仍按整个 optimizer step、所有 rank 的非空 block 总数归一化。

在采集参数 `theta_old` 处，`w_i` 的前向值等于具体 token 的接受概率
`r_i = min(1, p_i(d_i)/q_old_i(d_i))`，所以前向长度为：

```text
R_hat = a_1 + r_1*a_2 + r_1*r_2*a_3 + ...
```

每一项只采样到倒数第二个位置，最后一个 token 用完整词表精确求平均。
对固定 round 起点和预算，若 proposal 确实来自 `q_old`，这个估计量的期望等于该轮
接受的 draft token 数（包含 EOS，排除 root/correction/bonus）。它保留了具体 token
的接受概率与后续前缀分布的关联。各长度项对末 token 的边缘化能去掉该项的一层采样噪声，
但不保证整个和式的方差一定低于所有其他估计器。

分母必须停止梯度，分子 `min(p,q_theta)` 和所有前缀乘积保留梯度。这等价于把 proposal
采样概率的变化纳入 block 内的导数。直接对固定样本上的 `min(1,p/q_theta)` 求导会遗漏
采样分布项：低估 token 的分支会没有接受率梯度，而高估分支只看到减小分母的收益。
把整个 `r_i` detach 也会漏掉前缀接受质量变化对后续长度的贡献。

实现以当前同版本、未更新的 replay `q_theta.detach()` 重建 `q_old`，用 log-softmax 和
log 概率差避免小概率除法下溢；没有概率下限、权重裁剪或自归一化。
**因此精确值/梯度结论以 rollout/replay 分布相同为前提**，不是异步或旧轨迹的离策略目标。
SGLang 和训练侧的 kernel、精度或模型实现差异仍需真实 GPU 上做 logits parity 检查。
该梯度也不包括 round 起点、跨轮状态占用和抽样权重的变化，不能称为完整生成过程的
无偏吞吐梯度。未来若重复更新同一批数据，必须实际保存采集时的 draft 概率。

新目标固定除以 `H`，不除以采样后 EOS 截断的 `K`。随机分母会改变目标，也会破坏上述
逐长度项边缘化的推导。例如 `H=7`、只剩一个有效位置且 `p=q`，新 loss 为 `6/7`，
而旧 loss 为 0；这反映当前最多只能接受一个 draft token。新旧 loss 数值不宜直接比较。

现有日志保持兼容：`tv_acceptance_length_proxy` 在此模式记录 `R_hat` 的 block 均值；
`tv_position_i` 和 `tv_mean_overlap` 仍是全词表 TV/overlap，不是具体 token 的接受率。
同时检查 `tv_mean_valid_length`、真实 `rollout/accepted_draft_tokens_per_round` 和独立
评测集吞吐。固定 H 时，`tv_acceptance_loss = 1 - tv_acceptance_length_proxy / H`。

CPU 精确性测试穷举小词表的完整 proposal 树，对比真实接受长度及其参数梯度，包含
条件分布随前缀变化、EOS、预算截断、单 token 退化为 TV、低精度和无效 padding：

```bash
python -m pytest -q tests/test_utils/test_proposal_survival.py tests/test_scripts/test_train_dspark.py
```
