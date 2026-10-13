# DSpark ground-truth 前缀 TV 接收长度代理目标

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
检查全部文件后才修改两个插入点所在的源码文件。其他实现会明确失败，不能只改版本号
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
- anchor `s`、槽位 `j`：目标 `input_ids[s+j+1]`，Markov predecessor
  `input_ids[s+j]`，target final hidden `last_hidden_states[s+j]`。
  **接受长度仅作 telemetry，绝不截断训练 block**；仍计算最多 K 个最终序列位置。
- response mask 从 `prompt_length` 构造，可叠加 trace 的 `loss_mask` 和右侧 padding
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
8 卡的峰值显存、吞吐与 GPU 端到端行为尚未验证。首版用 safetensors 本地交换完整权重
和 capture hidden，重视同步可核验性，存在每步磁盘/CPU 传输开销。
