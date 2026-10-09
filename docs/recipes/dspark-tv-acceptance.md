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
