# DSpark TV 接收长度代理目标

使用恢复的 `scripts/train_dspark.py` 入口：

```bash
bash examples/run_qwen3_8b_dspark_tv_acceptance.sh 8 flex_attention sglang
```

脚本沿用 Qwen3-8B、70w 数据、6 epochs、batch size 4、学习率 `6e-4`、
block size 7 和 512 anchors，输出到独立的 `qwen3-8b-dspark-tv-acceptance` 目录。
也可以在原训练命令后增加：

```bash
--dspark-loss-type tv-acceptance \
--tv-sampling-temperature 1.0 \
--tv-objective-chunk-blocks 8 \
--tv-verification-batch-size 4
```

## 数学与前缀语义

在实际 draft 候选前缀下，使用完整词表分布
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
固定采样 token 后求这个代理目标的梯度，不包含采样分布的 score-function 项，
不声称是真实期望接收长度的无偏梯度。

每个 anchor 单独构造原上下文和候选序列：对于 anchor `s` 和有效长度 `K`，
target 输入为 `input_ids[:s+1] + sampled_tokens[:K-1]`；
最终 hidden state 的 `s..s+K-1` 行预测这 K 个候选。
不同 block 相互独立。原始答案后缀的 hidden states 不用于这个目标。
draft 的 vanilla Markov head 使用上一步实际采样 token，采样与损失均使用相同温度。

首个无效 mask 位置截断 block；EOS 本身计入，EOS 后续位置不计入。
`K` 是截断后的有效候选数，空 block 不计入均值。
不包含 target 的额外 bonus token。FP16/BF16 的 softmax、TV 和连乘使用 FP32。

## 实现范围与开销

- 默认 `--dspark-loss-type ce-l1` 保留原有 loss。新模式只优化上述代理目标；
  `--loss-decay-gamma`、CE/L1/confidence 权重和 PACE 权重不参与新 loss。
  confidence head 冻结，target 始终不接收梯度。
- 当前支持给定配置的 vanilla Markov head（或关闭 Markov），`T > 0`，
  全词表采样；不包含 greedy、top-k、top-p 或实验性 CARH/selector/refiner。
- 当前要求 `--tp-size 1`、`--accumulation-steps 1`、`--micro-batch-size 0`。
  支持 torchrun/FSDP 数据并行，按所有 rank 的有效 block 总数归一化。
  全部 rank 都没有有效 block 时明确报错。
- sampling 和 target capture 在 checkpoint 之外执行一次，反向只重算
  确定性的 logits/loss。`--tv-objective-chunk-blocks` 控制词表张量峰值；
  `--tv-verification-batch-size` 控制 target 重算批量。
- 每个有效 anchor 需要独立 target prefill。batch 4、512 anchors 最多产生
  2048 个验证请求/rank/step，代价明显高于 teacher forcing；chunk 仅限制峰值，
  不减少这些请求。首次远程验证可以用更少 anchors 和 `--max-steps` 做 smoke test。

日志：`tv_acceptance_loss`、`tv_acceptance_length_proxy`、`tv_mean_overlap`、
`tv_valid_blocks`、`tv_mean_valid_length`。代理长度不是 SGLang 实测接收长度。
原有 `accuracy` 日志在此模式表示候选前缀上 draft/target 的 argmax 一致率；
position loss 是对应位置的 TV。

改变 objective 或采样配置不能静默恢复 optimizer 状态。
从旧 checkpoint 切换 loss 应使用 `--init-draft-model-path`，开始新的训练目录。

## 源码兼容与验证

旧入口所需缺失模块从用户指定的 `AQ/SpecForge` 仓库复制。
与当前统一训练接口冲突的 draft model、optimizer 和 dataloader 实现保存在
`specforge/legacy/`，入口显式导入它们；当前统一训练模块没有被覆盖。
checkpoint 导出复制这套兼容模型及其相对导入依赖。
SGLang in-process backend 也来自 AQ；远程需要其对应的 SGLang 环境。
本次没有更改项目依赖版本，也没有在个人电脑启动 GPU 训练。

CPU 测试：

```bash
python -m pytest -q tests/test_utils/test_tv_acceptance.py
```

覆盖公式、梯度有限差分、空 block、mask 截断、EOS、温度、低精度，
候选前缀和 target hidden 对齐，checkpoint 重算一致性，以及真实小型 DSpark
的前向反向。远程 8 卡 FSDP/SGLang 吞吐和显存仍需服务器 smoke test。
