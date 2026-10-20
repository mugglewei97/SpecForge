# DSpark TV 接收长度训练：teacher-forced 与 beta

`tv-acceptance` 保留原 ground-truth 前缀代理目标；新增的
`tv-acceptance-beta` 在实际 proposal 前缀上优化固定轮次起点下的单轮
期望连续接受长度及其梯度。默认 `ce-l1` 不变。

## Beta：实际 proposal 前缀与采样分布梯度

在现有示例命令末尾覆盖 loss 类型和输出目录即可：

```bash
bash examples/run_qwen3_8b_dspark_tv_acceptance.sh 8 flex_attention sglang \
    --dspark-loss-type tv-acceptance-beta \
    --output-dir /path/to/dspark-tv-beta
```

从已有模型后训练时追加 `--init-draft-model-path /path/to/checkpoint`。
模型的实际 `block_size` 必须为 7（包括从 checkpoint 加载的配置），不能
只依据 CLI 的默认值判断。beta 支持 vanilla Markov head、无 Markov head，
以及下述单前驱 CARH；沿用 `T > 0`、全词表 softmax、`tp-size=1`、
`accumulation-steps=1`、`micro-batch-size=0` 的限制，并支持 FSDP 数据并行。

### 从 CARH + DynamicConv checkpoint 后训练

`scripts/train_dspark.py` 的 beta 路径支持以下 CARH 配置组合：

```json
{
  "block_size": 7,
  "markov_head_type": "carh",
  "carh_predecessor_count": 1,
  "carh_predecessor_context_mode": "none",
  "carh_sampled_prefix_memory_rank": 0
}
```

CARH 的逐步采样和 block replay 都以相同的实际前驱 token、当前位置
draft hidden、depth embedding 为条件，所以可直接使用 beta 的目标与
梯度公式。无需转换成 vanilla、重置 CARH 或移除卷积。多前驱 CARH、
predecessor context 和 sampled-prefix memory 尚未接入此采样路径，
会在配置 objective 时明确报错；原 `tv-acceptance` 仍只支持 vanilla
或无 Markov head。

对于 `Qwen3DSparkModel`、CARH rank 256、5 层 draft、以下卷积配置的
checkpoint，各部分处理如下：

| checkpoint 配置 | beta 后训练行为 |
| --- | --- |
| `architectures=["Qwen3DSparkModel"]`、`model_type="qwen3"`、`auto_map.AutoModel="dspark.DSparkDraftModel"` | 使用仓库的 legacy DSpark 类加载，保留原配置和权重；不依赖 checkpoint 内的远程 Python 实现 |
| `conv_mode="legacy"`、`conv_kernel_size=2`、`conv_group_size=16`、`conv_kernel_conditioning="low-rank64"`、`conv_source_rank=32` | 已有 backbone 支持，卷积权重参与 beta 训练 |
| `conv_apply_to="attention-mlp"`、`conv_last_n_layers=0` | 全部 draft 层的 attention 和 MLP 卷积保留 |
| `enable_confidence_head=true`、`confidence_head_with_markov=true` | 权重保留并冻结；beta 不训练 confidence loss，也不重新校准其预测 |
| selector / refiner / recall correction / block summary 的 rank 为 0，prefix mixer 为 `none` | 不构造这些可选模块，兼容；仅 `parallel_refiner_runtime_enabled=true` 不会启用 rank 为 0 的 refiner |
| predictive auxiliary 两个 alpha 均为 0，`elastic_horizon_enabled=false` | 与独立 beta objective 兼容 |

必须提供与原草稿匹配的 **target checkpoint、tokenizer 和训练数据**。
该草稿记录了 `hidden_size=4096`、`vocab_size=151936`、
`num_target_layers=36`、capture layers `[1,9,17,25,33]`，不能仅凭示例
脚本名称选择 target。beta 会对采样后分叉的前缀执行 target 验证，
只有原序列的离线 hidden states 不足以完成此训练。

下面是服务器上先跑 5 步的命令模板；替换路径，并按实际 GPU 数调整
`--nproc_per_node`。小 batch、anchors 和词表分块用于先检查加载与更新，
不代表正式训练的最优超参数或显存保证：

```bash
torchrun --standalone --nproc_per_node 8 scripts/train_dspark.py \
    --target-model-path /path/to/original-target \
    --init-draft-model-path /path/to/carh-checkpoint \
    --train-data-path /path/to/train.jsonl \
    --output-dir /path/to/new-carh-tv-beta-smoke \
    --dspark-loss-type tv-acceptance-beta \
    --target-model-backend sglang \
    --attention-backend flex_attention \
    --sglang-mem-fraction-static 0.3 \
    --chat-template qwen \
    --block-size 7 --num-anchors 8 \
    --batch-size 1 --max-length 512 \
    --tp-size 1 --accumulation-steps 1 --micro-batch-size 0 \
    --tv-temperature 1.0 \
    --tv-objective-chunk-blocks 1 \
    --tv-verification-batch-size 4 \
    --learning-rate 1e-5 --warmup-ratio 0 \
    --max-steps 5 --log-interval 1 --save-interval 5
```

`--init-draft-model-path` 自动使用 checkpoint 的 `config.json`，读取完整
draft 权重并新建 optimizer/scheduler；切换训练目标不要使用 `--resume`。
不必额外指定 `--markov-head-type` 或卷积覆盖参数。加载器会检查缺失、
多余和形状不匹配的权重。仅凭 config 可以判断结构兼容性，无法确认远端
权重是否完整、来自相同实现，或实际 GPU/SGLang 环境是否可运行。

CARH 回归测试覆盖非零 head 的采样/replay 概率、逐步参考梯度，以及
缩小 hidden/vocab 后保留 CARH + 卷积结构的 checkpoint 加载和更新：

```bash
python -m pytest -q \
    tests/test_utils/test_tv_acceptance_beta_carh.py \
    tests/test_utils/test_tv_beta_carh_checkpoint.py
```

### Beta 目标与梯度

对固定 anchor 上采到的 proposal `d_1, ..., d_K`，所有分布均以相同的
实际 proposal 前缀为条件：

```text
a_i = sum_v min(p_i(v), q_i(v)) = 1 - TV(p_i, q_i)
r_i = min(1, p_i(d_i) / q_old,i(d_i))
R_hat = a_1 + r_1*a_2 + r_1*r_2*a_3 + ...
L_block = 1 - R_hat / 7
```

当前位置使用全词表重叠率；前面的位置使用实际采样 token 的接受概率。
这样保留了“某个 token 被接受”和“它构成的后续前缀容易继续接受”的
关联。单条路径的 `R_hat` 是估计量；对 proposal 采样取期望，才得到
固定起点、给定有效预测范围下的期望连续接受长度。

反向传播使用：

```text
r_tilde_i(theta) = min(p_i(d_i), q_theta,i(d_i)) / stopgrad(q_old,i(d_i))
```

`q_old` 是本次采样保存的 token log probability，对应同一版本模型，
不是历史 checkpoint；`q_theta` 由同一份 backbone hidden 和相同
前驱 token、温度 replay 得到。分母停止梯度，分子和整个前缀连乘
保留梯度，不对训练比值再做 `clamp(max=1)`。实际计算采用
`exp(min(log_p(d), log_q(d)) - old_log_q(d))`，避免先转概率再相除
导致的小概率下溢。

对路径取期望时，采样概率中的 `q_old` 与分母抵消，留下前缀的
`prod min(p(d), q_theta(d))`，因此同时包含接受质量和 proposal
采样分布变化的梯度。只对 `min(1,p/q_theta)` 反传或 detach
前缀权重，都不能得到相同梯度。

默认 `--tv-anchor-mode random` 时，anchors 仍从监督区域选取。padding、序列尾部或监督 mask 的首个
无效位置限制该 anchor 的预测范围；实际采样的 EOS 计入当前位，
其后不再计分。reference EOS 不替代 proposal EOS 判断。
有效长度不足 7 时，loss 分母仍为 7；空 block 不参与均值。
各 rank 按全局有效 block 数归一化，全部 rank 为空时明确报错。
该目标不含 bonus token，也不计算策略变化对下一轮起点分布的影响。

### 数据流与计算复用

原序列 target capture 提供上下文 hidden，draft backbone 对这些
anchors 运行一次。之后用同一份 draft hidden 采样并 replay：首位
Markov 前驱为 anchor token，后续前驱为实际采样 token。target 必须
在对应的 proposal 前缀上评分。

首位置直接复用原 target final hidden；只要前面的 proposal 与
reference 仍相同，后续对应位置也继续复用。若第 `j` 个 proposal
首次分叉，预测它的分布仍可复用，只有第 `j+1` 位起需要验证。
分叉后即使 token 再次相同，也不会恢复复用 reference 后缀。
整轮所需前缀全部命中时，不发送额外验证请求；验证只需消费到最后
一个有效 proposal 的前驱，无需再为最后一个 proposal 计算 bonus。
同一验证批次的变长后缀 hidden 合并回填，避免逐 anchor 发起索引拷贝。

全注意力 Qwen3 使用额外的后端优化：

- HF capture 直接调用 Transformer，省去无用的整段词表投影。
  原 batch 的 KV 保留到当前 draft 前向完成；每批候选复制所需
  prefix KV，通过 attention mask 和显式位置编号批量计算短续接。
- SGLang 共享完整 KV 页，候选独占新增页。不共享可被续写的部分页，
  页边界最多重算 `page_size-1` 个 token。每批验证后释放新增页，
  当前前向结束后释放原缓存；异常退出也清理。prefix 索引使用 int64。
- SGLang 缓存路径默认最多批量验证 64 个候选；HF 和完整候选 fallback
  默认 4 个。可用 `--tv-verification-batch-size` 调整。

其他架构使用完整候选验证，但仍复用相同前缀的 target hidden。
SGLang beta 路径要求非空、右侧 padding 的输入。
没有为全部 anchors 常驻完整词表 logits/probabilities；全词表 TV
投影仍按 `--tv-objective-chunk-blocks` 分块。

proposal、旧 token log probability 和验证后的 target hidden 在反向前
固定。checkpoint 反向只重算词表投影、概率和 loss，不重新采样，
不重新运行 target Transformer，也不依赖已经释放的 KV。

### 日志、恢复与验证

beta 日志使用 `tv_acceptance_length_estimate`，区别于旧模式的
`tv_acceptance_length_proxy`。其余 `tv_acceptance_loss`、
`tv_mean_overlap`、`tv_valid_blocks`、`tv_mean_valid_length` 保留。
`accuracy` 表示实际 proposal 前缀上 draft/target 的 argmax 一致率，
不是实测接受率；position loss 仍为各位置 TV。

默认 AdamW 路径还记录 `train/grad_norm`（最近一步裁剪前的全局范数）、
`train/grad_norm_mean` 和 `train/grad_clip_fraction`（自上次日志以来所有
optimizer steps 的平均范数与超过裁剪阈值的比例）。这些指标用于判断
loss 尺度与裁剪阈值是否合适，不改变 loss。全局梯度范数出现 NaN/Inf
时会终止更新，保留模型参数、Adam 状态和学习率调度进度，避免写入坏权重。

beta checkpoint 保存 `tv_prefix_mode=sampled-beta-v1`。切换 loss
类型、前缀语义、anchor 模式或温度不能直接恢复 optimizer；使用
`--init-draft-model-path` 和新的输出目录重新开始。分块和验证批量
可以调整，但 beta 的分块变化可能改变随机数消费顺序，不保证逐步
复现同一条采样轨迹。

```bash
python -m pytest -q \
    tests/test_utils/test_tv_acceptance.py \
    tests/test_utils/test_tv_acceptance_beta.py \
    tests/test_scripts/test_train_dspark.py \
    tests/test_modeling/test_target/test_proposal_verification.py \
    tests/test_modeling/test_target/test_sglang_proposal_cache.py
```

测试包括小词表路径穷举的期望值/梯度、EOS、有限差分、低精度和
log 空间小概率、采样/replay 一致性、完整候选参考值与梯度、真实
小型 DSpark 前后向，以及真实小型 Qwen3 的 HF KV 等价性。
SGLang 的 CPU 测试使用模拟分页 allocator 检查页隔离、int64 索引、
释放和异常清理，不替代真实 SGLang GPU 执行。

训练速度、显存峰值及实际解码收益仍需 GPU 实测。应使用相同模型、
batch、anchors 和序列长度，对比旧模式与 beta 的完整 step 吞吐，
并观察前缀命中率和分叉位置；当前不承诺接近旧模式速度。

## DSpark beta：真实推理起点与完整 proposal replay

DSpark 的 `tv-acceptance-beta` 现在可启用 `--tv-anchor-mode on-policy`。
默认仍为 `random`。新模式保留原 DSpark checkpoint 的 **7 个 proposal**
布局、全词表 softmax、vanilla Markov head、无 Markov head，或前文的
单前驱无状态 CARH 及兼容卷积。不能把原 DSpark checkpoint 换成
DFlash2 的 8 槽位配置；selector、refiner、有状态 CARH 等未支持的
组合仍会报错。target 必须是全注意力 Qwen3，backend 为 HF 或 SGLang，
draft 使用 `--attention-backend sdpa` 或 `flex_attention`，并要求全注意力、
零 attention/fusion/mixer dropout。DSpark on-policy 不支持 `eager` 或
`flash_attention_2`；当前只为上述两种 backend 提供增量 rollout
与 packed replay 一致的 mask 路径，其他 backend 会明确报错。

使用与原 DSpark checkpoint 匹配的 target、tokenizer 和 reference JSONL：

```bash
TARGET_MODEL_PATH=/path/to/original-target \
INIT_DRAFT_MODEL_PATH=/path/to/original-dspark-checkpoint \
TRAIN_DATA_PATH=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/new-dspark-tv-beta-onpolicy \
bash examples/run_qwen3_8b_dspark_tv_beta_on_policy.sh 8 flex_attention sglang
```

前三个环境变量必须显式提供。脚本从 checkpoint 读取完整模型配置，
新建 optimizer/scheduler；默认模板为 `qwen3-instruct`，适合对应的
Qwen3 non-thinking 数据。每卡 batch 4、8 卡时每次更新采集 32 条输入；
学习率为 `2e-4`、warmup 0.04、最大总长度 4096、每条最多 512 轮、
新 token 预算默认 256。通过 `TV_ROLLOUT_MAX_TOKENS=4096` 可扩大预算，
实际生成上限仍是 `min(预算, 4096 - prompt 长度)`，并受 EOS 和轮数上限限制。
新 token 预算包含首个 target seed、接受的 proposal 以及 correction / bonus。
`--num-anchors 512` 表示到第 512 轮即停止生成，**不做 reservoir sampling**，
也不保证生成到 token 预算耗尽。

先用较小 batch 和轮数验证服务器环境：

```bash
TARGET_MODEL_PATH=/path/to/original-target \
INIT_DRAFT_MODEL_PATH=/path/to/original-dspark-checkpoint \
TRAIN_DATA_PATH=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/new-dspark-tv-beta-smoke \
TV_ROLLOUT_MAX_TOKENS=64 \
bash examples/run_qwen3_8b_dspark_tv_beta_on_policy.sh 8 flex_attention sglang \
    --batch-size 1 --num-anchors 8 \
    --stop-after-steps 5 --log-interval 1 --save-interval 5
```

在线闭环为：从 reference 中提取首个监督 token 前的 prompt → frozen
target 采样首 token 作为 root → 当前 draft 生成真实 proposal →
target 验证、接受前缀并采样 correction / bonus → 保存各轮记录 →
同一个 draft 实例 replay / backward → 一次 optimizer step。
proposal 与 replay 使用同版本的全部 draft 参数，包括 Markov/CARH 和卷积；
没有外部 draft 服务副本，因此不需要额外权重传输和版本 ACK。
FSDP 使各 rank 同步执行轮次和更新，replay、draft cache 和 target session
均不跨 optimizer step；每批只训练一次。

SGLang session 在初始 capture 时保存 CPU token 前缀，之后只追加实际
提交的 proposal 和 seed / correction / bonus。每轮 verify 将各行
proposal、长度、active 和有效性检查结果合并回传，commit 将接受长度、
active、emit 标志和下一 token 合并回传，不再逐行拷贝完整 GPU 前缀。
这只合并 session 层的 metadata 传输，不代表 acceptance、KV 页管理和
跨 rank 检查中的同步也已消除。

DSpark / DFlash2 的增量 draft rollout 直接构造 FlexAttention `BlockMask`，
不再每轮调用 `create_block_mask()` 推导稠密 mask。KV tile 数不变时复用
block 索引，每轮刷新 token 有效位；所有 tile 都保留逐 token 检查，屏蔽
ragged KV cache 的中间 padding 空洞和 inactive 行。rollout 不生成反向
block 索引，replay 继续使用独立的训练 mask。此路径仍遍历完全无效的 KV
tile，实际端到端收益需 GPU 实测。

session 在 capture 时初始化 CPU / GPU 已提交长度（包含 pending anchor），
seed / commit 使用已有的接受长度和 emit 标志更新两侧状态。draft rollout
直接使用这些长度，以 CPU 长度差检查倒退并确定新增 KV 宽度，不再每轮对
attention mask 求长度、读取 `(delta < 0).any()` 或执行 `delta.max().item()`。
inactive 行长度保持不变；长度快照按轮替换，避免修改历史轮次状态。

SGLang backend 初始化时缓存 logits processor 引用；之后仅在请求的
`return_last_hidden_states / return_logits / shard_returns` 组合变化时更新
属性。同一 rollout 的后续轮次不再遍历 target 模块树或重复设置 flags。
若 setter 中途失败，flags 缓存失效，下一次调用会重新应用设置。

每轮保留原 proposal、实际前驱 `[root,d1,...]`、采样时的 `log q_old`、
真实验证前缀上的 teacher final hidden 和 anchor。首拒 token 以及未提交的
proposal suffix 仍参与 beta loss；backbone 输入只使用合法已提交上下文，
不把 correction 后的序列冒充原 proposal 前缀。target Transformer 的
verification 结果直接复用，不在 backward 重新采集。DSpark replay
从保存的 teacher hidden 计算全词表 target 分布，draft 与 target
共享 `--tv-temperature`，不添加 top-k/top-p 截断。

目标保持前文 beta 公式：只 detach `q_old` 分母，prefix survival 保留梯度，
EOS 本身计入有效位置，固定除以 7；各 rank 按整步有效 block 总数归一化。
因此仍要求 `--tp-size 1 --accumulation-steps 1 --micro-batch-size 0`。
无偏性限于给定 round 起点和预算下的单轮目标，不包含后续 anchor
访问分布变化的梯度。是否提高实际接收长度，需在相同部署采样设置下单独评估。

`tv_rollout_accepted_length` 记录实际每轮接受的 proposal 数，
`tv_acceptance_length_estimate` 记录 beta 估计量，
`tv_rollout_round_limit_count` 记录因轮数上限而提前结束的行数。
扩大到 4096 预算会增加逐轮 target 验证和 FSDP 等待；
`--tv-objective-chunk-blocks 8` 只限制 logits/loss 分块，
不会把全部 packed backbone 激活限制为 8 个 block。长轨迹显存和吞吐需 GPU 实测。

reference completion 不参与 rollout，但当前 loader 仍根据 reference
的 `loss_mask` 识别 prompt，并过滤掉监督 token 少于 `2 * block_size`
的样本；不能直接用没有监督边界的纯 prompt JSONL 替换输入。
随机 anchor 切换到 on-policy，或改变 token/轮数/总长度预算时，
应使用 `--init-draft-model-path` 和新输出目录，不能直接恢复旧 optimizer。
脚本本身是 warm start 入口，不应再追加 `--resume`。
恢复同一次 on-policy 实验时，直接用 `scripts/train_dspark.py` 的原始训练参数，
移除 `--init-draft-model-path` 并添加 `--resume`，保持原 `--output-dir`、
数据、模板、batch、训练调度及采集预算；如从 pilot 恢复，应移除或提高
`--stop-after-steps`。改过 batch 或采集预算的 smoke run 应作为独立实验，
正式运行使用新输出目录重新 warm start。
重复启动同一输出目录可能覆盖同名 checkpoint；长 rollout 实验可缩短
`--save-interval`，避免依赖正常结束时的最终保存。

## DFlash2 beta：真实推理起点与完整 proposal replay

原生 DFlash2 的 `tv-acceptance-beta` 可启用 `--tv-anchor-mode on-policy`。
默认仍是 `random`，随机 anchor 实验不变。原生 DFlash2 路径要求
原生 `DFlash2DraftModel`、8 个槽位（1 个 anchor + 7 个 proposal）、
全注意力 Qwen3 target，以及 HF 或 SGLang backend；DSpark 使用前述独立路径，
原 `tv-acceptance` 或 native CE / D-PACE loss 不支持此模式。

设计参考 [Draft-OPD 论文的 rollout / replay](https://arxiv.org/html/2605.29343v3#S4.SS2)
和[官方实现](https://github.com/Simplified-Reasoning/Draft-OPD)：使用真实
verification 起点，并保留被拒绝的 draft block。本实现继续使用上述
beta 接收长度目标；论文对 accepted / rejected tokens 分别使用的
KL 与位置衰减不加入这里的 loss，以免改变已验证有效的 beta 梯度。

使用已有 epoch-5 checkpoint 的服务器命令：

```bash
cd /ossfs/workspace/SpecForge
bash examples/run_qwen3_8b_dflash2_tv_beta_on_policy.sh 8 flex_attention sglang
```

示例保留 batch 4、学习率 `2e-4`、warmup 0.04、梯度裁剪 1、
最大长度 4096、512 轮上限、100 个数据处理进程、SGLang memory
fraction 0.3，以及 W&B offline。默认输出目录在原实验名后增加
`-onpolicy`；应使用新目录，重复启动可能覆盖同名 checkpoint。target、训练集、
checkpoint、输出目录可分别通过 `TARGET_MODEL_PATH`、
`TRAIN_DATA_PATH`、`INIT_DRAFT_MODEL_PATH`、`OUTPUT_DIR` 覆盖。
`NUM_EPOCHS` 默认 1，`TV_ROLLOUT_MAX_TOKENS` 默认 256，
`TV_TEMPERATURE` 默认 1.0；脚本末尾仍可追加 CLI 覆盖参数。

```bash
TV_ROLLOUT_MAX_TOKENS=64 OUTPUT_DIR=/path/to/onpolicy-smoke \
bash examples/run_qwen3_8b_dflash2_tv_beta_on_policy.sh 8 flex_attention sglang \
    --batch-size 1 --num-anchors 8 --max-steps 5 \
    --log-interval 1 --save-interval 5
```

数据中的 reference response 不作为 rollout continuation：每条输入保留
首个监督 token 之前的 prompt。frozen target 先从该 prompt 的下一词分布
采样首个 response token，作为原生 DFlash 的首轮 anchor，再运行当前
draft + frozen target 的真实随机 speculative decoding。下一轮 anchor 来自本轮
实际提交的 accepted prefix 和 target correction / bonus，不再从
reference response 随机抽位置。多轮对话样本使用首段 assistant
response 之前的上下文，不把后面的 reference 对话当成在线生成结果。
若输入从第 0 个 token 就开始监督（如无聊天 prompt 的连续文本），
则保留第 0 个 token 作为必需的非空 prefix，其余内容丢弃。

`--tv-rollout-max-tokens` 限制每个 prompt 新提交的 token 数，包含首个
target seed 和之后的 correction / bonus；`--num-anchors` 在此模式下限制每个 prompt
最多采集的 verification 轮数。`--max-length` 同时限制 prompt +
新 continuation 的总长度。EOS、token 预算、总长度或轮数上限到达时
停止采集，避免单个训练 step 随生成长度无界增长。默认 token 预算 256
时，每条 prompt 最多还有 255 轮；`num-anchors=512` 是上限，不保证收集
512 个 anchor。FSDP 各 rank 同步执行相同数量的 draft 轮次，已完成行
只占空占位，因此长尾 prompt 会影响整个训练 step 的用时。

proposal 使用 strict unary top-k selector 定义的真实分布 `q` 随机
采样；target 使用全词表分布 `p`，两者共享 `--tv-temperature > 0`。
接收概率为 `min(1, p(d)/q(d))`，首拒位置从归一化的
`(p-q)_+` 采样 correction，全接受时从 target 采样 bonus。
这里的温度同时用于 rollout、verification 和 loss replay；不是
greedy proposal 配上未经变换的 softmax TV，也不叠加额外 top-p 截断。

每一轮保留全部有效 proposal token、采样时的 `q_old` log probability
及对应 target 的 candidate log probability，包括首拒 token 和拒绝后的未提交后缀。
后缀继续为 beta 的单轮期望提供信号，但不加入实际 verified
continuation。EOS 与预算仍可限制有效范围。loss 保持上文 beta
公式，分母固定为 **7**，不按接受长度或首拒位置重归一化，也不改成
first-reject-only。bonus 影响下一轮起点，不计入 beta 的单轮 reward。

一次采集对应一次训练，rollout 与 replay 使用同版本参数、相同
proposal 和 selector support；不存在跨 optimizer step 的旧 top-k
replay buffer。target 始终冻结。其 verified prefix KV 和 draft 所需
auxiliary hidden 在轮次间复用；draft 也增量缓存已提交上下文的 KV，
每轮丢弃 noise block 的临时 KV。验证分支在回滚前保存 selector
candidate IDs 对应的 `log p`，保留 **target 全词表 softmax 的归一化常数**，
不会把 candidate 内的 target 概率重新归一化。replay 使用这些已保存的
teacher 概率，不以 correction 后的分布替换 rejected-prefix 分布，
也不重算 target 的词表投影；只重算 draft 及其 candidate logits/loss。
target KV 在反向前释放，不重新抽样或运行 target Transformer。
target prefill 之后，每轮只执行一次增量 verification：correction /
bonus 暂存为 pending anchor，与下一轮 proposal 一起处理。SGLang
共享完整 prefix KV 页；HF 虽然省去 prefix Transformer 重算，仍需
复制和整理 prefix KV，因此两种后端的实际性能并不相同。

无偏性仍限定于**给定 anchor 的单轮期望及其梯度**。真实 rollout
改善了训练起点的访问分布，但本实现不对参数变化如何改变后续
anchor 访问分布求梯度，因此不声称是整条生成轨迹吞吐的完整无偏
梯度；接收长度和 wall-clock 加速仍需独立推理评估。

checkpoint 保存并比较 `tv_anchor_mode`、温度；on-policy 还比较
`tv_rollout_max_tokens`、`num_anchors` 和 `max_length`。random 与
on-policy 不能互相 `--resume`，改变采集预算也不能直接恢复旧 optimizer。
切换时使用 `--init-draft-model-path` 和新输出目录。缺少新字段的旧
beta checkpoint 按 random 解释，仍可恢复原实验。词表 chunk 可在相同
语义下调整。on-policy 每轮验证当前活跃 prompt batch，
`--tv-verification-batch-size` 在此模式不生效；它仅控制 random-anchor
beta 的候选验证批次。

新增日志 `tv_rollout_accepted_length` 是采集过程实际每轮接受的
proposal 数（不含 seed/correction/bonus），与估计量
`tv_acceptance_length_estimate` 分开记录。`tv_rollout_rounds`、
`tv_rollout_generated_tokens`、`tv_rollout_round_limit_count` 分别记录
全局轮数、生成 token 数和因轮数上限仍未结束的行数。
全局没有有效 block 时执行零梯度 backward，但跳过 Adam、weight
decay 和学习率调度更新；数据步计数仍推进。

本机验证包含候选集与全词表 beta 的数值及梯度等价性、精确残差
采样、EOS/预算、真实 tiny Qwen3 的增量 KV 与整段计算等价性、
draft 增量采样与 packed replay 概率一致、两 rank CPU FSDP，
以及实际 trainer 保存和恢复。SGLang 使用 CPU 分页 allocator
测试；Flex Attention mask 通过 CPU 等价计算检查，不替代 CUDA
kernel、8 卡显存和吞吐测试。

## 原 tv-acceptance 模式

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
- 原 `tv-acceptance` 支持 vanilla Markov head（或关闭 Markov），`T > 0`，
  全词表 softmax；CARH 仅支持上文的 beta 配置，selector/refiner 不在此范围。
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

## Native DFlash2 的 TV-beta 训练

同一入口 `scripts/train_dspark.py` 现在按 draft config 的 `architectures`
识别 `DFlash2DraftModel`，使用独立的 DFlash2 beta 包装层。DSpark 仍使用
原来的模型、采样、目标配置检查和 beta 计算路径；不需要修改已有 DSpark 命令。

```bash
TARGET_MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_DATA_PATH=/path/to/qwen3-8b-regenerated.jsonl \
OUTPUT_DIR=/path/to/qwen3-8b-dflash2-tv-beta \
bash examples/run_qwen3_8b_dflash2_tv_acceptance_beta.sh 8 flex_attention sglang \
    --report-to wandb --wandb-offline
```

也可以在原训练命令中把 `--draft-config-path` 改成
`configs/qwen3-8b-dflash2.json`，把 `--block-size` 改成 `8`，并使用新的
输出目录。**删除原 DSpark 的 `--init-draft-model-path`，或者换成兼容的
Qwen3-8B DFlash2 checkpoint。** 此功能不执行 DSpark 到 DFlash2 的权重迁移；
两者的标签位置约定不同，不能只改 architecture 名称转换模型。

本实现固定优化七个 proposal：DFlash2 的八个原生槽为一个 anchor 加七个
预测槽；第零槽不进入 beta loss。DSpark 的七个槽则全部是预测槽。
从已有 DFlash2 checkpoint 初始化时，checkpoint 的实际 block size 必须为 8，
配置与 target 的 hidden size、词表、捕获层及 token IDs 必须匹配。

DFlash2 的 proposal 分布由 strict unary top-k 候选和条件 selector 分数定义：
先应用模型的 unary logit transform，再选 top-k；对当前实际采样前驱计算
selector 分数，除以 `--tv-temperature` 后在候选集上 softmax。候选集外的
proposal 概率为零。target 仍按完整词表归一化，不能在候选集上重归一化。
训练温度应与计划使用的随机推理温度一致；此目标不等同于 greedy 解码目标。

每次前向保存候选 IDs、proposal tokens、实际前驱和 detached `q_old`。
反向回放同一候选集和前驱，经 selector 与 unary logits 构造全词表稀疏分布，
复用原 beta 公式和 target suffix verification。Top-k 成员选择是离散的，
梯度在当前候选集内传播；不对成员替换求导。卷积、backbone、selector 都参与
优化，target head 和 embeddings 冻结；不叠加 selector CE 或 DSpark 辅助目标。
EOS、部分尾块、监督 mask 中断和全局有效 block 归一化沿用 beta 语义。

导出保留 `DFlash2DraftModel` 架构及原生参数名，不写入 DSpark 模型文件。
`--init-draft-model-path` 严格加载完整 DFlash2 权重，重新建立 optimizer；
`--resume` 同时恢复训练状态，并检查候选集大小、卷积配置、block 约定等
proposal 语义。改变这些设置时应使用新的输出目录和 weight-only 初始化。
当前支持范围为 standalone 入口、`T > 0`、`tp-size=1`、
`accumulation-steps=1`、`micro-batch-size=0`，以及原有 FSDP 数据并行训练流程。
统一 YAML 入口的 `training.loss_type` 尚不包含这个 beta 目标。
