# 让 `--dspark-loss-type dpace` 适配 DFlash checkpoint（官方语义 D-PACE）

## 背景与目标

当前 `--dspark-loss-type dpace` 被硬绑定到 DFlash2DraftModel（`setup.py:179-182` 直接报错），无法用于现有的 `DFlashDraftModel` checkpoint（如 `qwen3-8b-dflash-mix`，block_size=7，DeepSpec 槽位布局）。现有替代 `--pace-mode token` 虽然权重形状与官方 D-PACE 一致，但归一化语义不同（逐样本质量重标定 + 按权重质量归一，而非官方的按序列数归一）。

目标：让 `--dspark-loss-type dpace --dpace-alpha 0.5` 在 DFlash/DSpark checkpoint 上可用，且语义与官方 release（`/personal/repo/D-PACE`）完全一致：

- 权重：`q = exp(-CE)`（draft 在 target token 的概率）→ `smooth=(1-α)q+α` → 块内 cumprod → 后缀和 → detach，无效位置在乘积中置 1、在后缀和中剔除
- 归一化：`loss = Σ(CE·w) / 全局序列数`（**不做**逐样本质量重标定）
- 纯 CE 目标：无 L1 / confidence head / BV / 其他辅助项 → **硬报错要求显式置零**（用户已确认，遵循 dflash2.py 的"拒绝静默改变实验"惯例）

## 明确不改（与官方的剩余差异均属 drafter 接口/数据约定，非 D-PACE 内容）

- 槽位布局：DSpark slot j → anchor+j+1 全监督（官方 slot 0 排除是 DFlash 布局产物）
- eval_mask 的 cumprod 连续前缀截断（dspark.py:6002）
- anchor 采样（两仓库 `_sample_anchor_positions` 已逐行一致）
- DFlash2 native 路径（`dflash_family_model.py`）完全不动

## 改动清单

### 1. `specforge/core/dspark.py`（核心）

**a) 抽模块级权重函数**（便于单测对照官方参考实现）：

```python
def dpace_suffix_credit(quality, eval_mask, alpha):
    """与官方 _dpace_weight 逐行等价的 fp32 实现：smooth→cumprod→后缀和→乘 mask。无质量重标定。"""
```

`_pace_weights` 改为内部调用它 + 可选的逐样本质量重标定（加 `mass_match: bool = True` 参数，`False` 时跳过 5659-5669 的 `mass_scale` 逻辑，直接返回 raw weights；diagnostics 相应调整）。

**b) `OnlineDSparkModel.__init__`** 新增参数 `official_dpace: bool = False`、`official_dpace_alpha: float = 0.5`：
- 校验 alpha ∈ [0,1]；`self.dpace_alpha = float(official_dpace_alpha)`（兼容 objective.py:173 的打印）
- 互斥校验（报清晰错误）：`pace_mode != "none"`；`l1_loss_alpha > 0`；`bv_loss_alpha > 0`；`confidence_head_alpha > 0`；`offline_acceptance_objective != "none"`；`vat_enabled`；非默认 pace 调度参数（warmup/ramp/decay_start/blend_max<1，官方无调度，全程全替换）

**c) forward 的 PACE 段**（~6437 处加分支）：`self.official_dpace` 时 `quality = exp(-ce_per_token.detach().float())`，调 `dpace_suffix_credit(quality, eval_mask, self.dpace_alpha)`，`ce_weight_mask = 结果`（无 blend 调度，固定全替换）。复用现有 pace_diagnostics 上报质量/存活均值。

**d) loss 聚合**（~8267 与 ~8342 处）：official 模式下 CE 项分母换为全局序列数：

```python
ce_den = base_logits.new_tensor(float(bsz))
if dp_size > 1: dist.all_reduce(ce_den, SUM, group=dp_group)   # = dp·bsz
# loss 中：hard_loss_alpha * ce_num / ce_den  （其余项仍用 global_den，此模式下均已校验为 0）
```

原理：FSDP 梯度跨 rank 取均值、外层 `* dp_size` 抵消后，有效全局目标 = `Σ_全局(CE·w)/(dp·bsz)`，与官方 `Σ(CE·w)/bsz` 按 rank 平均后的有效目标恰好相等。

### 2. `specforge/legacy/dspark_training/setup.py`

179-182 报错收窄：`dflash2` 仍要求 DFlash2；`dpace` 放开（落入下方 DSparkDraftModel 构建分支）：

```python
if args.dspark_loss_type == "dflash2":   # 原：in DFLASH2_NATIVE_OBJECTIVES
    raise ValueError("--dspark-loss-type dflash2 requires a DFlash2DraftModel config")
```

### 3. `specforge/legacy/dspark_training/trainer.py`

`is_dflash2 == False` 分支构建 `OnlineDSparkModel`（~570 行）时传：
`official_dpace=(args.dspark_loss_type == "dpace")`, `official_dpace_alpha=args.dpace_alpha`。

### 4. `specforge/legacy/dspark_training/objective.py`

- `validate_objective_args`：保留现有 `dpace` × `--pace-mode` 互斥；新增 dpace 与 `l1_loss_alpha>0` / `bv_loss_alpha>0` / `confidence_head_alpha>0` / `offline_acceptance_objective!="none"` / 非默认 pace 调度的冲突硬报错，错误信息写明需显式置零的参数（如 `--l1-loss-alpha 0.0`）。
- `configure_objective`：dpace 打印按模型类型区分（用 `is_dflash2_config` 判断，仿照 TV 分支 181 行），DSpark 模型打印 "official D-PACE semantics: per-sequence normalization, no mass matching"。
- `--dspark-loss-type` help 文本更新：dpace 亦可用于 DFlash/DSpark checkpoint（官方语义 D-PACE）。
- `validate_objective_resume`：dpace 模式下把 `dpace_alpha` 加入 resume invariants（153 行列表）。

### 5. 测试

- `tests/test_scripts/test_train_dspark.py`：
  - dpace + 非 DFlash2 不再报 "requires a DFlash2DraftModel"（检查并更新现有断言）
  - dpace + `l1_loss_alpha>0` / `confidence_head_alpha>0` / `--pace-mode token` → ValueError
  - resume：改动 dpace_alpha 报错
- `tests/test_utils/test_dflash_losses.py`：新增 `dpace_suffix_credit` 数值测试——复用文件里已有的 `_naive_dpace_weight` 参考（301 行）逐元素对照；并验证 official 模式 loss = `Σ(CE·w)/bsz`（小规模构造）。

### 6. 训练脚本（`/personal/scripts/specforge/qwen3-8b-dflash-dpace.sh`）

```bash
# 删除: --pace-mode token --pace-alpha 0.5 --pace-apply-to ce-l1 --pace-blend-max 1.0
# 新增:
--dspark-loss-type dpace --dpace-alpha 0.5 --confidence-head-alpha 0.0
# 保留: --l1-loss-alpha 0.0 --ce-loss-alpha 1.0（其余不变）
```

## 验证

1. `pytest tests/test_utils/test_dflash_losses.py tests/test_scripts/test_train_dspark.py`
2. 数值对照：同一组 prob/mask 下，新函数输出与官方 `_dpace_weight`（/personal/repo/D-PACE/specforge/core/dflash.py:234）逐元素一致
3. 用修改后的脚本小规模跑数个 step：确认 loss 正常下降、wandb 中 pace 诊断量（quality/survival 均值）合理；注意归一化变化后 loss 量级与 grad_norm 会与旧脚本不同，LR 2e-4 可能需要重调

## 非目标

- 不实现官方的两个消融变体（`dpace-cumulative-confidence-only` / `dpace-continuation-value-only`）；如后续需要，`dpace_suffix_credit` 加一个分支即可
- 不改 DSpark 默认 `ce-l1` 路径的任何行为（`mass_match=True` 保持现状）
