# Baseline 冻结声明

> 日期: 2026-03-24
> 版本: Day 1 Semantic Freeze

---

## 1. 一句话定义

当前低层执行 baseline 固定为: **VQ tokenizer (64-codebook, token_steps=4) -> token dataset (30 active codes) -> OneStepTokenBC (MLP, H=4) -> VQ decode -> P6DOF profile execution (roll_from_token + vert_speed)**。Day 1 不修改 tokenization、OneStep 模型结构和 P6DOF 主契约。

---

## 2. 当前闭环图

```
                         训练侧 (已冻结)
  ┌─────────────────────────────────────────────────────┐
  │  ACMI -> build_vq_clean_dataset.py                  │
  │       -> dt2hz_H2s_vqclean.npz [500748, 20, 8/3]   │
  │                                                      │
  │  train_vqvae.py -> vqvae_clean_t4_cb64/best.pt      │
  │       codebook=64, token_steps=4, latent=32          │
  │                                                      │
  │  tokenize_npz.py -> *_tok.npz (30 active tokens)    │
  │                                                      │
  │  train_onestep_token_bc.py                           │
  │       -> onestep_token_bc_t4_cb64_h4/best.pt        │
  │       MLP, obs_hist=4, hidden=128, vocab=30          │
  └─────────────────────────────────────────────────────┘

                         部署侧 (已冻结)
  ┌─────────────────────────────────────────────────────┐
  │  token_policy_runtime.py                             │
  │    obs_hist[4,8] -> ego_transform -> normalize       │
  │    -> OneStepTokenBC -> dense_token                  │
  │    -> dense_to_raw -> VQ decode -> chunk[4,3]        │
  │    -> denormalize -> action [dpsi, dalt, dspd]       │
  └──────────────────────┬──────────────────────────────┘
                         │
                         v
  ┌─────────────────────────────────────────────────────┐
  │  token_bridge_server.py                              │
  │    exec_mode=first_row -> chunk[0] only              │
  │    heading: dpsi -> horizon-fraction -> turn_deg     │
  │    altitude: current + dalt -> abs setpoint (latch)  │
  │    speed: current + dspd -> abs setpoint (latch)     │
  │    G-load: dynamic n=sqrt(1+(V*psi_dot/g)^2)        │
  │                                                      │
  │  P6DOF profile: p6dof_semantic (default)             │
  │    lateral=roll_from_token (SetAutopilotRollAngle)   │
  │    vertical=vert_speed (SetAutopilotVerticalSpeed)   │
  │    bank_max=80 deg                                   │
  │                                                      │
  │  -> 640-element action frame -> AFSIM TCP            │
  └─────────────────────────────────────────────────────┘
```

---

## 3. 冻结边界

### 3.1 今天之后不改的部分

| 组件 | 冻结范围 | 关键检查点/文件 |
|------|---------|----------------|
| VQ-VAE | 模型结构, 码本, 权重 | `checkpoints/vqvae_clean_t4_cb64/best.pt` |
| Token vocab | 30 个 active codes 的 dense<->raw 映射 | `datasets/*_tok.vocab.json` |
| OneStepTokenBC | 模型结构, 权重, 推理逻辑 | `checkpoints/onestep_token_bc_t4_cb64_h4/best.pt` |
| Ego 变换 | current-anchor 旋转逻辑 | `training/vq/ego_obs_utils.py` |
| Action 语义 | 3D [dpsi_rad, dalt_m, dspd_mps], 2s lookahead | `ml/dataset_default_config.py` |
| P6DOF 契约 | profile 定义, lateral/vertical mode 编号, 4D output format | `sim/p6dof_profiles.py` |
| Bridge 核心 | chunk reduction, horizon-fraction, dynamic-G, latch | `sim/token_bridge_server.py` |
| Guardrails | 阈值和 fallback 策略 | `sim/token_guardrails.py` |

### 3.2 高层语义层允许新增的部分

| 允许项 | 说明 | 约束 |
|--------|------|------|
| 语义标签系统 | 新增 semantic_state 分类层 | 不修改 token 本身 |
| Token family 映射表 | 对 30 个 token 做粗分类 | 只读映射, 不改码本 |
| Profile hint 机制 | 语义层建议 profile | bridge 侧可忽略 |
| Constraints 传递 | 语义层下发约束 (如 prefer_energy_preserve) | 执行层解释约束, 不修改已有 action |
| 态势摘要输入 | 新增高层输入 schema | 基于已有 8D obs 提取, 可扩展 |
| 弱监督标注工具 | 离线标注工具 | 不改训练管线 |

---

## 4. 文档 vs 代码一致性审计

### 4.1 用户假设 vs 代码实际

| 项目 | 用户假设 (任务书) | 代码实际 | 状态 |
|------|-------------------|---------|------|
| token_steps | 2 (每 token 1 秒) | **4** (每 token 2 秒) | **不一致** |
| codebook_size | 128 | **64** | **不一致** |
| active vocab | 未指定 | 30 (64 中) | -- |
| tokens_per_window | 10 (20/2) | **5** (20/4) | 由 token_steps 导出 |
| chunk 形状 | [2, 3] | **[4, 3]** | 由 token_steps 导出 |

> **说明**: 用户任务书中的 `token_steps=2, codebook=128` 可能引用了早期版本或 TRAINING_PIPELINE.md 中的参考值。当前实际部署的冻结模型使用 `token_steps=4, codebook=64`。本文档以**代码实际**为准。

### 4.2 TRAINING_PIPELINE.md vs 代码实际

| 项目 | 文档描述 | 代码实际 | 状态 |
|------|---------|---------|------|
| token_steps | 4 | checkpoint 实际 = 4 | **已一致** |
| codebook_size | 64 | checkpoint 实际 = 64 | **已一致** |
| 命令参考 | `--token_steps 4 --codebook_size 64` | 实际 checkpoint 用 `t4_cb64` | **已一致** |

> TRAINING_PIPELINE.md 已统一为 `t4_cb64` 口径, 与实际冻结 checkpoint 一致。

### 4.3 其他文档 vs 代码

| 文档 | 一致性 | 备注 |
|------|--------|------|
| MIGRATION_HANDOFF.md | **完全一致** | token_steps=4, cb=64, exec_mode=first_row, dynamic-G |
| P6DOF_CONTROL_ARCHITECTURE.md | **完全一致** | 2 profiles, roll_from_token, vert_speed |
| control_interface_summary.md | **完全一致** | 3D action + runtime 4th dim G-load |
| timebase_audit.md | **完全一致** | Bug 修复已落地, 80 项回归通过 |

---

## 5. 冻结检查点摘要

```
checkpoints/vqvae_clean_t4_cb64/best.pt
  token_steps = 4
  codebook_size = 64
  latent_dim = 32
  normalize_action = True
  act_mean = [-0.00138, 4.543, 0.262]     (dpsi_rad, dalt_m, dspd_mps)
  act_std  = [0.120, 122.613, 13.998]

checkpoints/onestep_token_bc_t4_cb64_h4/best.pt
  vocab_size = 30
  obs_hist_len = 4
  hidden_dim = 128
  n_layers = 2
  dropout = 0.1
  测试集 Top-1 = 78.45%, 机动子集 Top-1 = 59.96%

datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json
  active_vocab_size = 30
  codebook_size_original = 64
  token 19 (raw 45) = 39.7% (最高频, 稳态巡航)
  token 8  (raw 25) = 21.0% (次高频, 轻微爬升)
  Top-5 tokens 占 76.1%
```
