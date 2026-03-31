# 当前接口汇总

---

## A. OneStepTokenBC

### 输入张量

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `obs_hist` | `[B, H=4, 8]` | float32 | 4 个 token 边界处的 ego-centric 观测 |
| `prev_token` | `[B]` | int64 | 上一步预测的 dense token ID (首步 = vocab_size = 30 作为 BOS) |

### 预处理流程

1. **Ego 变换** (`ego_transform_current_anchor`): 以最后一个时间步为锚点
2. **Z-score 归一化**: `(obs - obs_mean) / obs_std`, 统计量来自训练集全集

### 输出

| 字段 | 形状 | 说明 |
|------|------|------|
| `logits` | `[B, 30]` | 30 个 active token 的 unnormalized logits |

### 推理循环

```
prev_token = 30 (BOS)
每个 token 边界 (每 2 秒):
  1. 收集 obs_hist [4, 8] (4 个 token 边界, 即 6 秒历史)
  2. ego_transform_current_anchor(obs_hist)
  3. normalize
  4. dense_token = argmax(model(obs, prev_token))
  5. raw_token = dense_to_raw[dense_token]
  6. z_q = vqvae.vq.embedding[raw_token]
  7. chunk = vqvae.decoder(z_q) -> denormalize -> [4, 3]
  8. 执行 chunk[0] (first_row)
  9. prev_token = dense_token
```

**来源**: `sim/token_policy_runtime.py`, `training/vq/ego_obs_utils.py`

---

## B. Ego-Centric 观测

### 8 维观测向量

| 索引 | 字段 | 原始含义 | 变换后含义 |
|------|------|---------|-----------|
| 0 | x_e_m | 东向位置 | dx' = 相对锚点, 旋转后 |
| 1 | y_n_m | 北向位置 | dy' = 相对锚点, 旋转后 |
| 2 | z_u_m | 高度 (m) | **不变** (绝对高度) |
| 3 | vx_e_mps | 东向速度 | vx' = 旋转后 |
| 4 | vy_n_mps | 北向速度 | vy' = 旋转后 |
| 5 | vz_u_mps | 垂直速度 | **不变** |
| 6 | heading_rad | 航迹角 (unwrapped) | heading' = 相对锚点 (锚点处 = 0) |
| 7 | speed_mps | 地面速度 | **不变** |

### Ego 变换公式 (锚点 = 最后时间步)

```
x0, y0, h0 = obs[-1, 0], obs[-1, 1], obs[-1, 6]
cos_h, sin_h = cos(-h0), sin(-h0)

dx = x - x0,  dy = y - y0
x' =  cos_h * dx + sin_h * dy
y' = -sin_h * dx + cos_h * dy

vx' =  cos_h * vx + sin_h * vy
vy' = -sin_h * vx + cos_h * vy

heading' = heading - h0
z, vz, speed: 不变
```

**来源**: `training/vq/ego_obs_utils.py` 第 24-73 行

---

## C. Token / Chunk

### 基本参数

| 参数 | 值 | 来源 |
|------|-----|------|
| token_steps | 4 | `best.pt` metadata |
| 每 token 覆盖时间 | 2 秒 (4 x 0.5s) | `dt=0.5s` |
| tokens_per_window | 5 | 20 / 4 |
| chunk 形状 | [4, 3] | [token_steps, action_dim] |
| active vocab | 30 | 64 个码字中 30 个被使用 |

### 每步动作维度

| 索引 | 字段 | 单位 | 语义 |
|------|------|------|------|
| 0 | dpsi_rad | rad | 2 秒前瞻航向增量 |
| 1 | dalt_sp_m | m | 2 秒前瞻高度增量 |
| 2 | dspd_sp_mps | m/s | 2 秒前瞻速度增量 |

### Chunk 4 行的含义

chunk 的 4 行是**同一段轨迹**上滑动 2s-lookahead 窗口, **不是** 4 个独立连续动作:

```
chunk[0] = state(t+4) - state(t)     <- 当前决策点 (t=0)
chunk[1] = state(t+5) - state(t+1)   <- t=0.5s
chunk[2] = state(t+6) - state(t+2)   <- t=1.0s
chunk[3] = state(t+7) - state(t+3)   <- t=1.5s
```

### 解码后如何进入执行层

```
chunk[4,3] --first_row--> action[3] = chunk[0]
  |
  +-- dpsi_rad --> psi_dot = dpsi / 2.0 --> turn_deg = deg(psi_dot * dt_cmd)
  |                                          --> SetAutopilotRollAngle(phi)
  |                                              phi = atan(V * psi_dot / g)
  |
  +-- dalt_m   --> alt_target = current_alt + dalt (clamp 200~20000 m)
  |                --> latch --> SetAutopilotVerticalSpeed(vz_cmd) 或 GoToAltitude
  |
  +-- dspd_mps --> spd_target = current_spd + dspd (clamp 120~650 m/s)
  |                --> latch --> GoToSpeed(target)
  |
  +-- [runtime computed] G-load = sqrt(1 + (V * psi_dot / g)^2) (clamp 1~7)
                                  --> 640-frame action[base+3] = g * 9.81
```

**来源**: `sim/token_bridge_server.py` 第 206-642 行

---

## D. P6DOF Profiles

### 两个可用 profile

| 参数 | p6dof_semantic (默认) | p6dof_aggressive_turn (实验) |
|------|----------------------|---------------------------|
| lateral_mode | 2 (roll_from_token) | 2 (roll_from_token) |
| vertical_mode | 2 (vert_speed_from_token) | 1 (gload_feedforward) |
| bank_angle_max | 80 deg | 80 deg |
| 用途 | 保留 token 的 dalt 语义 | 高 bank 角时保持高度 |

### 差异说明

- **p6dof_semantic**: 纵向用 `SetAutopilotVerticalSpeed(vz_cmd)`, vz_cmd 从 dalt 推算。保留了 token 训练时的 2s-lookahead 高度增量语义。大 bank 角时**允许高度损失**。
- **p6dof_aggressive_turn**: 纵向用 `SetPitchGLoad(1/cos(phi))`, 前馈补偿 bank 角引起的升力损失。大 bank 角时**试图保持高度**, 但会增加速度损失。

### Runtime 可覆盖参数

| 参数 | CLI flag | 默认值 | 说明 |
|------|----------|--------|------|
| profile | `--profile` | p6dof_semantic | 整体 profile 选择 |
| lateral_mode | `--lateral_mode` | 2 | 覆盖 profile 的 lateral |
| vertical_mode | `--vertical_mode` | 2 | 覆盖 profile 的 vertical |
| bank_angle_max | `--bank_angle_max` | 80 | 度, 覆盖 profile |
| exec_mode | `--exec_mode` | first_row | chunk 压缩策略 |
| g_mode | `--g_mode` | dynamic | fixed / dynamic |
| g_fixed | `--g_fixed` | 3.0 | 固定 G 值 (g_mode=fixed 时) |
| g_max | `--g_max` | 7.0 | 动态 G 上限 |
| h_action_sec | `--h_action_sec` | 2.0 | horizon-fraction 分母 |
| latch | `--latch` | enabled | 逐通道 latch |
| heading_update_thresh | `--heading_update_thresh` | 3.0 deg | heading latch 阈值 |
| alt_update_thresh | `--alt_update_thresh` | 50.0 m | altitude latch 阈值 |
| speed_update_thresh | `--speed_update_thresh` | 10.0 m/s | speed latch 阈值 |
| max_abs_dpsi | `--max_abs_dpsi` | 0.35 rad | guardrail |
| max_abs_dalt | `--max_abs_dalt` | 400.0 m | guardrail |
| max_abs_dspd | `--max_abs_dspd` | 40.0 m/s | guardrail |
| conf_threshold | `--conf_threshold` | 0.35 | 模型置信度阈值 |

**来源**: `sim/p6dof_profiles.py`, `sim/token_bridge_server.py` 第 805-844 行

---

## E. 风险标记

### aoa_saturation_flag

- **含义**: 迎角饱和, 飞机达到可用升力极限
- **触发条件**: `|alpha_deg| > 9.5 deg`
- **影响**: 限制可达 G 为约 3G (FA-LGT 平台)
- **来源**: `docs/P6DOF_CONTROL_ARCHITECTURE.md`
- **当前代码状态**: 文档定义, bridge 内**未直接检测** alpha (AFSIM 不在状态帧中回传 AoA)

### infeasible_level_turn_flag

- **含义**: 当前 bank 角下无法维持等高度转弯
- **触发条件**: `|phi_cmd| > arccos(1/nz_max_est)`, 其中 `nz_max_est ~ 3.0G` -> 阈值约 70.5 deg
- **影响**: bank > 70.5 deg 的转弯**必然产生高度损失** (物理不可避免)
- **来源**: `docs/P6DOF_CONTROL_ARCHITECTURE.md` 第 6 节
- **当前代码状态**: 文档分析, 实验确认 (T29 token 产生 phi=78 deg, 实测 60 m/s 沉降率)

### 补充: speed_bleed

- **含义**: 持续高 bank 转弯导致的速度衰减
- **典型值**: 60-90 kt / 30 秒 (bank=80 deg, T29 token)
- **来源**: `docs/P6DOF_ADAPTATION_REPORT.md` 实验数据
- **当前代码状态**: 无自动检测, 仅遥测记录

### 补充: guardrail 触发

- **检测层级**: chunk 级 (token 边界) + tick 级 (每帧)
- **触发条件**: NaN/Inf, 置信度 < 0.35, |dpsi| > 0.35 rad, |dalt| > 400 m, |dspd| > 40 m/s, chunk 内步间跳跃过大
- **Fallback**: HOLD_LAST (复用上一个有效动作) 或 NEUTRAL (发零增量)
- **来源**: `sim/token_guardrails.py`
