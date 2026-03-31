# WOW / AFSIM P6DOF 控制适配报告

> 日期：2026-03-24
> 范围：P6DOF 横向/纵向/能量控制链路分析、实验验证、Profile 落地

---

## 1. 背景与目标

WOW 的 3D 模型输出 token，每个 token 解码为 `[dpsi_rad, dalt_m, dspd_mps]` 的 2 秒前瞻增量。
这些增量需要映射为 AFSIM P6DOF 飞机的自动驾驶指令。

**核心问题：** 原有 `TurnToRelativeHeading` + `bank_angle_max=60°` 的映射方式，
实际转弯率仅 ~0.45°/s，远低于 token T29 所需的 ~10°/s。

**目标：** 找到最合适的 P6DOF 控制映射，使仿真飞机的响应尽可能匹配 token 的语义意图。

---

## 2. P6DOF 信号链分析

### 2.1 三级串级控制结构

通过代码审计确认，P6DOF 横向控制链路如下：

```
TurnToRelativeHeading(turn_deg, g_cmd)
  │
  │  WsfP6DOF_Mover.cpp:2074
  │  将 relative heading 转为 absolute heading target
  │
  ▼
【外环】pid_roll_heading
  │  P6DofCommonController.cpp:2889-3062
  │  输入：heading_error (deg, wrapped ±180°)
  │  输出：commanded_turn_rate (deg/s)
  │  限幅：±maxTurnRate_dps（由 bank_angle_max 和速度计算得出）
  │
  │  turn_rate → bank_angle (via 转弯半径方程)
  │  bank_angle clamp to ±bank_angle_max  ← 硬饱和
  │
  ▼
【中环】pid_bank_angle
  │  P6DofCommonController.cpp:3079-3116
  │  输入：bank_error (deg)
  │  输出：roll_rate_cmd (deg/s)
  │  限幅：±rollRate_Max
  │
  ▼
【内环】pid_roll_rate
  │  P6DofCommonController.cpp:3118-3141
  │  输入：roll_rate_error
  │  输出：stickRight [-1.0, +1.0]
  │
  ▼
副翼执行
  Aileron ±20° 偏转, 执行器 ±80°/s 速率限制
```

### 2.2 关键发现

| 项目 | 结论 | 证据位置 |
|------|------|---------|
| pid_roll_heading 输入 | heading_error (deg, wrapped ±180°) | CommonController.cpp:2915 |
| pid_roll_heading 输出 | **转弯率 (deg/s)**，非 bank 命令 | CommonController.cpp:3017-3019 |
| bank_angle_max 性质 | **硬饱和**，在两处生效 | lines 3014-3015 和 3042-3049 |
| g_cmd 参数 | **在 heading mode 下被忽略**（代码注释 "ignored for now"） | PilotObject.cpp:3402-3420 |

### 2.3 bank_angle_max 的双重限制

`bank_angle_max` 不仅 clamp 最终 bank 命令，还**先限制了 PID 输出范围**：

```
274 m/s, bank_angle_max = 60° 时:
  maxTurnRate_dps = 360 / (2π × 274² / (9.81 × tan(60°))) = 3.55°/s
```

即使 Kp 再大，PID 输出也不会超过 3.55°/s。bank_angle_max 是 PID 输出限幅的上游来源。

### 2.4 可用的低层接口

| 脚本方法 | 作用 | 绕过层级 |
|---------|------|---------|
| `SetAutopilotRollAngle(deg)` | 直接设目标横滚角 | 绕过外环 |
| `SetAutopilotRollRate(rad/s)` | 直接设横滚角速率 | 绕过外环+中环 |
| `SetBankAngleMax(deg)` | **运行时修改 bank 上限** | 无需改共享配置 |
| `SetPitchGLoad(nz)` | 设纵向 G-load | 纵向通道 |
| `SetAutopilotVerticalSpeed(fpm)` | 设目标垂速 | 纵向通道 |
| `GoToSpeed(mps)` / `GoToAltitude(m)` | 标准速度/高度保持 | 平台级 |

**横向与纵向通道互相独立**，设置 RollAngle 不会清空纵向模式（代码确认 `P6DofAutopilotAction` 各通道解耦）。

---

## 3. 横向控制实验

### 3.1 三种横向适配模式

| 模式 | ID | 命令方式 | 描述 |
|------|----|---------|------|
| heading_fraction | 0 | `TurnToRelativeHeading(fractional_dpsi)` | 原有模式，每 tick 发送 dpsi × (dt/H_action) |
| heading_lookahead | 1 | `TurnToRelativeHeading(full_dpsi)` | 每 tick 发送完整 dpsi |
| **roll_from_token** | **2** | `SetAutopilotRollAngle(phi_cmd)` | 从 token 直接计算 bank 角度 |

roll_from_token 的计算公式：

```
psi_dot = dpsi_rad / H_action_sec        # rad/s (token 语义转弯率)
phi_cmd = atan(V × psi_dot / g)          # 协调转弯所需 bank 角
phi_cmd = clamp(phi_cmd, ±bank_angle_max)
→ SetAutopilotRollAngle(phi_cmd)
```

### 3.2 实验结果

统一条件：T29 token，V ≈ 274 m/s (533 kt)，高度 9144 m (30 kft)。

| 实验 | 模式 | bank 上限 | 实测转弯率 | vs T29 目标 (10.3°/s) |
|------|------|----------|-----------|---------------------|
| A1 | heading_fraction | 60° | **0.45°/s** | 4% |
| B1 | heading_lookahead | 60° | **3.5°/s** | 34% |
| B2 | heading_lookahead | 80° | **4.4°/s** | 43% |
| **C1** | **roll_from_token** | **80°** | **10.9°/s** | **106%** |

**结论：roll_from_token 是唯一能匹配 token 语义的横向模式。**

### 3.3 heading_hold 慢的根因

1. **heading_fraction 模式**：每 tick heading_error 仅 ~2°，Kp=0.22 → 输出仅 ~0.44°/s 转弯率
2. **heading_lookahead 模式**：error 更大但 PID 输出被 maxTurnRate_dps 限幅（60° bank → 3.55°/s 上限）
3. **提升 bank 到 80°**：maxTurnRate 提升到 ~13°/s，但 PID 响应仍然慢，实测仅 4.4°/s
4. **roll_from_token**：完全绕过 heading-hold 外环，直接设 bank 角 → 中环+内环快速跟踪

---

## 4. 纵向/能量控制实验

### 4.1 四种纵向适配模式

| 模式 | ID | 命令方式 | 描述 |
|------|----|---------|------|
| alt_hold | 0 | `GoToAltitude(target)` | 标准高度保持 |
| **gload_feedforward** | **1** | `SetPitchGLoad(1/cos(phi))` | 协调转弯所需 G-load |
| **vert_speed_from_token** | **2** | `SetAutopilotVerticalSpeed(vz_cmd)` | 保留 token dalt 语义 |
| gload_ff_vz | 3 | `SetPitchGLoad(nz_base + Kz×alt_err)` | 组合模式 |

### 4.2 实验结果

统一条件：roll_from_token + bank80°，T29 token。

| 实验 | 纵向模式 | nz 命令 | 实际 G | 30s 掉高 | 30s 掉速 |
|------|---------|--------|-------|---------|---------|
| L1 | alt_hold | 1.0G | 0.8G | ~2400m | ~0 kt |
| **L2** | **gload_ff** | **4.5G** | **2.9G** | **1764m** | **-75 kt** |
| **L3** | **vert_speed** | **1.0G** | **3.0G** | **1764m** | **-74 kt** |
| L4 | gload+vz | 5.0G | 3.0G | 1818m | -75 kt |

### 4.3 关键发现

1. **L2 vs L1**：gload_ff 显著减少掉高（2400m → 1764m），但飞机在 AoA ~10° 时饱和，实际 G 只有 2.9G
2. **L2 vs L3**：两者掉高几乎一致（1764m），说明纵向控制的改善天花板已到
3. **物理不可行性**：T29 的 phi_cmd ≈ 78° 需要 nz = 1/cos(78°) = 4.8G 才能水平转弯，但 AoA 限制实际 nz 只有 ~3G
4. **infeasible_level_turn 阈值**：phi > arccos(1/3.0) = 70.5° 时，水平转弯不可行，掉高不可避免

---

## 5. 落地方案

### 5.1 Profile 体系

创建了 `wow/sim/p6dof_profiles.py`，定义两个 profile：

#### p6dof_semantic（默认）

```
横向: roll_from_token       → SetAutopilotRollAngle(phi_cmd)
纵向: vert_speed_from_token → SetAutopilotVerticalSpeed(vz_cmd)
能量: GoToSpeed             → platform.GoToSpeed(spd_target)
bank: 80°                   → SetBankAngleMax(80) 运行时设置
```

**选择理由：** 保留 token 的 dalt 语义。token 编码的是 2s 前瞻高度变化，映射为垂速命令 (`vz = dalt / H_action_sec`) 最忠实于原始意图。

#### p6dof_aggressive_turn（实验）

```
横向: roll_from_token       → SetAutopilotRollAngle(phi_cmd)
纵向: gload_feedforward     → SetPitchGLoad(1/cos(phi))
能量: GoToSpeed             → platform.GoToSpeed(spd_target)
bank: 80°                   → SetBankAngleMax(80) 运行时设置
```

**选择理由：** 大转弯时优先保高度，但牺牲 token 的 dalt 语义。适用于需要最小掉高的场景。

### 5.2 遥测标记

在 AFSIM draw_processor 中新增两个饱和标记：

| 标记 | 条件 | 含义 |
|------|------|------|
| `aoa_saturation_flag` | \|alpha\| > 9.5° | AoA 接近极限，无法产生更多 G |
| `infeasible_level_turn_flag` | \|phi_cmd\| > arccos(1/3.0) ≈ 70.5° | 水平转弯物理不可行，掉高不可避免 |

当 `infeasible_level_turn_flag = 1` 且 token 的 dalt 非明显下降时，表示横纵语义互斥——飞机会优先执行横向（bank 角度），接受高度损失。

### 5.3 风险隔离

| 措施 | 说明 |
|------|------|
| 不改共享 FA-LGT 配置 | bank_angle_max 通过 `SetBankAngleMax()` 运行时设置 |
| WOW 专用场景 | `2v2_p6dof_lateral_test.txt` 独立于其他 demo |
| Profile 参数可覆盖 | `--profile p6dof_semantic --bank_angle_max 70` |

### 5.4 使用方式

```bash
# 默认 profile
python -m sim.token_sweep_server \
  --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
  --vocab datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json \
  --single_token 29 \
  --profile p6dof_semantic

# 激进转弯 profile
python -m sim.token_sweep_server \
  ... --profile p6dof_aggressive_turn

# 覆盖个别参数
python -m sim.token_sweep_server \
  ... --profile p6dof_semantic --bank_angle_max 70
```

---

## 6. 结论

### 6.1 已确定

| 项目 | 结论 |
|------|------|
| 默认横向模式 | `roll_from_token` — 直接从 token 计算 bank 角，绕过 heading-hold 外环 |
| 默认纵向模式 | `vert_speed_from_token` — 保留 token 的 dalt 语义 |
| 默认能量模式 | `GoToSpeed` — 标准速度保持 |
| 默认 bank 上限 | 80° — 运行时设置 |
| g_cmd 参数 | 在 P6DOF heading mode 下不生效（代码确认） |

### 6.2 已知限制

| 限制 | 说明 |
|------|------|
| AoA 饱和 | FA-LGT 在 ~10° AoA 饱和，限制实际 G ≈ 3G |
| 水平转弯阈值 | phi > 70.5° 时水平转弯不可行 |
| T29 类大转弯 | phi ≈ 78°，30s 掉高 ~1764m，掉速 ~75kt |
| 以上为飞机模型可达性限制 | 不是实现 bug |

### 6.3 待补齐

| 项目 | 状态 |
|------|------|
| 多 token 组合测试 | 尚未进行（直飞→大转→小转切换等） |
| 2v2 对抗场景验证 | 需在完整 2v2 场景中确认控制稳定性 |
| KINEMATIC_MOVER 支持 | 已探索并放弃（控制映射成本过高） |
| 模型训练对齐 | 需将控制架构同步给训练侧 |

---

## 7. 文件清单

| 文件 | 用途 |
|------|------|
| `wow/sim/p6dof_profiles.py` | Profile 定义 |
| `wow/sim/token_sweep_server.py` | Token 执行服务器（含 `--profile` 支持） |
| `wow/docs/P6DOF_CONTROL_ARCHITECTURE.md` | 控制架构文档 |
| `wow/docs/P6DOF_ADAPTATION_REPORT.md` | 本报告 |
| `build/demos/air_to_air/scenarios/2v2_p6dof_lateral_test.txt` | WOW 专用测试场景 |
| `build/demos/air_to_air/output/lateral_telemetry.csv` | 遥测数据输出 |
