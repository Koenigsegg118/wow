# AFSIM 控制接口与模型动作对齐总结

> 目标：让模型训练侧理解「模型输出的 3D 动作」如何经过 runtime 变换后下发到 AFSIM，以及仿真侧的实际响应能力和限制。

---

## 1. 模型输出 → AFSIM 命令 全链路

```
模型推理 (Token Policy / BC Policy)
  │
  │  输出 3D action（2 秒前瞻增量）:
  │    [dpsi_rad, dalt_m, dspd_mps]
  │
  ▼
Python Runtime (token_bridge_server / token_sweep_server)
  │
  │  转换为每 tick 命令（dt ≈ 0.2s）:
  │    turn_deg  = degrees(dpsi_rad / H_action × dt)
  │    alt_cmd   = clamp(current_alt + dalt_m, 200, 20000)    ← token 开始时锁定
  │    spd_cmd   = clamp(current_spd + dspd_mps, 120, 650)   ← token 开始时锁定
  │    g_cmd     = dynamic_g(speed, |dpsi_rad/H_action|)      ← 或 fixed 3G
  │
  ▼
TCP Binary (STATUS + 640×float32)
  │
  │  每平台 8 个 float:
  │    action[base+0] = turn_deg      (°, 相对航向增量)
  │    action[base+1] = alt_cmd       (m, 绝对高度设定点)
  │    action[base+2] = spd_cmd       (m/s, 绝对速度设定点)
  │    action[base+3] = g_cmd         (m/s², G 载荷命令)
  │
  ▼
AFSIM draw_processor (WSF_SCRIPT_PROCESSOR, 20Hz)
  │
  │  当 action_seq 变化时调用:
  │    platform.TurnToRelativeHeading(turn_deg, g_cmd)
  │    platform.GoToAltitude(alt_cmd, 50.0)       ← 50 m/s 爬升率限制
  │    platform.GoToSpeed(spd_cmd)
  │
  ▼
WSF_P6DOF_MOVER 自动驾驶仪执行
```

---

## 2. 动作语义详解

### 2.1 模型输出的 3D 动作

| 通道 | 符号 | 单位 | 语义 | 来源 |
|------|------|------|------|------|
| 0 | dpsi_rad | rad | 2 秒前瞻航向变化量 | VQ-VAE 解码 chunk[0][0] |
| 1 | dalt_m | m | 2 秒前瞻高度变化量 | VQ-VAE 解码 chunk[0][1] |
| 2 | dspd_mps | m/s | 2 秒前瞻速度变化量 | VQ-VAE 解码 chunk[0][2] |

**chunk 结构**：shape = [4, 3]，4 个时间步 × 3 通道。`first_row` 模式只使用 chunk[0]。

**H_action_sec = 2.0s**：每个动作描述的是「未来 2 秒内的累积变化量」。

### 2.2 Runtime 转换公式

**航向**（每 tick 逐步施加）：
```
psi_dot = dpsi_rad / H_action_sec          # 角速率 (rad/s)
turn_deg = degrees(psi_dot × dt)            # 每 tick 转角 (°)
```

**高度/速度**（token 开始时一次性锁定）：
```
alt_cmd = clamp(current_alt + dalt_m, 200, 20000)   # 绝对高度 (m)
spd_cmd = clamp(current_spd + dspd_mps, 120, 650)   # 绝对速度 (m/s)
```

**G 载荷**（dynamic 模式）：
```
n_req = sqrt(1 + (V × ψ̇ / g)²)
n_cmd = clamp(n_req, 1.0, g_max)            # g_max = 7.0
g_cmd = n_cmd × 9.81                        # 转为 m/s²
```

### 2.3 AFSIM 接收的 4D 命令

| 通道 | 名称 | 单位 | 范围 | 说明 |
|------|------|------|------|------|
| a0 | turn_deg | ° | 不限 | 相对航向增量，每 tick 施加一次 |
| a1 | alt_cmd | m | [200, 20000] | 绝对高度设定点 |
| a2 | spd_cmd | m/s | [120, 650] | 绝对速度设定点 |
| a3 | g_cmd | m/s² | [9.81, 68.67] | G 载荷命令（1-7G） |

---

## 3. Token 动作量级参考（30 个 token 的 first_row 统计）

| 统计量 | dpsi (°) | dalt (m) | dspd (m/s) |
|--------|----------|----------|------------|
| 最大右转 | +43.5° (T24) | - | - |
| 最大左转 | -42.7° (T28) | - | - |
| 最大爬升 | - | +202 (T26) | - |
| 最大俯冲 | - | -526 (T27) | - |
| 最大加速 | - | - | +92 (T1) |
| 最大减速 | - | - | -98 (T23) |
| 近平飞 | -0.04° (T19) | +4 (T19) | +1 (T19) |

### 对应的命令转弯率（token → runtime）

| Token | dpsi_rad | 期望转弯率 (°/s) | 需要的 G @ 274m/s | 实际可达 |
|-------|----------|-----------------|------------------|---------|
| T24 (大右转) | +0.759 | 21.7 | 10.6G | **受限** |
| T28 (大左转) | -0.745 | 21.3 | 10.5G | **受限** |
| T2 (中右转) | +0.443 | 12.7 | 6.3G | 勉强可达 |
| T14 (中左转) | -0.287 | 8.2 | 4.1G | 可达 |
| T29 (小左转) | -0.360 | 10.3 | 5.1G | 可达 |
| T19 (平飞) | -0.001 | 0.02 | 1.0G | 可达 |

---

## 4. P6DOF 自动驾驶仪实际响应能力

### 4.1 FA-LGT 自动驾驶仪关键限制

**配置文件**：`p6dof_types/aircraft/fa-lgt/controls/autopilot_config.txt`

| 参数 | 值 | 影响 |
|------|------|------|
| **bank_angle_max** | **60°** | **核心瓶颈**：限制最大侧向 G 为 1/cos(60°) = 2G |
| roll_rate_max | 180°/s | 横滚速率充裕，不是瓶颈 |
| pitch_gload_max | 8G | 纵向 G 充裕 |
| yaw_gload_max | 0.5G | 偏航 G（正常，靠横滚转弯） |
| **pid_roll_heading Kp** | **0.219** | **航向跟踪增益极低**，响应迟缓 |
| pid_roll_heading Ki | 0.000225 | 积分项极小，几乎无修正作用 |
| control_method | BANK_TO_TURN_WITH_YAW | 靠横滚实现转弯 |

### 4.2 转弯能力计算

**bank_angle_max = 60° 决定的理论上限**：

```
最大侧向加速度 = g × tan(60°) = 9.81 × 1.73 = 17.0 m/s²
在 274 m/s 时:
  最大转弯率 = 17.0 / 274 = 0.062 rad/s = 3.55°/s
  最小转弯半径 = 274² / 17.0 = 4414 m
```

**pid_roll_heading Kp = 0.219 决定的实际表现**：

```
PID 输出 = Kp × heading_error_deg
若航向误差 = 4.35°（单 tick 命令）:
  命令转弯率 = 0.219 × 4.35 = 0.95°/s   ← 与实测 0.9°/s 吻合
```

### 4.3 实测对比（T24, 274 m/s）

| | 期望值 | 理论上限 | 实际表现 | 利用率 |
|---|--------|---------|---------|--------|
| 转弯率 | 21.7°/s | 3.55°/s (bank_angle_max) | **0.9°/s** | **25% of 上限** |
| 所需 G | 10.6G | 2.0G (bank 60°) | ~0.45G | - |
| 响应延迟 | 即时 | ~1-2s 建立横滚 | >5s 才渐入稳态 | - |

**结论**：模型期望的大机动 token（±40°/2s）在当前 P6DOF 配置下根本无法执行，实际执行能力约为期望的 4%。

---

## 5. 时间基准

| 参数 | 值 | 说明 |
|------|------|------|
| dt_data（训练数据采样） | 0.5s | ACMI 数据 2Hz |
| H_action_sec（前瞻时间窗） | 2.0s | 每个 token 描述 2s 变化量 |
| token_steps（chunk 长度） | 4 | chunk = [4, 3]，4 步 × 0.5s = 2s |
| dt_cmd（runtime 帧率） | ~0.2s | SAC_PROCESSOR TCP 帧间隔 |
| dt_sim（仿真步长） | 0.05s | draw_processor 更新间隔 |
| dt_p6dof（P6DOF 内部步长） | 0.01s | 100Hz 物理积分 |

---

## 6. 状态观测（AFSIM → Python）

**StatesObserver.txt** 每 0.1s 采样，SAC_PROCESSOR 以 ~0.2s 间隔通过 TCP 推送。

每平台 8 维状态向量：

| 索引 | 名称 | 单位 | 说明 |
|------|------|------|------|
| 0 | live | 0/1 | 平台是否存活 |
| 1 | lat | ° | 纬度 |
| 2 | lon | ° | 经度 |
| 3 | alt | m | 高度 MSL |
| 4 | velN | m/s | 北向速度（NED） |
| 5 | velE | m/s | 东向速度（NED） |
| 6 | velD | m/s | 下向速度（NED） |
| 7 | heading | ° | 航向 |

**平台索引映射**（2v2_model_only 场景）：
```
idx 0 → red_1    （受控）
idx 1 → red_2    （受控）
idx 2 → blue_1   （目标）
idx 3 → blue_2   （目标）
idx 4-79 → 未使用
```

---

## 7. 已知问题与对齐建议

### 7.1 转弯能力缺口

| 问题 | 详情 |
|------|------|
| bank_angle_max = 60° | 硬性限制最大转弯率为 3.55°/s @ 274m/s |
| pid_roll_heading Kp = 0.219 | 实际转弯率仅 0.9°/s，只用到 25% 的理论能力 |
| 大机动 token 无法执行 | T24/T28 等 ±40° token 需要 10G+，远超 2G 上限 |

**建议**：训练侧应知晓仿真侧的转弯率上限（当前配置下约 0.9°/s ~ 3.55°/s），避免训练出超过物理可执行范围的动作。或者调整自动驾驶仪参数提升转弯能力。

### 7.2 高度/速度响应

- `GoToAltitude` 的 climbDiveRate 硬编码为 50 m/s
- `GoToSpeed` 无加速度限制参数，靠 P6DOF 内部推力/阻力模型
- 高度和速度的 delta 在 token 开始时一次性锁定为绝对设定点，后续 tick 不更新

### 7.3 通道间耦合

- 转弯消耗升力 → 导致高度下降（T24 测试中 9144m → 200m）
- 大俯冲增加速度 → 与减速命令冲突
- 自动驾驶仪内部优先级：航向 > 高度 > 速度（从 PID 配置推断）
