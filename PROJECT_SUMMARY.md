# WoW 项目总结

> 更新日期：2026-03-03

---

## 一、项目概述

本项目在 AFSIM（Warlock）仿真平台上构建战斗机智能控制系统，包含两条并行技术路线：

| 路线 | 说明 | 入口 |
|---|---|---|
| **LLM 实时控制** | 双 LLM Agent（Planner + Executor）通过 LangGraph 生成实时飞行指令 | `llm-llm_with_connection.py` |
| **BC 行为克隆** | 从 Tacview ACMI 录像提取战斗机飞行数据，训练 Transformer 策略模型 | `training/` |

两路线共用 AFSIM TCP 协议接口，均可接入闭环仿真。

---

## 二、系统架构

### 2.1 LLM 实时控制系统

```
AFSIM (C++) ──TCP──► StateReceiver.recv_frame()
                          │
                          ├──► send_status_data(last_action)   [立即响应 ~0ms]
                          │
                          └──► state_queue ──► llm_worker 线程
                                                    │
                                              translate_sim_data_to_llm_context()
                                                    │
                                              LangGraph app.invoke()
                                                    │
                                    ┌─── Planner Node (Qwen 4B) ───┐
                                    │     输出战术计划（JSON）        │
                                    └─── Executor Node (Gemma 1B) ─┘
                                          输出控制增量（JSON）
                                                    │
                                          apply_llm_decision_to_sim()
                                                    │
                                              last_action 更新
```

**双线程设计**：主线程快速响应 AFSIM（~100ms），LLM 推理线程异步更新指令（~5s），通过 lock + put_nowait 模式保证线程安全。

**TCP 协议**：
- 接收：`simTime {t} float0 ... float639`（80 平台 × 8 值：live/lat/lon/alt/v_north/v_east/v_down/heading）
- 发送：`"STATUS"(6B)` + `640 × float32 little-endian`（仅平台 0–3 有效）

### 2.2 BC 训练管道

```
tra_data/*.acmi
    │
    ├── analyze_acmi_coverage.py   → 质量分析报告（reports/coverage/）
    │
    └── build_training_dataset.py  → 训练张量 NPZ（datasets/）
                                        │
                                   train_bc_transformer_smooth.py
                                        │
                                   bc_transformer_fighteronly.pt
                                        │
                                   bc_policy_socket_server.py  → AFSIM 接入
```

---

## 三、数据管道

### 3.1 原始数据

| 项目 | 值 |
|---|---|
| 来源目录 | `wow/tra_data/` |
| 文件数量 | 82 个（`.acmi` + `.zip.acmi` 混合） |
| 内容 | DCS World 真实对抗场次录像，含多个联队 |

### 3.2 坐标与约定

| 约定 | 值 | 备注 |
|---|---|---|
| 坐标系 | ENU | x=East(m), y=North(m), z=Up(m) |
| 航向定义 | `atan2(vx_east, vy_north)` | 0=North，顺时针为正 |
| dpsi_sign | **+1** | 右转 → 正 dpsi，与 AFSIM `TurnToRelativeHeading` 验证一致 |
| 单位 | 全公制 | m / m/s / rad；发给 AFSIM 时才 rad→deg |
| 决策频率 | 2 Hz（dt=0.5s） | |
| 预测时域 | H=2s（k=4步） | dpsi = wrap(heading[t+k] − heading[t]) |

### 3.3 实体过滤

#### Regex 名称/类型过滤（`training/entity_filter.py`）

默认排除以下非战斗机实体：

```
(?i)(A-50|E-3|E-2|E-767|AWACS|Sentry|Hawkeye|
     KC-135|KC-10|KC-46|IL-78|Il-78|Tanker|
     C-130|C-17|C-5|An-26|An-12|Il-76|
     Tu-95|B-52|B-1|B-2|Bomber|Transport|JSTAR)
```

#### Orbiter 行为过滤

飞行轨迹同时满足以下条件时判定为"盘旋/稳定巡航"并排除：

| 参数 | 阈值 | 含义 |
|---|---|---|
| `n_valid_steps` | ≥ 600（= 300s@2Hz） | 足够长才判定 |
| `std(altitude)` | < 80 m | 高度几乎不变 |
| `std(speed)` | < 7 m/s | 速度几乎不变 |
| `p95(│Δψ│)` | < 0.05 rad | 几乎不转弯 |

#### 过滤效果

| 指标 | 过滤前 | 过滤后 | 变化 |
|---|---|---|---|
| 总 Windows | 707,613 | 504,774 | −29% |
| \|Δψ\| p95 | 0.237 rad | 0.289 rad | +22% |
| Entropy(\|Δψ\|) | 0.372 nats | 0.466 nats | +25% |
| 被排除实体 | — | 164（regex） | 每文件约 2 个预警/支援机 |

### 3.4 质量检测

| 检测项 | 阈值 |
|---|---|
| 加速度突变 | `accel_mag > 150 m/s²` |
| 速度跳变 | `speed_jump > 200 m/s` |
| 速度天花板 | `MAX_SPEED = 800 m/s`（≈ Mach 2.4，超出 = 轨迹损坏） |
| 最低速度 | `MIN_SPEED = 60 m/s`（过滤滑行/悬停） |

---

## 四、训练集

### 4.1 超参数（固化在 `training/dataset_default_config.py`）

```python
DT        = 0.5      # 决策步长 (s)
H_SEC     = 2.0      # 预测时域 (s)
K         = 4        # 时域步数
SEQ_LEN   = 20       # 每窗口序列长度
STRIDE    = 5        # 窗口滑动步长
MIN_SPEED = 60.0     # m/s
MAX_SPEED = 800.0    # m/s
DPSI_SIGN = +1.0     # 与 AFSIM 验证一致
TYPE_FILTER = "Air+FixedWing"
```

### 4.2 数据模式

```
obs    [N, T=20, 8]   float32
         ├─ [0] x_e_m               ENU East (m)
         ├─ [1] y_n_m               ENU North (m)
         ├─ [2] z_u_m               高度 (m)
         ├─ [3] vx_e_mps            东向速度 (m/s)
         ├─ [4] vy_n_mps            北向速度 (m/s)
         ├─ [5] vz_u_mps            垂直速度 (m/s)
         ├─ [6] heading_rad_unwrapped  展开航向 (rad)
         └─ [7] ground_speed_mps    地速 (m/s)

action [N, T=20, 3]   float32
         ├─ [0] dpsi_rad    H=2s 内航向变化量 (rad)，CW=正
         ├─ [1] alt_sp_m    高度设定点 (m)
         └─ [2] spd_sp_mps  速度设定点 (m/s)
```

### 4.3 统计快照

| 指标 | p5 | p50 | p95 | p99 | entropy |
|---|---|---|---|---|---|
| Altitude (m) | 532 | 6606 | 12865 | 14847 | 2.56 nats |
| Speed (m/s) | 177 | 359 | 600 | 713 | 2.51 nats |
| \|Δψ\| (rad) | 0.00006 | 0.00486 | 0.2886 | 0.5414 | 0.47 nats |

### 4.4 训练集文件

```
datasets/dt2hz_H2s_fighteronly.npz          101 MB — 训练张量
datasets/dt2hz_H2s_fighteronly.meta.json    完整元数据（可读）
datasets/dt2hz_H2s_fighteronly.filelist.txt 参与合成的 82 个 ACMI 路径
datasets/dt2hz_H2s_fighteronly.rejected.txt 排除记录
datasets/dt2hz_H2s_fighteronly.stats.json   分布快照
```

**规模：** 504,774 windows × (20 步 × 8 obs + 20 步 × 3 act) = **10,095,480 样本**

---

## 五、模型训练

### 5.1 架构：TransformerBC

```
输入: obs [B, T=20, 8]
  └─ Linear(8 → 128)
  └─ TransformerEncoder × 4  (nhead=4, FFN=512, GELU, norm_first=True)
  └─ LayerNorm → Linear(128 → 3)
输出: action [B, T=20, 3]
```

非因果（full-window context）；推理时取最后一步 `pred[:, -1, :]` 作为当前动作。

### 5.2 损失函数

```
L = w_dpsi · MSE(dpsi)
  + w_alt   · Huber(alt,  δ=1)
  + w_spd   · Huber(spd,  δ=1)
  + λ_smooth · mean((Δpred)²)    # 平滑正则，抑制抖动

权重: w_dpsi=1.0  w_alt=0.01  w_spd=0.1  λ_smooth=0.5
```

### 5.3 训练曲线

| 阶段 | Epoch | lr | Train Loss | Val Loss |
|---|---|---|---|---|
| 初训 | 1 | 3e-4 | 0.7962 | 0.5668 |
| 初训 | 10 | 3e-4 | 0.2818 | 0.2109 |
| 初训 | 20 | 3e-4 | 0.2026 | 0.1599 |
| 初训 | 30 | 3e-4 | 0.1726 | 0.1309 |
| 续训 | 31 | 1e-4 | 0.1503 | 0.1155 |
| 续训 | 35 | 1e-4 | 0.1413 | 0.1082 |
| **续训** | **39** | **1e-4** | **0.1378** | **0.1052 ← best** |
| 续训 | 40 | 1e-4 | 0.1365 | 0.1097 |

### 5.4 模型文件

```
datasets/bc_transformer_fighteronly.pt   3.3 MB
  包含: model weights, obs_mean/std, act_mean/std, meta, args
```

---

## 六、使用命令（均从 `wow/` 目录运行）

### BC 训练管道

```bash
# 数据覆盖分析
python training/analyze_acmi_coverage.py
python training/analyze_acmi_coverage.py --write_recommended_list 1

# 合成训练集
python training/build_training_dataset.py
python training/build_training_dataset.py --out datasets/myset.npz
python training/build_training_dataset.py --exclude_orbiters 0

# 训练模型（初训）
python training/train_bc_transformer_smooth.py \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --save datasets/bc_transformer_fighteronly.pt \
    --epochs 30 --lr 3e-4

# 续训（降 lr）
python training/train_bc_transformer_smooth.py \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --save datasets/bc_transformer_fighteronly.pt \
    --epochs 10 --lr 1e-4

# 单元测试
python training/tests/test_coverage_utils.py
# 38 tests in ~0.3s — OK
```

### LLM 实时控制

```bash
conda activate wow
python llm-llm_with_connection.py   # 启动 Python socket 服务端，等待 AFSIM 连接
```

---

## 七、完整文件清单

```
wow/
├── PROJECT_SUMMARY.md               本文件
│
├── ── LLM 实时控制 ────────────────────────────────────────────────
├── llm-llm_with_connection.py       主入口（LangGraph 双 LLM 控制器）
├── llm_with_connection/
│   ├── config.py                    LLM 服务器地址、端口、模型名
│   ├── graph.py                     LangGraph 节点（Planner + Executor）
│   ├── realtime_server.py           TCP 双线程实时服务
│   ├── socket_protocol.py           StateReceiver / send_status_data
│   ├── sim_translation.py           640-float → 中文 LLM 上下文
│   ├── action_mapping.py            LLM JSON → 640-float 指令数组
│   └── clients.py                   OpenAI 兼容客户端
├── afsim_policy_bridge.py           BC 策略 AFSIM 接口层
├── bc_policy_socket_server.py       BC 模型 TCP 服务端（独立接入 AFSIM）
├── LLM-Rules.py                     旧版规则控制器（留档）
├── LLM-LLM.py                       旧版双 LLM 控制器（留档）
│
├── ── BC 训练管道 ─────────────────────────────────────────────────
├── training/
│   ├── README.md                    训练管道使用说明
│   ├── dataset_default_config.py    唯一超参数配置源
│   ├── entity_filter.py             Regex + Orbiter 过滤
│   ├── heading_convention_config.py 航向约定（atan2 顺序）
│   ├── acmi_to_dt_dataset_smooth.py ACMI 解析函数库（共享）
│   ├── build_training_dataset.py    一键合成入口
│   ├── analyze_acmi_coverage.py     覆盖分析与报告
│   ├── train_bc_transformer_smooth.py Transformer BC 训练脚本
│   └── tests/
│       └── test_coverage_utils.py   38 个单元测试
│
├── ── 数据 & 产物 ─────────────────────────────────────────────────
├── tra_data/                        原始 ACMI 文件（82 个）
├── datasets/
│   ├── dt2hz_H2s_fighteronly.npz    训练集（101 MB）
│   ├── dt2hz_H2s_fighteronly.meta.json
│   ├── dt2hz_H2s_fighteronly.stats.json
│   ├── dt2hz_H2s_fighteronly.filelist.txt
│   ├── dt2hz_H2s_fighteronly.rejected.txt
│   └── bc_transformer_fighteronly.pt  已训练模型（3.3 MB）
└── reports/coverage/
    ├── stats.json / stats_before.json / stats_after.json
    ├── report.md
    └── plots/                       7+ PNG 分布图
```

---

## 八、关键约定（勿改）

> 改动前必须重新跑 `sanity_heading_alignment.py` 验证。

| 约定 | 值 |
|---|---|
| 航向公式 | `heading = atan2(vx_east, vy_north)` |
| 零位 | North（正北） |
| 正方向 | 顺时针（CW） |
| dpsi_sign | **+1**（右转 → 正值） |
| AFSIM 接口 | `TurnToRelativeHeading(deg)` → 调用前 `math.degrees(dpsi)` |
| 训练单位 | rad / m / m/s |
| AFSIM 单位 | deg / m / m/s |
