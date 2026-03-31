# WOW Token → AFSIM 运行时控制契约审计

## Part 1: Timebase / Action Contract 审计表

| 参数 | 真实值 | 来源文件 & 行号 | 说明 |
|------|--------|----------------|------|
| **dt_data** | **0.5 s** | `ml/dataset_default_config.py:21` `DT = 0.5` | 训练数据重采样周期，2 Hz |
| **control_frequency_hz** | **2.0 Hz** | `ml/dataset_default_config.py:25` `CONTROL_FREQUENCY_HZ = 1.0/DT` | 训练时控制频率 |
| **lookahead_steps (K)** | **4** | `ml/dataset_default_config.py:23` `K = 4` | action label 里 +4 个采样步 |
| **H_action_sec** | **2.0 s** | `ml/dataset_default_config.py:22` `H_SEC = 2.0`; `K * DT = 4 * 0.5 = 2.0` | dpsi/dalt/dspd 的真实时间 horizon |
| **chunk_len (token_steps)** | **4** | VQ-VAE ckpt `args.token_steps = 4` | VQ 编码/解码的 chunk 长度 |
| **chunk[i] 语义** | **action[t + i·dt_data]** | `build_vq_clean_dataset.py:167-183` | 连续 4 步, 每步都是 2s lookahead delta |
| **dt_policy (训练时)** | **2.0 s** | `token_steps × dt_data = 4 × 0.5` | 训练时 token 边界间隔 |
| **dt_cmd (运行时)** | **≈0.2 s** | 实测 AFSIM SAC_PROCESSOR 帧率 | bridge 与 AFSIM 交互周期 |
| **dt_sim** | **0.05 s** | `scenarios/2v2_p6dof_model_only.txt:75` `update_interval 0.05 s` | AFSIM 物理步长 / draw_processor 轮询间隔 |
| **dt_policy (运行时, 当前错误)** | **0.8 s** | `token_bridge_server.py:309` `chunk_index >= token_steps (=4)`; 4 × 0.2s = 0.8s | 每 4 帧触发一次预测, 但应为 2.0s |
| **当前 runtime 是否把 chunk 四行顺序执行** | **是 (BUG)** | `token_bridge_server.py:337` `tick_action = chunk_buffer[chunk_index]` | 应该只用 chunk[0] |

## Action 语义确认 (VQ-clean 数据管线)

来源: `training/vq/build_vq_clean_dataset.py:167-183`

```
action = [dpsi_rad, dalt_sp_m, dspd_sp_mps]
  dpsi_rad    = wrap_pi(heading[t+K] - heading[t])     ← 2s lookahead heading DELTA
  dalt_sp_m   = z[t+K] - z[t]                          ← 2s lookahead altitude DELTA
  dspd_sp_mps = spd[t+K] - spd[t]                      ← 2s lookahead speed DELTA
```

**关键**: 全部三个维度都是 2s-lookahead **DELTA**, 非绝对值.
这与连续 BC 策略不同 (BC 的 alt_sp/spd_sp 是绝对 setpoint).

chunk 的 4 行是来自连续时间步的重叠 lookahead 窗口:
```
chunk[0] = [h(t+4)-h(t),     z(t+4)-z(t),     s(t+4)-s(t)]       t=0
chunk[1] = [h(t+5)-h(t+1),   z(t+5)-z(t+1),   s(t+5)-s(t+1)]     t=0.5s
chunk[2] = [h(t+6)-h(t+2),   z(t+6)-z(t+2),   s(t+6)-s(t+2)]     t=1.0s
chunk[3] = [h(t+7)-h(t+3),   z(t+7)-z(t+3),   s(t+7)-s(t+3)]     t=1.5s
```
4 个值描述 **同一段轨迹** 的滑动窗口, 不是 4 个独立连续动作.

## 当前 runtime 中确认的 BUG

### BUG-1: chunk 被当作 4 个顺序动作 replay
- 位置: `token_bridge_server.py:337`
- 现象: `chunk_buffer[chunk_index]` 顺序取 index 0→1→2→3
- 正确行为: 只用 chunk[0] (receding-horizon) 或 mean/median 压缩

### BUG-2: dpsi 未做 horizon-fraction 转换
- 位置: `token_bridge_server.py:353`
- 现象: `turn_deg = degrees(dpsi_rad)` 直接把 2s 全量 delta 当作每帧 TurnToRelativeHeading 命令
- 应为: `turn_deg = degrees(dpsi_rad × dt_cmd / H_action_sec)` = dpsi × 0.1
- 影响: heading 命令 10× 过大

### BUG-3: timebase 不匹配未处理
- 训练: chunk 跨 4 × 0.5s = 2.0s
- 运行时: chunk 在 4 × 0.2s = 0.8s 内耗尽
- chunk 消耗速度快 2.5× (由 BUG-1 导致, 修复 BUG-1 后此问题消失)

### BUG-4: TurnToRelativeHeading 每帧重发
- 位置: AFSIM script `platform.TurnToRelativeHeading(turn_deg, 3.0*9.8)` 在每个新 action_seq 时调用
- 效果: 目标永远在当前 heading 前方 20.6°, 飞机永远追不上

### BUG-5: 固定 3G 不足以跟踪大转弯 token
- T29: dpsi≈0.36rad → 10.3°/s turn rate needed
- 3G @274m/s → max 5.8°/s sustainable → 只能跟踪 56% 的指令
- 应动态计算 G: n = sqrt(1 + (V·ψ̇/g)²)

### BUG-6: token_sweep_server.py 同样存在 BUG-2 和 BUG-5
- 位置: `token_sweep_server.py:292-294`
- 同样的 dpsi 直接使用 + 固定 3G 问题

### BUG-7: token_shadow.py 同样存在 BUG-1 和 BUG-2
- 位置: `token_shadow.py:233` — shadow 建议的 action 也未做 horizon-fraction 转换

## BC 策略 bridge 参考实现 (已正确)

`afsim_policy_bridge.py` 中的 `SmoothSetpointController` 已正确实现:
```python
horizon_fraction = min(dt / lookahead_horizon_s, 1.0)  # 0.2/2.0 = 0.1
dpsi_tick_target = dpsi_lookahead * horizon_fraction     # 只取 10%
```
Token bridge 应采用相同逻辑.

---

## Part 7: 最终结论报告

### 1. Action Timebase 审计结论

| 参数 | 训练侧 | 运行时 (修复前) | 运行时 (修复后) |
|------|--------|----------------|----------------|
| dt_data / dt_cmd | 0.5s | 0.2s | 0.2s (不变) |
| policy 调用间隔 | 2.0s (token_steps×dt) | 0.8s (4×0.2) | 0.8s (不变, 但语义正确) |
| chunk 消耗方式 | 4行=2s轨迹描述 | 顺序播放4行×0.2s | 只用第1行, 持续4帧 |
| dpsi→heading 映射 | 2s lookahead delta | 直接当单帧命令 | `dpsi * dt/H_sec` 分帧 |
| alt/spd 映射 | 2s delta setpoint | 每帧重算 `cur+delta` | token边界计算一次,latch持有 |
| G-load | N/A | 固定3G硬编码 | 动态计算 n=√(1+(Vψ̇/g)²) |

### 2. 当前 runtime 是否误把 lookahead chunk 当 4 个顺序动作执行

**是, 确认为 BUG-1.**

旧代码 `token_bridge_server.py:337`:
```python
tick_action = ps.chunk_buffer[ps.chunk_index].copy()  # index 0→1→2→3
```
将 chunk 的 4 行依次播放. 但 chunk 的 4 行是同一段轨迹的重叠 2s 窗口,
不是 4 个独立连续动作.

**修复**: 默认 exec_mode=first_row, 只使用 chunk[0]. 旧行为保留为 `--exec_mode replay_4rows` 供对照.

### 3. 固定 3G 是否足以跟踪 T29

**不足.**

T29 解码: dpsi ≈ -0.36 rad (≈ -20.6°/2s)

```
psi_dot_cmd = 0.36 / 2.0 = 0.18 rad/s = 10.3°/s
V = 274 m/s (900 ft/s)

n_required = sqrt(1 + (274 × 0.18 / 9.81)²) = sqrt(1 + 5.03²) = sqrt(26.3) ≈ 5.1G

3G max turn rate: g·sqrt(n²-1)/V = 9.81·sqrt(8)/274 = 0.101 rad/s = 5.8°/s
5.1G max turn rate: g·sqrt(n²-1)/V = 9.81·sqrt(25.01)/274 = 0.179 rad/s = 10.3°/s
```

在 3G 限制下, 理论最大稳态转弯率 = 5.8°/s, 只能跟踪 T29 的 56%.

### 4. Measured turn rate 低于理论 3G 上限多少

在旧 runtime (BUG-1 + BUG-2 + 固定3G) 下:
- 实测 T29 约 3.6°/s (来自之前的 sweep 数据)
- 理论 3G 上限 5.8°/s
- 比值: 3.6/5.8 = **62%**

额外差距(62% vs 100%)来自:
- BUG-2: dpsi 未做 horizon-fraction, 每帧发 20.6° 而非 2.06°,
  导致 TurnToRelativeHeading 目标永远在 20° 外, P6DOF 的滚入响应跟不上
- BUG-4: 每帧重发高层任务, 导致 P6DOF 无法稳定进入持续转弯

### 5. 根因诊断

| 因素 | 影响程度 | 说明 |
|------|---------|------|
| **语义错配 (BUG-1+2)** | **高** | chunk 顺序播放 + dpsi 未做 horizon-fraction, 命令量级完全错误 |
| **固定 3G 不足** | **中** | 5.1G 需求 vs 3G 限制, 缺口 ≈ 44% |
| **高频重置任务 (BUG-4)** | **中** | 每 0.2s 重发不同的 chunk row 作为新目标, P6DOF 永远在 roll-in |
| **纯动力学滚入延迟** | **低** | P6DOF 滚入延迟 ≈ 1-2s, 但不是主因 |
| **多因素叠加** | **是** | 语义错配是根因, 固定G和任务重置放大了问题 |

**结论**: T29 跟踪不足是**多因素叠加**, 但**语义错配 (BUG-1+2) 是根因**.
固定 3G 是次要因素, 任务重置是第三因素, 纯动力学延迟影响最小.

### 6. 推荐默认线上模式

```
--exec_mode first_row --g_mode dynamic --g_max 7.0 --latch enabled
```

理由:
- `first_row`: 语义最正确, chunk[0] 是当前决策点的 action
- `dynamic`: 自动计算所需 G-load, 不需要调参
- `g_max=7.0`: 战斗机典型极限 7-9G, 7G 保守安全
- `latch=enabled`: alt/spd 在 token 边界计算一次, 不每帧重设

### 7. 验收检查清单

| 条件 | 状态 | 文件/证据 |
|------|------|----------|
| 明确 timebase 表 | ✅ | `docs/timebase_audit.md` Part 1 |
| 确认并修正 chunk 执行语义 | ✅ | `token_bridge_server.py` ExecMode enum, 默认 first_row |
| runtime 默认不再盲目 replay 4 行 | ✅ | 默认 `--exec_mode first_row` |
| 支持动态 G | ✅ | `compute_dynamic_g()`, AFSIM 场景读取 action[3] |
| 支持 latch 防止每帧高层任务重置 | ✅ | 逐通道 `_per_channel_latch_check()` |
| token sweep 能比较不同 exec_mode | ✅ | `token_sweep_server.py` 支持 6 种组合 |
| 最终报告能说明 T29 差距来源 | ✅ | Part 7 第 4-5 节 |

---

## Part 8: 工程收尾 — 动作契约与默认配置固化

### 1. VQ / Policy 动作契约

模型输出严格为 **3 维**：`[dpsi_rad, dalt_m, dspd_mps]`

- chunk 形状：`[4, 3]` — 4 行重叠 2s-lookahead 描述
- chunk 的 4 行描述**同一段轨迹**的滑动窗口，不是 4 个独立连续动作
- 默认 runtime 采用 **receding-horizon**：只执行 `chunk[0]`（first_row 模式）

### 2. 控制通道映射

| 通道 | 动作语义 | AFSIM 命令类型 | Runtime 映射 |
|------|---------|--------------|-------------|
| **heading (dpsi)** | 2s-lookahead heading delta (rad) | `TurnToRelativeHeading(deg, G)` — 相对命令 | `turn_deg = deg(dpsi / H_sec × dt)` — 需要 horizon-fraction |
| **altitude (dalt)** | 2s-lookahead altitude delta (m) | `GoToAltitude(target, rate)` — 绝对 setpoint | `target = current_alt + dalt` — 不做 fraction |
| **speed (dspd)** | 2s-lookahead speed delta (m/s) | `GoToSpeed(target)` — 绝对 setpoint | `target = current_spd + dspd` — 不做 fraction |

**heading 使用 fraction 而 alt/spd 不使用**：因为 heading 是相对命令（每帧累加），而 alt/spd 是绝对 setpoint（设一次目标，飞机自行收敛）。

### 3. Runtime 可附加第 4 维控制字段

- `action[base+3] = g_cmd_mps2` — 动态 G 载荷（runtime 计算，非模型输出）
- JSONL 日志中严格分离：
  - `token_action_3d`: ML 模型原始 3D 输出
  - `runtime_control_4d`: 发送给 AFSIM 的 4D 控制帧
  - `decoded_chunk`: 完整 `[4,3]` 解码 chunk

### 4. Latch 机制

**逐通道独立 latch**（`_per_channel_latch_check` 返回 3-tuple）：

- 每个通道（heading / altitude / speed）独立判断是否需要更新
- 一个通道的阈值触发**不会**导致其他通道的 setpoint 被重新计算
- 防止 alt/spd 从飞机的新位置重复叠加 delta

默认阈值：
- heading: 3 deg
- altitude: 50 m
- speed: 10 m/s

### 5. 默认推荐配置（固定，不再改动）

```
--exec_mode first_row
--g_mode dynamic
--g_max 7.0
--latch enabled
--heading_update_thresh_deg 3.0
--alt_update_thresh_m 50.0
--speed_update_thresh_mps 10.0
```

### 6. Telemetry 统一字段

以下字段在 bridge / shadow / sweep 三处保持一致命名：

| 字段 | 说明 |
|------|------|
| `turn_rate_cmd_deg_s` | 命令转弯速率 (deg/s) |
| `turn_rate_measured_deg_s` | 实测转弯速率 (deg/s) |
| `turn_rate_theory_deg_s` | 理论 G 限稳态转弯速率 |
| `token_action_3d` | ML 原始 3D 动作 [dpsi, dalt_delta, dspd_delta] |
| `runtime_control_4d` | 发送 AFSIM 的 4D 控制 [turn, alt, spd, g] |
| `heading_task_reset_count` | heading 通道累计重置次数 |
| `alt_task_reset_count` | altitude 通道累计重置次数 |
| `spd_task_reset_count` | speed 通道累计重置次数 |
| `heading_latch_hold_count` | heading 通道累计持有次数 |
| `alt_latch_hold_count` | altitude 通道累计持有次数 |
| `spd_latch_hold_count` | speed 通道累计持有次数 |
| `g_saturation_count` | G 载荷饱和累计帧数 |
| `g_saturation_ratio` | G 饱和比率 (0~1) |
| `n_req` | 需求 G 载荷 |
| `n_cmd` | 命令 G 载荷 (clamp 后) |

### 7. 回归测试覆盖 (80 tests, all pass)

| 测试类 | 覆盖内容 | 测试数 |
|--------|---------|--------|
| TestGuardrailTickChecks | 逐帧幅值/NaN/Inf 检查 | 8 |
| TestGuardrailChunkChecks | chunk 级置信度/跳变检查 | 6 |
| TestFallbackResolution | fallback 策略 | 3 |
| TestChunkQueueMechanics | 4 步 chunk 消耗/重预测 | 4 |
| TestRelativeToAbsolute | delta 转绝对值/clamp | 7 |
| TestModeBehaviour | dryrun/limited 模式 | 3 |
| TestRuntimeIntegration | 真实 checkpoint 加载 | 5 |
| TestTokenShadow | shadow 只读/日志字段 | 9 |
| TestObsHistory | H=4 obs 历史/左填充 | 5 |
| TestEgoTransformContract | ego 变换锚点归零 | 3 |
| **TestHeadingAccumulation** | 2s 累计 heading 匹配 token 目标 | **3** |
| **TestPerChannelLatch** | 逐通道 latch 隔离 | **4** |
| **TestActionDimensionIsolation** | action[3] ML/runtime 隔离 | **4** |
| **TestAltSpdRecedingHorizon** | alt/spd 不做 fraction | **4** |
| **TestSoakStability** | 60s/120s × 5 token 稳定性 | **12** |
| **总计** | | **80** |

### 修改文件清单

| 文件 | 改动类型 |
|------|---------|
| `wow/sim/token_bridge_server.py` | **重写** — exec_mode, 动态G, horizon-fraction, latch |
| `wow/sim/token_sweep_server.py` | **重写** — 6 种实验模式, 新日志字段 |
| `wow/sim/token_shadow.py` | **修正** — horizon-fraction, receding-horizon, 动态G |
| `wow/tools/analyze_sweep.py` | **重写** — transient/steady-state 分析 |
| `wow/docs/timebase_audit.md` | **新增** — 审计报告+结论 |
| `wow/sim/tests/test_token_bridge.py` | **更新** — 适配新 API |
| `wow/sim/bc_policy_socket_server.py` | **小改** — action[3]传G值 |
| `build/.../2v2_p6dof_model_only.txt` | **修改** — 动态G从 action[3] 读取 |
| `build/.../2v2_p6dof_bc.txt` | **修改** — 同上 |
| `build/.../2v2_p6dof_bc_close.txt` | **修改** — 同上 |
| `build/.../2v2_p6dof_tspi_blue.txt` | **修改** — 同上 |

### 使用示例

**Token bridge (正式运行)**:
```bash
python -m sim.token_bridge_server \
    --policy_ckpt checkpoints/onestep_token_bc_t4_cb64_h4/best.pt \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --vocab datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json \
    --mode full --exec_mode first_row --g_mode dynamic --g_max 7.0
```

**Token sweep (T29 对比实验)**:
```bash
# 实验 1: first_row + dynamic_g
python -m sim.token_sweep_server \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --vocab datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json \
    --single_token 29 --exec_mode first_row --g_mode dynamic \
    --log_path logs/sweep_T29_fr_dg.jsonl

# 实验 2: first_row + fixed_3g
python -m sim.token_sweep_server \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --vocab datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json \
    --single_token 29 --exec_mode first_row --g_mode fixed --g_fixed 3.0 \
    --log_path logs/sweep_T29_fr_fg.jsonl

# 实验 3: replay_4rows + fixed_3g (旧行为复现)
python -m sim.token_sweep_server \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --vocab datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json \
    --single_token 29 --exec_mode replay_4rows --g_mode fixed --g_fixed 3.0 \
    --log_path logs/sweep_T29_rp_fg.jsonl
```

**分析日志**:
```bash
python tools/analyze_sweep.py logs/sweep_T29_fr_dg.jsonl
```

---

## Part 9: Python 入口 ↔ AFSIM 场景（速查）

AFSIM 通过 `SAC_PROCESSOR` 以 **TCP 客户端**连接到 Python（默认 `localhost:65432`）；**先 Python 监听，后启动仿真**。

| Python | 典型场景（根目录在 `build/demos/air_to_air/`） |
|--------|-----------------------------------------------|
| `sim.token_bridge_server` | `scenarios/2v2_p6dof_token.txt`（或复制 `2v2_model_only.txt` 改 `SCNRIO`）；亦可 `2v2_model_only.txt`、`2v2_kinematic_model_only.txt`、`2v2_bc_close.txt`、`2v2_tspi_blue.txt` |
| `sim.token_sweep_server` | 同上（协议一致即可） |
| `sim.bc_policy_socket_server`（可选 `--token_shadow`） | `2v2_bc_transformer.txt`（`SCNRIO=2v2_p6dof_bc`）、`scenarios/2v2_p6dof_bc.txt`、`2v2_bc_close.txt` 等 |
| `llm_with_connection` | 需单独协议对齐，**不**默认与上表混用 |

完整说明、端口与注意事项见 **`docs/MIGRATION_HANDOFF.md` §4.1**。
