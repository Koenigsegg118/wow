# WOW Token Runtime — 迁移与交接总结

本文档汇总本轮 AFSIM 适配、控制契约修复、测试与遥测固化，便于在新环境或仓库中复现与迁移。

---

## 1. 范围与约束

| 项目 | 说明 |
|------|------|
| **已冻结（未改）** | VQ-VAE checkpoint、one-step token policy checkpoint、vocab、训练主线、tokenizer |
| **本仓库负责** | `token → AFSIM` 运行时：bridge / sweep / shadow、场景侧 `action[3]` 接 G、回归测试、JSONL 遥测、文档 |

---

## 2. 动作语义（单一事实来源）

- **模型输出**：严格 **3 维** `[dpsi_rad, dalt_m, dspd_mps]`，chunk 形状 `[4, 3]`。
- **语义**：三者均为 **2 秒 lookahead 增量**（非瞬时、非 chunk 内四行求和）。
- **Chunk 四行**：重叠时间窗的标签，描述同一段轨迹；**默认 runtime 只执行 `chunk[0]`**（receding-horizon / `first_row`）。
- **Heading**：相对转弯 → `psi_dot = dpsi / H_action_sec`，`turn_deg = deg(psi_dot × dt_cmd)`（horizon-fraction）。
- **高度/速度**：绝对 setpoint → `target = current + delta`（**不做** heading 那套 fraction），在 token 边界计算后经 **逐通道 latch** 持有。

---

## 3. 默认推荐配置（固定）

```text
--exec_mode first_row
--g_mode dynamic
--g_max 7.0
--latch enabled（逐通道独立阈值）
```

典型阈值：`heading_update_thresh_deg=3`，`alt_update_thresh_m=50`，`speed_update_thresh_mps=10`。

---

## 4. 关键文件路径

| 用途 | 路径 |
|------|------|
| 在线 bridge | `wow/sim/token_bridge_server.py` |
| Token sweep（联调逐个 token） | `wow/sim/token_sweep_server.py` |
| Shadow（只读对照） | `wow/sim/token_shadow.py` |
| Policy + VQ 推理封装 | `wow/sim/token_policy_runtime.py` |
| Sweep 日志分析 | `wow/tools/analyze_sweep.py` |
| 离线验收脚本 | `wow/tools/final_audit.py` |
| 契约/审计/收尾文档 | `wow/docs/timebase_audit.md`（含 Part 8 工程固化） |
| 回归测试（本轮新增） | `wow/sim/tests/test_regression.py` |
| 原 bridge 测试 | `wow/sim/tests/test_token_bridge.py` |
| AFSIM 场景（动态 G） | `build/demos/air_to_air/2v2_p6dof_*.txt`（若干，均从 `GetAction(i,3)` 读 G） |

**典型 checkpoint / 数据（相对 `wow/`）**

- Policy: `checkpoints/onestep_token_bc_t4_cb64_h4/best.pt`
- VQ: `checkpoints/vqvae_clean_t4_cb64/best.pt`
- Vocab: `datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json`

---

## 5. 启动方式

**环境**：建议使用已安装 `torch` 的 conda 环境（如 `wow`）。

### 5.1 Token Sweep（可继续用「原来那套」方式）

先启动 Python 端，再启动 AFSIM 连 `localhost:65432`。

```powershell
cd d:\AF2.9\afsim-2.9.0-win64\wow

python -m sim.token_sweep_server ^
  --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt ^
  --vocab datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json ^
  --exec_mode first_row ^
  --g_mode dynamic ^
  --g_max 7.0 ^
  --log_path logs/token_sweep.jsonl
```

**只测单个 token（如 T29）并重复：**

```powershell
python -m sim.token_sweep_server ^
  --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt ^
  --vocab datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json ^
  --single_token 29 ^
  --exec_mode first_row --g_mode dynamic ^
  --log_path logs/sweep_T29.jsonl
```

**指定多个 token：**

```powershell
--tokens 29,14,15,25
```

AFSIM：在 `build/demos/air_to_air` 下用与联调一致的场景（如 `2v2_p6dof_model_only.txt` 或你当前用的 model-only 场景），保证 `SAC_PROCESSOR` 与 bridge 协议一致。

### 5.2 Token Bridge（在线策略）

```powershell
python -m sim.token_bridge_server ^
  --policy_ckpt checkpoints/onestep_token_bc_t4_cb64_h4/best.pt ^
  --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt ^
  --vocab datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json ^
  --mode full --exec_mode first_row --g_mode dynamic --g_max 7.0
```

（具体 CLI 以 `token_bridge_server.py` 内 `argparse` 为准。）

### 5.3 分析 Sweep 日志

```powershell
python tools/analyze_sweep.py logs/token_sweep.jsonl
```

---

## 6. AFSIM 侧约定

- `action[base+0]`：相对转弯角（度）
- `action[base+1]`：高度目标（m）
- `action[base+2]`：速度目标（m/s）
- **`action[base+3]`：G 载荷（m/s²）**，即 `n_cmd × 9.81`；由 Python runtime 计算，**不属于 ML 的 3 维动作**。

连续 BC 的 `bc_policy_socket_server` 需对 `action[3]` 写入固定 3G 等值以兼容上述场景（若场景已改为读 `action[3]`）。

---

## 7. JSONL 遥测（统一命名）

Bridge / Sweep / Shadow 对齐使用（部分字段仅 bridge 有全量计数器）：

- `token_action_3d`：ML 侧 3 维（或 sweep 中 `[dpsi, dalt, dspd]` 解码值）
- `runtime_control_4d`：`[turn_deg, alt_m, spd_mps, g_mps2]`
- `turn_rate_cmd_deg_s` / `turn_rate_measured_deg_s` / `turn_rate_theory_deg_s`
- `heading_task_reset_count`、`alt_task_reset_count`、`spd_task_reset_count`
- `heading_latch_hold_count`、`alt_latch_hold_count`、`spd_latch_hold_count`
- `g_saturation_count`、`g_saturation_ratio`、`n_req`、`n_cmd`

Sweep 中旧字段 `rel_action` / `abs_command` 已重命名为 **`token_action_3d` / `runtime_control_4d`**（含义相同）。

---

## 8. 测试与验收

| 内容 | 说明 |
|------|------|
| 回归 | `pytest wow/sim/tests/test_regression.py`（heading 累计、逐通道 latch、3D/4D 隔离、alt/spd 无 fraction、60s/120s soak） |
| 原 bridge 套件 | `pytest wow/sim/tests/test_token_bridge.py` |
| 全量 `sim/tests` | 可能含 **1 个与 BC `SmoothSetpointController` 相关的预存失败**（`test_same_physical_rate_across_dt`），与 token 栈无关 |

---

## 9. 本轮重要修复摘要

1. **Chunk**：默认不再顺序 replay 四行；`first_row` 为默认，`replay_4rows` 仅对照。
2. **Heading**：`dpsi` 按 `H_action_sec` 与 `dt_cmd` 做分帧，避免过量转弯命令。
3. **动态 G**：`n_req = sqrt(1 + (V·ψ̇/g)²)`，`n_cmd = clip(n_req, 1, g_max)`，经 `action[3]` 下发。
4. **Latch**：**逐通道**更新，避免「speed 触发更新却把 alt 从当前位置再加一遍 delta」的叠加错误。
5. **文档**：详见 `docs/timebase_audit.md`（审计历史 + Part 8 固化结论）。

---

## 10. 迁移检查清单

- [ ] 复制 `wow/sim/` 下 bridge、sweep、shadow、runtime、guardrails
- [ ] 复制 `wow/sim/tests/test_regression.py` 与相关测试
- [ ] 复制 `wow/docs/timebase_audit.md`、`MIGRATION_HANDOFF.md`
- [ ] 确认 checkpoint、vocab 路径或环境变量
- [ ] 确认 AFSIM 场景读取 `action[3]` 为 G（m/s²）
- [ ] 在目标环境运行：`pytest sim/tests/test_regression.py sim/tests/test_token_bridge.py`

---

*文档版本：与当前实现一致；默认控制语义与 CLI 默认值未再改动。*
