# D2 Runtime Obs-Sidecar Contract

## 目标

从 AFSIM 640 值帧中实时提取与 `semantic_record_schema_v1.json` obs 节对齐的态势快照，供后续 semantic layer / AI 自动标注使用。

**不是** full canonical record — 只输出 obs 部分（ego/enemy/ally/engagement/history），不包含 source/label/evidence/quality/extras。

---

## 主链路零污染声明

D2 不修改以下任何模块：
- VQ tokenizer (t4_cb64)
- OneStepTokenBC (obs_hist=4)
- VQ decode chunk[4,3]
- bridge first_row
- P6DOF profile 主逻辑

---

## 640 值帧字段来源

| 偏移 | 字段 | 单位 | 说明 |
|------|------|------|------|
| base+0 | live | - | 1.0=alive |
| base+1 | lat | deg | WGS-84 |
| base+2 | lon | deg | WGS-84 |
| base+3 | alt_m | m | MSL |
| base+4 | velN | m/s | 北向 |
| base+5 | velE | m/s | 东向 |
| base+6 | velD | m/s | **正向下** (转 vz_u = -velD) |
| base+7 | heading_deg | deg | AFSIM 诊断值 |

平台映射: idx 0=red_1, 1=red_2, 2=blue_1, 3=blue_2

---

## Sidecar 输出字段映射

### ego

| 字段 | 来源 | 当前状态 |
|------|------|---------|
| runtime_id | PLATFORM_NAMES[idx] | 可得 |
| callsign | 同 runtime_id | 可得 |
| coalition | COALITION_MAP[idx] | 可得 |
| speed_mps | sqrt(velE^2 + velN^2) | 可得 |
| altitude_m | frame[base+3] | 可得 |
| heading_deg | atan2(velE, velN), fallback to frame[base+7] | 可得 |
| pitch_deg | asin(vz_u / speed_3d) | 可得 (近似) |
| roll_deg | - | **null** (640 帧不含 roll) |
| vertical_speed_mps | -velD | 可得 |
| pos_east_m | lat/lon → ENU 东向 | 可得 |
| pos_north_m | lat/lon → ENU 北向 | 可得 |
| energy_state | E_spec = alt + v^2/(2g) → 分桶 | **null** (q30/q70 阈值未配置) |

### enemy (单对象或 null)

| 字段 | 计算公式 | 符号约定 |
|------|---------|---------|
| range_m | sqrt(dx^2+dy^2+dz^2) | 正值 |
| bearing_deg | atan2(dx,dy) - ego_heading, wrap [-180,180] | 正=右舷 |
| aspect_deg | abs(wrap(LOS_to_ego - enemy_heading)) | 0=迎头, 180=尾追 |
| closure_mps | (rel_pos · rel_vel) / range | 负=接近, 正=远离 |
| alt_diff_m | ego_alt - enemy_alt | 正=己方更高 |

### engagement

| 字段 | 判定逻辑 | 阈值来源 |
|------|---------|---------|
| is_merge | range < 3000m | annotation_guideline_v1.md |
| is_defensive | aspect>120° AND closure<0 AND range<15000m | annotation_guideline_v1.md |
| has_shot_opportunity | - | **null** (无武器包线) |

### history

| 字段 | 来源 | 当前状态 |
|------|------|---------|
| prev_semantic_state | SidecarContext 维护 | null (未接入 semantic layer) |
| prev_token_family | SidecarContext 维护 | null (未接入) |
| prev_target_ref | SidecarContext 维护 | 可得 (sticky target 自动跟踪) |

---

## Target 选择逻辑

**Sticky-target** (稳定优先):
1. 若上一帧的目标仍然活着，优先沿用
2. 仅当新目标距离 < 旧目标距离 × 0.8 时才切换
3. 无历史时选最近有效敌机

配置: `semantic_thresholds.py::TARGET_SWITCH_RATIO = 0.80`

---

## 集成点

```python
from sim.runtime.afsim_obs_sidecar import build_obs_sidecar, SidecarContext

ctx = SidecarContext()
# 在每帧收到 640 值后:
obs = build_obs_sidecar(frame_values, ego_platform_idx=0, ctx=ctx)
# obs 可直接作为 AI 标注 format_A 输入
```

---

## 文件清单

| 文件 | 职责 |
|------|------|
| `sim/runtime/semantic_thresholds.py` | 所有阈值/常量的唯一来源 |
| `sim/runtime/afsim_obs_entities.py` | 640帧 → EntityState 解析 |
| `sim/runtime/afsim_obs_geometry.py` | range/bearing/aspect/closure/alt_diff 计算 |
| `sim/runtime/afsim_obs_sidecar.py` | 总入口: build_obs_sidecar() |
| `sim/runtime/tests/test_afsim_obs_sidecar.py` | 5 类场景测试 |
