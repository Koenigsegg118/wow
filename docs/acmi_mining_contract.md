# ACMI Mining Contract — obs-sidecar 对齐说明

## 目的

本文档说明 ACMI 事件挖掘管线如何与已有 runtime obs-sidecar 对齐，
确保 ACMI 采样产出的 obs 结构可直接进入 D2b 标注管线。

---

## 几何约定（必须与 d2_runtime_obs_contract.md 一致）

| 字段 | 公式 | 符号约定 |
|------|------|---------|
| `range_m` | `‖enemy.pos - ego.pos‖` | 正值 |
| `bearing_deg` | `LOS_angle - ego_heading`，机头=0 | 正=右侧，[-180, 180] |
| `aspect_deg` | `LOS_enemy_to_ego_angle - enemy_heading` | 0=迎头，180=尾追 |
| `closure_mps` | `d(range)/dt` | **负=接近，正=远离** |
| `alt_diff_m` | `ego_alt - enemy_alt` | **正=己方更高** |

### 注意

- `bearing_deg` 在 ego body frame 下计算，不是 ENU 全局角
- `aspect_deg` 取绝对值后范围 [0, 180]
- `closure_mps` 通过相邻帧 range 差分计算

---

## ACMI 到 obs-sidecar 的映射桥

ACMI 解析后每个实体每个时间步有：
```
x (east, m), y (north, m), z (alt, m)
vx (east, m/s), vy (north, m/s), vz (up, m/s)
heading_u (rad, 0=N, CW+, unwrapped)
spd (ground speed, m/s)
```

obs-sidecar 字段映射：

| obs-sidecar 字段 | ACMI 来源 | 变换 |
|-----------------|----------|------|
| ego.speed_mps | spd | 直接 |
| ego.altitude_m | z | 直接 |
| ego.heading_deg | degrees(heading_u) | rad→deg |
| ego.pitch_deg | atan2(vz, sqrt(vx²+vy²)) | rad→deg |
| ego.roll_deg | null | ACMI 不提供 |
| ego.vertical_speed_mps | vz | 直接 (正=向上) |
| ego.pos_east_m | x | 相对参考点 |
| ego.pos_north_m | y | 相对参考点 |
| ego.energy_state | null | 需要 q30/q70 配置 |
| ego.runtime_id | 名称规范化 | e.g. "red_1" |
| ego.coalition | Coalition 属性 | "red"/"blue" |

### enemy/ally 选择

与 runtime sidecar 一致：
1. **enemy**：sticky target 优先（target_switch_ratio=0.8）
2. **ally**：同阵营最近有效实体
3. 无匹配时 = null

### engagement 标志

阈值引用 `annotation_guideline_v1.md`（唯一来源）：
- `is_merge`: range < MERGE_RANGE_M (3000m)
- `is_defensive`: aspect > DEF_ASPECT_DEG (120) AND range < DEF_RANGE_M (15000)
- `has_shot_opportunity`: null（无可靠来源）

---

## 与 runtime sidecar 的差异

| 项目 | runtime sidecar | ACMI mining |
|------|----------------|-------------|
| 数据源 | AFSIM 640 值帧 (lat/lon) | ACMI 解析 (ENU 坐标) |
| 位置坐标 | lat/lon → ENU 转换 | 直接 ENU |
| 速度 | 帧内字段 | 轨迹差分 |
| heading | 帧内字段 | atan2(vx, vy) |
| roll | null | null |
| energy_state | null | null |
| has_shot_opportunity | null | null |

**关键保证**：
几何计算公式（bearing/aspect/closure/alt_diff）完全相同，
只是输入坐标来源不同。最终产出的 obs 结构与 runtime sidecar 格式相同，
可直接送入 D2b adapter / rule_labeler / GPT hard-case 流程。

---

## 不可得字段

以下字段在 ACMI mining 阶段统一为 null：

| 字段 | 原因 |
|------|------|
| ego.roll_deg | ACMI 不记录横滚 |
| ego.energy_state | 需要数据集 q30/q70 配置 |
| engagement.has_shot_opportunity | 无武器系统建模 |
| history.prev_semantic_state | 首次标注，无历史 |
| history.prev_token_family | 非 VQ 管线 |
| ally.is_supporting | 无编队协议 |
