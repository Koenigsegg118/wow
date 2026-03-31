# 三路数据对齐检查清单 (v1.1)

用于验证 manual / acmi_ai / afsim_rule 三种标注来源是否与 canonical schema 对齐。

> **v1.1 关键约束**: 样本粒度=单一时刻; canonical 内部用 null 不用 "unknown"; enemy/ally=单对象; engagement 阈值唯一来源=annotation_guideline_v1.md; conflict_flag=true -> tier 上限=silver。

---

## 检查方法

对每个检查项，逐一验证三种来源。通过标 PASS，不通过标 FAIL 并注明原因。

---

## 1. 字段名一致性

| 检查项 | manual | acmi_ai | afsim_rule | 说明 |
|--------|--------|---------|------------|------|
| schema_version = "semantic_record_v1" | | | | 所有记录必须包含 |
| sample_id 格式 = {type}_{scenario}_{time}_{seq} | | | | 前缀应匹配 source.type |
| source.platform 使用 enum_registry 枚举 | | | | afsim_acmi/manual/mixed |
| source.type 使用 enum_registry 枚举 | | | | manual/acmi_ai/afsim_rule/mixed |
| obs.ego 字段名与 schema 完全匹配 | | | | 不允许缩写或别名 |
| obs.enemy 字段名与 schema 完全匹配 | | | | target_ref 不是 target_id |
| obs.ally 字段名与 schema 完全匹配 | | | | ally_ref 不是 ally_id |
| label 字段名与 schema 完全匹配 | | | | semantic_state 不是 state |
| evidence 字段名与 schema 完全匹配 | | | | rule_ids 是数组 |
| quality 字段名与 schema 完全匹配 | | | | tier 不是 level |

---

## 2. 枚举值一致性

| 检查项 | manual | acmi_ai | afsim_rule | 说明 |
|--------|--------|---------|------------|------|
| label.semantic_state 仅使用 9 个枚举值 + null | | | | 见 enum_registry |
| label.role 仅使用 4 个枚举值 + null | | | | engaged/support/egressing/neutral |
| label.target_ref 仅使用 3 个枚举值 + null | | | | enemy_1/enemy_2/ally_1 |
| label.profile_hint 仅使用 3 个枚举值 + null | | | | p6dof_semantic/p6dof_aggressive_turn/auto |
| quality.tier 仅使用 4 个枚举值 | | | | gold/silver/bronze/weak |
| obs.ego.coalition 仅使用 3 个枚举值 + null | | | | red/blue/neutral |
| obs.ego.energy_state 仅使用 3 个枚举值 + null | | | | low/medium/high |
| obs.enemy.target_ref 仅使用 2 个枚举值 + null | | | | enemy_1/enemy_2 |
| obs.ally.ally_ref 仅使用 1 个枚举值 + null | | | | ally_1 |

---

## 3. 单位一致性

| 字段 | 预期单位 | manual | acmi_ai | afsim_rule |
|------|---------|--------|---------|------------|
| speed_mps | m/s | | | |
| altitude_m | m | | | |
| heading_deg | deg (0=北, 顺时针正) | | | |
| pitch_deg | deg | | | |
| roll_deg | deg | | | |
| vertical_speed_mps | m/s (正=上升) | | | |
| pos_east_m | m | | | |
| pos_north_m | m | | | |
| range_m | m | | | |
| bearing_deg | deg (正=右侧, [-180,180]) | | | |
| aspect_deg | deg (0=头对头, 180=尾追) | | | |
| closure_mps | m/s (负=接近) | | | |
| alt_diff_m | m (正=己方更高) | | | |
| timestamp_sec | s | | | |

---

## 4. target_ref 规范一致性

| 检查项 | manual | acmi_ai | afsim_rule | 说明 |
|--------|--------|---------|------------|------|
| enemy_1/enemy_2 是角色中性 (不含 red/blue) | | | | |
| 同一 episode 内 enemy_1/enemy_2 分配一致 | | | | |
| ally_1 固定指友方 (不含 teammate 等别名) | | | | |
| target_ref=null 仅用于无目标场景 | | | | 不用 "none" 字符串 |
| event_flags.target_switch 与 prev_target_ref 逻辑一致 | | | | |

---

## 5. null / 缺失语义一致性

| 检查项 | manual | acmi_ai | afsim_rule | 说明 |
|--------|--------|---------|------------|------|
| null 表示"信息不足", 不表示"否" | | | | |
| 不使用字符串 "unknown" / "unavailable" / "N/A" | | | | |
| semantic_state=null 不被替换为 hold_geometry | | | | |
| constraints 子字段允许独立为 null | | | | |
| obs.enemy 整体为 null (无敌方信息) vs 部分为 null | | | | |
| obs.ally 整体为 null vs 部分为 null | | | | |

---

## 6. quality tier 一致性

| 检查项 | manual | acmi_ai | afsim_rule | 说明 |
|--------|--------|---------|------------|------|
| manual 标注 tier >= silver | | | | 人工标注不应为 weak |
| acmi_ai 高置信 (>0.8) = silver | | | | |
| acmi_ai 低置信 (<0.5) = weak | | | | |
| afsim_rule 无人工确认 = bronze | | | | |
| conflict_flag=true 时 tier <= **silver** (未审查时推荐 bronze) | | | | |
| needs_review 与 tier 逻辑一致 | | | | gold 不需要 review |

---

## 7. evidence 可追溯性

| 检查项 | manual | acmi_ai | afsim_rule | 说明 |
|--------|--------|---------|------------|------|
| acmi_refs 格式 = filename:object_id:timestamp | | | | |
| frame_refs 格式 = frame_{simTime} | | | | |
| rule_ids 使用统一编号 (R01, R02, ...) | | | | |
| 至少有一种 evidence 非空 | | | | 完全无证据的标注需要说明 |
| weapon_event_refs (如有) 可追溯到 acmi2weapon_events 输出 | | | | |

---

## 8. 样本粒度一致性

| 检查项 | manual | acmi_ai | afsim_rule | 说明 |
|--------|--------|---------|------------|------|
| 每条记录对应单一时刻 (非窗口) | | | | |
| timestamp_sec 精度一致 (秒级) | | | | |
| ego 观测为该时刻的瞬时值 | | | | |
| enemy/ally 几何为该时刻的瞬时计算值 | | | | |
| history 反映前一时刻的标注输出 | | | | |

---

## 执行建议

1. 首批验证: 取每种来源 5 条记录, 逐项填表
2. 自动化: 编写 `validate_canonical_record.py` 做枚举/类型/范围自动检查
3. 人工抽检: 重点关注 null 语义、target_ref 分配、单位
4. 迭代: 发现不一致后更新 schema 或标注规范, 不要在各来源中各自修补
