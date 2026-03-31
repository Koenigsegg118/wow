# Canonical Record 字段对照表 (v1.1)

供人工标注员和脚本开发者快速查阅。所有字段均基于 `semantic_record_schema_v1.json`。

> **v1.1 关键规则**:
> - 样本粒度: **单一决策时刻** (非窗口)
> - canonical 内部统一用 **null** 表示缺失, 严禁 `"unknown"` / `"unavailable"` 字符串
> - enemy / ally 均为**单对象或 null**, 不是数组
> - engagement 阈值唯一来源: `annotation_guideline_v1.md`
> - schema 已设 `additionalProperties: false`, 不允许额外字段

---

## 顶层结构

| 路径 | 类型 | 必填 | 允许值 | 示例 | 来源 |
|------|------|------|--------|------|------|
| schema_version | string | 是 | `"semantic_record_v1"` | `"semantic_record_v1"` | 固定值 |
| sample_id | string | 是 | `{type}_{scenario}_{time}_{seq}` | `"acmi_ai_s01_120.5_001"` | 生成 |

---

## source (数据来源)

| 路径 | 类型 | 必填 | 允许值 | 示例 | 来源 |
|------|------|------|--------|------|------|
| source.platform | string | 是 | afsim_acmi, manual, mixed | `"afsim_acmi"` | 三路均需 |
| source.type | string | 是 | manual, acmi_ai, afsim_rule, mixed | `"acmi_ai"` | 三路均需 |
| source.scenario_id | string/null | 否 | ACMI 文件名 | `"2v2_bvr_001.acmi"` | ACMI |
| source.episode_id | string/null | 否 | 子回合标识 | `"ep01"` | ACMI |
| source.timestamp_sec | number/null | 否 | 仿真时间 (s) | `120.5` | ACMI |
| source.annotator | string/null | 否 | 人名或模型名 | `"qwen3-4b"` | 标注时 |
| source.tool | string/null | 否 | 脚本名 | `"acmi_auto_label_v1.py"` | 标注时 |

---

## obs.ego (己方状态)

| 路径 | 类型 | 必填 | 单位 | 示例 | 来源 |
|------|------|------|------|------|------|
| obs.ego.runtime_id | string/null | 否 | - | `"0xA1"` | ACMI object ID |
| obs.ego.callsign | string/null | 否 | - | `"Viper1"` | ACMI Name/Pilot |
| obs.ego.coalition | enum/null | 否 | - | `"red"` | ACMI Coalition/Color |
| obs.ego.speed_mps | number/null | 否 | m/s | `350.0` | obs[7] 或 ACMI |
| obs.ego.altitude_m | number/null | 否 | m | `8000` | obs[2] 或 ACMI Alt |
| obs.ego.heading_deg | number/null | 否 | deg (0=N,CW+) | `90.5` | obs[6]->deg 或 ACMI Yaw |
| obs.ego.pitch_deg | number/null | 否 | deg | `-1.2` | ACMI 或 asin(vz/v) |
| obs.ego.roll_deg | number/null | 否 | deg | `45` | ACMI 直接; AFSIM=null |
| obs.ego.vertical_speed_mps | number/null | 否 | m/s (正=上升) | `-3.5` | obs[5] |
| obs.ego.pos_east_m | number/null | 否 | m | `25000` | obs[0] |
| obs.ego.pos_north_m | number/null | 否 | m | `40000` | obs[1] |
| obs.ego.energy_state | enum/null | 否 | low/medium/high | `"high"` | E_spec=alt+v²/(2×9.81); low≤q30, med=q30~q70, high>q70; 阈值由数据集配置提供 |

---

## obs.enemy (敌方态势)

| 路径 | 类型 | 必填 | 单位 | 示例 | 来源 |
|------|------|------|------|------|------|
| obs.enemy | object/null | 否 | - | `null` (无敌方信息) | ACMI 配对 |
| obs.enemy.target_ref | enum/null | 否 | - | `"enemy_1"` | 运行时分配 |
| obs.enemy.target_runtime_id | string/null | 否 | - | `"0xB1"` | ACMI object ID |
| obs.enemy.callsign | string/null | 否 | - | `"Eagle1"` | ACMI |
| obs.enemy.range_m | number/null | 否 | m | `45000` | sqrt(dx^2+dy^2+dz^2) |
| obs.enemy.bearing_deg | number/null | 否 | deg (正=右,[-180,180]) | `12.5` | 计算 |
| obs.enemy.aspect_deg | number/null | 否 | deg (0=头对头,180=追尾) | `170` | 计算 |
| obs.enemy.closure_mps | number/null | 否 | m/s (负=接近) | `-250` | d(range)/dt |
| obs.enemy.alt_diff_m | number/null | 否 | m (正=己方高) | `200` | ego_alt - enemy_alt |
| obs.enemy.is_primary_threat | boolean/null | 否 | - | `true` | 规则判定 |

---

## obs.ally (友方态势)

| 路径 | 类型 | 必填 | 单位 | 示例 | 来源 |
|------|------|------|------|------|------|
| obs.ally | object/null | 否 | - | `null` (无友方信息) | ACMI 配对 |
| obs.ally.ally_ref | enum/null | 否 | - | `"ally_1"` | 运行时分配 |
| obs.ally.ally_runtime_id | string/null | 否 | - | `"0xA2"` | ACMI object ID |
| obs.ally.callsign | string/null | 否 | - | `"Viper2"` | ACMI |
| obs.ally.range_m | number/null | 否 | m | `3000` | 计算 |
| obs.ally.bearing_deg | number/null | 否 | deg | `-30` | 计算 |
| obs.ally.is_supporting | boolean/null | 否 | - | `true` | 规则/推断 |

---

## obs.engagement (交战标志)

| 路径 | 类型 | 必填 | 示例 | 来源 |
|------|------|------|------|------|
| obs.engagement | object/null | 否 | | 计算 |
| obs.engagement.is_merge | boolean/null | 否 | `false` | 判定规则见 annotation_guideline_v1.md |
| obs.engagement.is_defensive | boolean/null | 否 | `true` | 判定规则见 annotation_guideline_v1.md |
| obs.engagement.has_shot_opportunity | boolean/null | 否 | `null` | 当前仿真无武器包线, 通常 null |

---

## obs.history (时间上下文)

| 路径 | 类型 | 必填 | 示例 | 来源 |
|------|------|------|------|------|
| obs.history | object/null | 否 | | 运行时维护 |
| obs.history.prev_semantic_state | string/null | 否 | `"commit_intercept"` | 上一步输出 |
| obs.history.prev_token_family | string/null | 否 | `"sustain"` | token_family_catalog |
| obs.history.prev_target_ref | string/null | 否 | `"enemy_1"` | 上一步输出 |

---

## label (标注输出)

| 路径 | 类型 | 必填 | 允许值 | 示例 | 来源 |
|------|------|------|--------|------|------|
| label.semantic_state | enum/null | **是** | 9 个枚举 + null | `"commit_intercept"` | 标注 |
| label.target_ref | enum/null | 否 | enemy_1/enemy_2/ally_1/null | `"enemy_1"` | 标注 |
| label.role | enum/null | 否 | engaged/support/egressing/neutral/null | `"engaged"` | 标注 |
| label.constraints.prefer_energy_preserve | bool/null | 否 | true/false/null | `true` | 标注 |
| label.constraints.allow_aggressive_turn | bool/null | 否 | true/false/null | `false` | 标注 |
| label.constraints.must_support_teammate | bool/null | 否 | true/false/null | `false` | 标注 |
| label.constraints.should_abort_if_threatened | bool/null | 否 | true/false/null | `true` | 标注 |
| label.profile_hint | enum/null | 否 | p6dof_semantic/p6dof_aggressive_turn/auto/null | `"auto"` | 标注 |
| label.event_flags.target_switch | bool/null | 否 | true/false/null | `true` | prev_target vs current |
| label.rationale_short | string/null | 否 | <=120 字符 | `"远距接近中"` | 标注 |

---

## evidence (标注依据)

| 路径 | 类型 | 必填 | 示例 | 来源 |
|------|------|------|------|------|
| evidence.rule_ids | array[string] | 否 | `["R01_range_decreasing"]` | 规则标注 |
| evidence.acmi_refs | array[string] | 否 | `["file.acmi:0xA1:45.0"]` | ACMI 追溯 |
| evidence.frame_refs | array[string] | 否 | `["frame_120.0"]` | AFSIM 帧 |
| evidence.notes | string/null | 否 | `"AI初标+人工确认"` | 自由文本 |

---

## quality (质量评估)

| 路径 | 类型 | 必填 | 允许值 | 示例 | 来源 |
|------|------|------|--------|------|------|
| quality.tier | enum | **是** | gold/silver/bronze/weak | `"silver"` | 评估 |
| quality.confidence | number/null | 否 | [0.0, 1.0] | `0.82` | AI 标注 |
| quality.needs_review | boolean | **是** | true/false | `false` | 评估 |
| quality.conflict_flag | boolean | 否 | true/false | `false` | 多源对比 |

---

## extras (扩展)

| 路径 | 类型 | 必填 | 示例 | 来源 |
|------|------|------|------|------|
| extras | object/null | 否 | | |
| extras.raw_object_ids | object/null | 否 | `{"ego":"0xA1"}` | ACMI |
| extras.raw_tags | array[string]/null | 否 | `["rule_auto"]` | 自由标签 |
| extras.weapon_event_refs | array[string]/null | 否 | `["missile_0x501"]` | acmi2weapon_events |
