# SFT 导出规范 v1 (v1.1 patched)

> 本文档定义如何将规范记录 (`semantic_record_v1`) 导出为 SFT 训练数据。
> 对应 schema: `semantic_record_schema_v1.json`，枚举注册表: `enum_registry_v1.json`。

> **v1.1 补丁**: AI 标注输出 `label` + `quality_hint` 两个 key, adapter 负责将 `quality_hint` 搬入 canonical `quality` 字段。`quality.tier` 由规则派生, 不由 LLM 直接决定。样本粒度=单一决策时刻。canonical 内部用 null, 不用 "unknown" 字符串。

---

## 1. 导出目标

本规范解决的核心问题：**如何把已标注的空战态势记录转换为可直接用于大语言模型监督微调 (SFT) 的训练样本。**

两个目标格式：

| 格式 | 用途 | 输入包含 evidence | 输出包含 rationale_short |
|------|------|:-----------------:|:----------------------:|
| **标准 SFT** | 训练模型从观测推断战术意图 | 否 | 否 |
| **审阅型 SFT** | 训练模型同时利用证据链进行审阅判断 | 是 | 是 |

设计原则：

- 模型**只看到它在推理时能获得的信息**（obs），不泄漏 source / quality / extras 等元数据。
- null 字段一律省略，降低序列长度并避免模型学会生成无意义 null。
- quality tier 控制哪些样本参与训练，不进入模型输入输出。

---

## 2. 标准 SFT 格式

```json
{
  "instruction": "<instruction 模板，见第 8 节>",
  "input": {
    "ego": { "...非 null 的 ego 字段..." },
    "enemy": { "...非 null 的 enemy 字段（整体非 null 时）..." },
    "ally": { "...非 null 的 ally 字段（整体非 null 时）..." },
    "engagement": { "...非 null 的 engagement 字段（整体非 null 时）..." },
    "history": { "...非 null 的 history 字段..." }
  },
  "output": {
    "semantic_state": "commit_intercept",
    "target_ref": "enemy_1",
    "role": "engaged",
    "constraints": { "...非 null 约束字段..." },
    "profile_hint": "p6dof_semantic",
    "event_flags": { "...非 null 事件标志..." }
  }
}
```

**input 来源：** `obs` 的全部 5 个子节（ego / enemy / ally / engagement / history），按第 4/5 节规则过滤。

**output 来源：** `label` 中除 `rationale_short` 以外的全部字段。

---

## 3. 审阅型 SFT 格式

```json
{
  "instruction": "<审阅型 instruction 模板，见第 8 节>",
  "input": {
    "obs": {
      "ego": { "..." },
      "enemy": { "..." },
      "ally": { "..." },
      "engagement": { "..." },
      "history": { "..." }
    },
    "evidence": {
      "rule_ids": ["R01_range_decreasing", "R03_closure_negative"],
      "acmi_refs": ["scenario_001.acmi:0x1A2B:120.5"],
      "frame_refs": ["frame_120.5"],
      "notes": "距离持续缩小且接近率为负"
    }
  },
  "output": {
    "semantic_state": "commit_intercept",
    "target_ref": "enemy_1",
    "role": "engaged",
    "constraints": { "..." },
    "profile_hint": "p6dof_semantic",
    "event_flags": { "..." },
    "rationale_short": "距离缩小+接近率为负，己方主动向 enemy_1 建立截获态势"
  }
}
```

**与标准 SFT 的区别：**

1. `input` 额外包含 `evidence` 节。
2. `input.obs` 与标准 SFT 的 `input` 内容相同，但多嵌套一层 `obs` key，以与 `evidence` 平级。
3. `output` 额外包含 `rationale_short`。
4. `conflict_flag=true` 的记录**仅进入审阅型 SFT**（标准 SFT 排除），并在 evidence.notes 中附加冲突说明。

---

## 4. 字段映射规则

### 4.1 input 字段映射（obs -> input）

| 源路径 | 标准 SFT input | 审阅型 SFT input.obs | 规则 |
|--------|:-:|:-:|------|
| `obs.ego` | **始终包含** | **始终包含** | 保留所有非 null 字段 |
| `obs.enemy` | 整体非 null 时包含 | 整体非 null 时包含 | 整体为 null 时省略该 key |
| `obs.ally` | 整体非 null 时包含 | 整体非 null 时包含 | 整体为 null 时省略该 key |
| `obs.engagement` | 整体非 null 时包含 | 整体非 null 时包含 | 整体为 null 时省略该 key |
| `obs.history` | **始终包含** | **始终包含** | 即使全部子字段为 null，仍保留空对象 `{}` |

### 4.2 output 字段映射（label -> output）

| 源路径 | 标准 SFT output | 审阅型 SFT output | 规则 |
|--------|:-:|:-:|------|
| `label.semantic_state` | **必填** | **必填** | 不可省略；为 null 的记录整条排除 |
| `label.target_ref` | 非 null 时包含 | 非 null 时包含 | |
| `label.role` | 非 null 时包含 | 非 null 时包含 | |
| `label.constraints` | 非 null 时包含 | 非 null 时包含 | 内部各子字段也按 null 省略 |
| `label.profile_hint` | 非 null 时包含 | 非 null 时包含 | |
| `label.event_flags` | 非 null 时包含 | 非 null 时包含 | 内部各子字段也按 null 省略 |
| `label.rationale_short` | **不包含** | 非 null 时包含 | 仅审阅型 SFT |

### 4.3 evidence 映射

| 源路径 | 标准 SFT | 审阅型 SFT | 规则 |
|--------|:-:|:-:|------|
| `evidence` | **不包含** | **包含在 input.evidence** | 空数组保留为 `[]`，null notes 省略 |

### 4.4 永不进入模型输入/输出的字段

以下字段**严格禁止**出现在 SFT 样本的 input 或 output 中：

- `schema_version` — 内部版本控制
- `sample_id` — 内部追踪 ID
- `source` 整节 — 数据来源元信息（防止模型学到来源偏差）
- `quality` 整节 — 质量评估（仅用于过滤，不参与训练）
- `extras` 整节 — 扩展字段（原始 ID / 武器事件等非核心信息）

---

## 5. null 处理规则

| 场景 | 处理方式 |
|------|----------|
| input 中某字段值为 null | **省略该字段**，不序列化 `"field": null` |
| output 中某字段值为 null | **省略该字段**，模型不应学会生成 null 值字段 |
| `obs.enemy` 整体为 null | 省略 `enemy` key |
| `obs.ally` 整体为 null | 省略 `ally` key |
| `obs.engagement` 整体为 null | 省略 `engagement` key |
| `obs.history` 所有子字段为 null | 保留 `"history": {}`（空对象），不省略 key |
| `label.constraints` 整体为 null | 省略 `constraints` key |
| `label.constraints` 内部子字段为 null | 省略对应子字段；若所有子字段均为 null，省略整个 `constraints` |
| `label.event_flags` 处理逻辑 | 同上 |
| `label.semantic_state` 为 null | **整条记录排除**，不导出为 SFT 样本 |

---

## 6. quality tier 过滤规则

| quality.tier | conflict_flag | 标准 SFT | 审阅型 SFT | 说明 |
|:-------------|:-------------|:-:|:-:|------|
| gold | false | 导出 | 导出 | 最高质量，全量参与 |
| gold | true | N/A | N/A | gold 定义要求 conflict_flag=false; 此组合不应存在 |
| silver | false | 导出 | 导出 | 高质量，全量参与 |
| silver | true | 排除 | 导出 + 冲突信息 | 同上 |
| bronze | false | 导出 + 元数据标记 | 导出 + 元数据标记 | 可用于 curriculum learning，导出时附加 `_meta.tier = "bronze"` |
| bronze | true | 排除 | 导出 + 冲突信息 + 元数据标记 | |
| weak | — | **排除** | **排除** | 质量不足，不参与任何 SFT |

**bronze 元数据标记**：在导出的 JSONL 中，bronze 样本额外携带顶层 `_meta` 字段：

```json
{
  "instruction": "...",
  "input": { "..." },
  "output": { "..." },
  "_meta": { "tier": "bronze", "confidence": 0.65 }
}
```

`_meta` 不参与模型训练，仅供训练脚本实现 curriculum learning 策略时读取（如前 N epoch 仅用 gold + silver，后续混入 bronze）。

**冲突信息附加**：当 `conflict_flag=true` 的记录进入审阅型 SFT 时，在 `evidence.notes` 末尾追加 `[CONFLICT]` 标记，如：

```
"notes": "原始备注内容 [CONFLICT]"
```

---

## 7. 样例

### 7.1 标准 SFT 样本 — commit_intercept（含敌方数据）

```json
{
  "instruction": "根据以下空战态势信息，判断己方飞机当前的战术意图。请输出 JSON 格式的标签，包含 semantic_state（战术状态）、target_ref（目标引用）、role（角色）、constraints（行为约束）、profile_hint（飞行剖面建议）和 event_flags（事件标志）。仅输出你有把握判断的字段。",
  "input": {
    "ego": {
      "callsign": "Red-1",
      "coalition": "red",
      "speed_mps": 280.5,
      "altitude_m": 8500,
      "heading_deg": 45.2,
      "pitch_deg": -2.1,
      "vertical_speed_mps": -5.3,
      "energy_state": "high"
    },
    "enemy": {
      "target_ref": "enemy_1",
      "range_m": 35000,
      "bearing_deg": 12.5,
      "aspect_deg": 170.3,
      "closure_mps": -185.0,
      "alt_diff_m": 200,
      "is_primary_threat": true
    },
    "ally": {
      "ally_ref": "ally_1",
      "range_m": 4500,
      "bearing_deg": -30.0,
      "is_supporting": true
    },
    "engagement": {
      "is_merge": false,
      "is_defensive": false
    },
    "history": {
      "prev_semantic_state": "commit_intercept",
      "prev_target_ref": "enemy_1"
    }
  },
  "output": {
    "semantic_state": "commit_intercept",
    "target_ref": "enemy_1",
    "role": "engaged",
    "constraints": {
      "prefer_energy_preserve": false,
      "allow_aggressive_turn": false,
      "must_support_teammate": false
    },
    "profile_hint": "p6dof_semantic",
    "event_flags": {
      "target_switch": false
    }
  }
}
```

### 7.2 标准 SFT 样本 — energy_manage（仅己方数据）

```json
{
  "instruction": "根据以下空战态势信息，判断己方飞机当前的战术意图。请输出 JSON 格式的标签，包含 semantic_state（战术状态）、target_ref（目标引用）、role（角色）、constraints（行为约束）、profile_hint（飞行剖面建议）和 event_flags（事件标志）。仅输出你有把握判断的字段。",
  "input": {
    "ego": {
      "callsign": "Red-2",
      "coalition": "red",
      "speed_mps": 155.0,
      "altitude_m": 3200,
      "heading_deg": 270.0,
      "vertical_speed_mps": 12.5,
      "energy_state": "low"
    },
    "history": {
      "prev_semantic_state": "defensive_break",
      "prev_token_family": "break_turn"
    }
  },
  "output": {
    "semantic_state": "energy_manage",
    "role": "neutral",
    "constraints": {
      "prefer_energy_preserve": true
    },
    "profile_hint": "auto"
  }
}
```

**说明：** 此样本中 enemy / ally / engagement 整体为 null，因此 input 中不包含这些 key。output 中 target_ref / event_flags 为 null，同样省略。

### 7.3 审阅型 SFT 样本 — defensive_break（含证据链）

```json
{
  "instruction": "请审阅以下空战态势观测和标注证据，综合判断己方飞机当前的战术意图。请输出 JSON 格式的标签，包含 semantic_state（战术状态）、target_ref（目标引用）、role（角色）、constraints（行为约束）、profile_hint（飞行剖面建议）、event_flags（事件标志）和 rationale_short（一句话判断理由）。仅输出你有把握判断的字段。",
  "input": {
    "obs": {
      "ego": {
        "callsign": "Blue-1",
        "coalition": "blue",
        "speed_mps": 310.0,
        "altitude_m": 6800,
        "heading_deg": 180.5,
        "pitch_deg": -15.2,
        "roll_deg": 72.0,
        "vertical_speed_mps": -45.0,
        "energy_state": "medium"
      },
      "enemy": {
        "target_ref": "enemy_1",
        "range_m": 8200,
        "bearing_deg": 165.0,
        "aspect_deg": 25.0,
        "closure_mps": -120.0,
        "alt_diff_m": -500,
        "is_primary_threat": true
      },
      "engagement": {
        "is_merge": false,
        "is_defensive": true
      },
      "history": {
        "prev_semantic_state": "hold_geometry",
        "prev_target_ref": "enemy_1"
      }
    },
    "evidence": {
      "rule_ids": ["R05_high_bank", "R07_defensive_aspect", "R09_nose_away"],
      "acmi_refs": ["scenario_003.acmi:0xA1B2:245.0"],
      "frame_refs": ["frame_245.0"],
      "notes": "横滚角72度+大下俯角, 敌方展示角25度(接近尾追), 己方正在急转脱离"
    }
  },
  "output": {
    "semantic_state": "defensive_break",
    "target_ref": "enemy_1",
    "role": "engaged",
    "constraints": {
      "prefer_energy_preserve": false,
      "allow_aggressive_turn": true,
      "should_abort_if_threatened": true
    },
    "profile_hint": "p6dof_aggressive_turn",
    "event_flags": {
      "target_switch": false
    },
    "rationale_short": "大bank角急转+敌方低展示角尾追, 判定为防御性急转脱离"
  }
}
```

---

## 8. instruction 模板

### 8.1 标准 SFT instruction

```
根据以下空战态势信息，判断己方飞机当前的战术意图。请输出 JSON 格式的标签，包含 semantic_state（战术状态）、target_ref（目标引用）、role（角色）、constraints（行为约束）、profile_hint（飞行剖面建议）和 event_flags（事件标志）。仅输出你有把握判断的字段。
```

### 8.2 审阅型 SFT instruction

```
请审阅以下空战态势观测和标注证据，综合判断己方飞机当前的战术意图。请输出 JSON 格式的标签，包含 semantic_state（战术状态）、target_ref（目标引用）、role（角色）、constraints（行为约束）、profile_hint（飞行剖面建议）、event_flags（事件标志）和 rationale_short（一句话判断理由）。仅输出你有把握判断的字段。
```

**模板使用规则：**

- instruction 为固定文本，不做样本级别的动态拼接。
- 模型在训练时学到 "仅输出你有把握判断的字段" 这一约束，从而与 null 省略规则对齐。
- 两个模板的唯一差异：审阅型多出 "标注证据" 输入描述和 "rationale_short" 输出要求。

---

## 9. 导出脚本约定

### 9.1 脚本接口

```
python export_sft.py \
  --input   canonical_records.jsonl \
  --output  sft_samples.jsonl \
  --format  standard | review \
  --tiers   gold,silver,bronze \
  --exclude-conflict     # 仅对 standard 有效; review 模式下忽略此 flag \
  --bronze-meta          # 为 bronze 样本附加 _meta 字段 \
  --validate             # 导出前对每条记录做 schema 校验
```

### 9.2 参数说明

| 参数 | 必填 | 说明 |
|------|:----:|------|
| `--input` | 是 | 输入文件路径，每行一条 `semantic_record_v1` 的 JSONL |
| `--output` | 是 | 输出文件路径，每行一条 SFT 样本的 JSONL |
| `--format` | 是 | `standard`（标准 SFT）或 `review`（审阅型 SFT） |
| `--tiers` | 否 | 逗号分隔的 tier 列表，默认 `gold,silver`；`weak` 始终排除 |
| `--exclude-conflict` | 否 | 排除 `conflict_flag=true` 的记录（standard 模式默认开启） |
| `--bronze-meta` | 否 | 为 bronze 样本在输出中附加 `_meta` 字段 |
| `--validate` | 否 | 逐条校验输入是否符合 `semantic_record_schema_v1.json` |

### 9.3 处理流程

```
读取 JSONL
  │
  ├─ schema 校验（若 --validate）
  │    └─ 校验失败 → 写入 {output}.rejected.jsonl + 原因
  │
  ├─ semantic_state == null → 跳过
  │
  ├─ quality.tier 过滤
  │    ├─ weak → 跳过
  │    ├─ bronze → 保留，标记 _meta（若 --bronze-meta）
  │    └─ gold / silver → 保留
  │
  ├─ conflict_flag 过滤
  │    ├─ standard + conflict_flag=true → 跳过
  │    └─ review + conflict_flag=true → 保留，notes 追加 [CONFLICT]
  │
  ├─ 构建 input（按第 4/5 节规则从 obs 提取）
  │
  ├─ 构建 output（按第 4/5 节规则从 label 提取）
  │
  ├─ 填充 instruction（按第 8 节模板）
  │
  └─ 写入 JSONL
```

### 9.4 输出统计

脚本执行完毕后，应在 stderr 输出统计摘要：

```
[export_sft] 输入记录: 12500
[export_sft] 跳过 (semantic_state=null): 340
[export_sft] 跳过 (tier=weak): 1280
[export_sft] 跳过 (conflict, standard): 85
[export_sft] schema 校验失败: 12
[export_sft] 导出样本: 10783
[export_sft]   gold: 2150, silver: 5830, bronze: 2803
[export_sft] 输出文件: sft_samples.jsonl
```

### 9.5 幂等性与可复现性

- 输入顺序 = 输出顺序，不做 shuffle（shuffle 留给训练脚本）。
- 同一输入执行多次，输出完全一致。
- `sample_id` 不写入 SFT 样本，但可通过 `--output-mapping mapping.jsonl` 生成行号到 sample_id 的映射文件，用于事后溯源。
