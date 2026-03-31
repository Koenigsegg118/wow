# V1.1 一致性补丁摘要

最小改动的一致性补丁，修复 semantic_data_contract_v1 内部文档/schema/prompt/样例之间的结构性不一致。不扩展需求，不改项目主线，不新增标签。

---

## 修复的 8 类问题

### PATCH-1: confidence 路径统一

| 规则 | 说明 |
|------|------|
| AI 输出格式 | `{ "label": {...}, "quality_hint": { "confidence", "conflict_flag", "needs_review" } }` |
| canonical 落表 | `quality.confidence` <- `quality_hint.confidence`; `quality.tier` 由规则派生, 不由 LLM 决定 |
| label 中不含 confidence | label 只包含标注内容, 质量评估在 quality 节 |

**修改文件**: schema, io_examples, prompt, sft_export_spec, README

### PATCH-2: enemy/ally 统一为单对象

| 规则 | 说明 |
|------|------|
| canonical v1.1 | `obs.enemy: object|null`, `obs.ally: object|null` |
| 不是数组 | 只保留主敌/主友机, 次要目标放 extras |
| 选择逻辑 | 主敌=最近或最高威胁; 主友机=同阵营最近 |

**修改文件**: schema, template, prompt, io_examples

### PATCH-3: canonical 内部统一 null

| 规则 | 说明 |
|------|------|
| 严禁 | 字符串 `"unknown"` / `"unavailable"` / `"N/A"` |
| 统一用 | `null` |
| 解析层 | 如产生 "unknown", 必须在 adapter 中转为 null |

**修改文件**: schema, enum_registry, template, prompt, README

### PATCH-4: energy_state 单一定义

| 规则 | 说明 |
|------|------|
| 公式 | `E_spec = altitude_m + speed_mps^2 / (2 * 9.81)` |
| 分桶 | low: <=q30, medium: q30~q70, high: >q70 |
| 阈值来源 | 数据集预处理配置 (不在 schema 中硬编码数值) |
| 可为 null | speed 或 altitude 缺失时 |

**修改文件**: schema, enum_registry, template

### PATCH-5: engagement 阈值唯一来源

| 规则 | 说明 |
|------|------|
| 唯一规范来源 | `annotation_guideline_v1.md` |
| schema/template | 只保留字段语义, 不写具体数值阈值 |
| prompt | 引用 guideline, 不自创阈值 |
| mapping | 只写计算公式, 不自创判定标准 |

**修改文件**: schema, template, prompt

### PATCH-6: 样本粒度冻结为单一决策时刻

| 规则 | 说明 |
|------|------|
| canonical record = 一个 timestamp | 不是窗口/序列 |
| 上下文来源 | obs.history + evidence (前一时刻) |
| prompt | 输入=单一时刻 obs |

**修改文件**: schema, prompt, checklist, README

### PATCH-7: conflict_flag / quality.tier 统一规则

| 规则 | 说明 |
|------|------|
| conflict_flag=true | needs_review 强制为 true |
| conflict_flag=true 时 | tier 上限 = **silver** |
| conflict_flag=true + 未审查 | 推荐 tier = bronze |
| gold 定义 | conflict_flag 必须为 false |

**修改文件**: schema, enum_registry, checklist, sft_export_spec

### PATCH-8: schema 封口

| 规则 | 说明 |
|------|------|
| additionalProperties: false | 应用于: 顶层, source, obs, ego, enemy, ally, engagement, history, label, constraints, event_flags, evidence, quality |
| extras 例外 | extras 对象本身允许 additionalProperties, 因为它是扩展容器 |

**修改文件**: schema

---

## 唯一真相源清单

| 内容 | 唯一真相源 |
|------|-----------|
| 所有枚举值 | `enum_registry_v1.json` |
| 字段结构和类型 | `semantic_record_schema_v1.json` |
| engagement 判定阈值 | `annotation_guideline_v1.md` |
| 标签定义和边界 | `annotation_guideline_v1.md` |
| quality.tier 派生规则 | `enum_registry_v1.json` (quality.tier.rules) |
| AI 输出格式 | `acmi_auto_label_io_examples.json` (format_C) |
| SFT 导出规则 | `sft_export_spec_v1.md` |
| ACMI 字段映射公式 | `acmi_field_mapping_v1.md` |

---

## 保留为"可配置但 v1.1 不固定"的内容

| 项目 | 状态 | 说明 |
|------|------|------|
| energy_state 的 q30/q70 数值阈值 | 不固定 | 由数据集预处理配置提供 |
| is_merge / is_defensive 的具体距离/角度阈值 | 推荐值在 guideline 中 | 允许根据场景调整 |
| AI 标注模型选择 | 不固定 | annotator 字段记录所用模型 |
| weapon_event 详细解析规则 | 不固定 | 当前仿真无武器包线, 留待实弹场景 |

---

## 修改文件清单

| 文件 | 修改摘要 |
|------|---------|
| `semantic_record_schema_v1.json` | $id 改为 v1.1; additionalProperties:false; energy_state 公式; engagement 阈值引用 guideline; null 规则; confidence 在 quality |
| `enum_registry_v1.json` | v1.1 版本; energy_state computation; quality.tier rules; null_semantics 说明 |
| `acmi_auto_label_io_examples.json` | AI 输出改为 label + quality_hint 分离; adapter_mapping 说明 |
| `acmi_annotation_prompt_v1.txt` | 单一时刻; label+quality_hint 输出; null 不用 unknown; enemy/ally 单对象; 阈值引用 guideline |
| `semantic_record_template_readable.md` | v1.1 头注; energy_state 公式; engagement 阈值引用 |
| `contract_diff_checklist_v1.md` | v1.1 头注; conflict_flag tier 上限修正为 silver |
| `README.md` | v1.1 头注; 文件清单加 PATCH_SUMMARY; AI 标注工作流加 quality_hint 说明; 版本历史 |
| `sft_export_spec_v1.md` | v1.1 头注; gold+conflict 不可共存; quality_hint adapter 说明 |
| `canonical_record_example_set.jsonl` | 无需修改 (已符合 v1.1) |
| `V1_1_PATCH_SUMMARY.md` | 新建 (本文件) |

---

## 人工复核重点 (5 条)

1. **annotation_guideline_v1.md 中的 engagement 阈值** — 确认 is_merge / is_defensive 的推荐距离/角度值是否符合项目实际场景
2. **energy_state 的 q30/q70 分位数** — 需要从实际数据集跑一次统计确定具体数值, 当前 schema 只定义了公式
3. **conflict_flag=true 时 tier 上限=silver** — 检查已有标注数据中是否有违反此规则的记录
4. **extras 保持 additionalProperties 开放** — 确认此设计不会导致 schema 验证遗漏关键字段
5. **AI 输出 quality_hint.confidence 的校准** — 当前定义了 [0,1] 范围和低置信度规则 (null label 时 <0.3), 但实际模型输出需要做校准实验确认
