# Semantic Data Contract V1 (v1.1 patched)

空战语义标注数据规范。定义了从 ACMI 回放到 AI 自动标注到 SFT 训练的完整数据契约。

> **v1.1 补丁要点**: confidence 路径统一 (AI 输出 `quality_hint`, adapter 搬入 `quality`); enemy/ally 固定为单对象; canonical 内部用 null 不用 "unknown"; energy_state 有计算公式; engagement 阈值唯一来源=guideline; 样本粒度=单一时刻; conflict_flag=true -> tier 上限=silver; schema 已设 additionalProperties:false。

---

## 用途

本目录是项目内部的**唯一语义标注数据规范**。所有标注来源 (人工 / AI / 规则) 最终都归一到此格式。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `semantic_record_schema_v1.json` | **Canonical schema** - JSON Schema 正式定义, 所有记录的结构基准 |
| `enum_registry_v1.json` | **枚举注册表** - 所有枚举值的唯一定义, 新增枚举必须在此注册 |
| `acmi_field_mapping_v1.md` | **ACMI 字段映射** - ACMI 回放字段到 canonical record 的映射公式和规则 |
| `annotation_guideline_v1.md` | **标注规范** - 9 个 semantic_state + role + constraints 等的定义和边界 |
| `sft_export_spec_v1.md` | **SFT 导出规范** - canonical record 到 SFT 训练样本的转换规则 |
| `acmi_auto_label_io_examples.json` | **AI 标注 IO** - LLM 自动标注的输入/输出格式和校验规则 |
| `acmi_annotation_prompt_v1.txt` | **标注 prompt** - 给 AI 标注代理的标准提示词 |
| `canonical_record_example_set.jsonl` | **样例记录** - 12 条覆盖各来源和标签的示例 |
| `contract_diff_checklist_v1.md` | **对齐检查清单** - 三路数据 (manual/acmi_ai/afsim_rule) 一致性检查表 |
| `semantic_record_template_readable.md` | **字段对照表** - 人工/脚本快速查阅用的字段速查表 |
| `V1_1_PATCH_SUMMARY.md` | **v1.1 补丁摘要** - 8 项一致性修复的清单和规则确认 |

---

## 文件关系

```
enum_registry_v1.json  <-- 所有枚举的唯一来源
        |
        v
semantic_record_schema_v1.json  <-- 结构定义 (引用枚举)
        |
        +---> acmi_field_mapping_v1.md     (ACMI 怎么填这个 schema)
        +---> annotation_guideline_v1.md   (标注员怎么标这个 schema)
        +---> sft_export_spec_v1.md        (怎么从 schema 导出 SFT)
        +---> acmi_auto_label_io_examples.json  (AI 怎么读写这个 schema)
        +---> acmi_annotation_prompt_v1.txt     (AI 标注的 system prompt)
        |
        v
canonical_record_example_set.jsonl   (合法样例, 供验证)
contract_diff_checklist_v1.md        (三路对齐检查)
semantic_record_template_readable.md (速查表)
```

---

## 推荐工作流

### 新增标注来源时

1. 阅读 `annotation_guideline_v1.md` 理解标签定义
2. 查阅 `enum_registry_v1.json` 确认枚举值
3. 参考 `canonical_record_example_set.jsonl` 中对应 source.type 的样例
4. 输出格式严格遵循 `semantic_record_schema_v1.json`
5. 用 `contract_diff_checklist_v1.md` 验证一致性

### AI 自动标注时

1. 构造输入: 参考 `acmi_auto_label_io_examples.json` 的 format_A 或 format_B
2. 使用 `acmi_annotation_prompt_v1.txt` 作为 system prompt
3. 校验输出: 按 `format_D_validation_rules` 做自动检查
4. 组装 canonical record: AI 输出的 `label` 填入 canonical `label`, `quality_hint` 搬入 canonical `quality` (tier 由规则派生, 不由 LLM 决定)

### 导出 SFT 数据时

1. 按 `sft_export_spec_v1.md` 的过滤规则筛选 quality.tier
2. 按标准 SFT 或审阅 SFT 格式导出
3. null 字段在导出时省略 (不序列化为 null)

### ACMI 数据接入时

1. 按 `acmi_field_mapping_v1.md` 的公式计算几何字段
2. 使用 `sim/tools/acmi2tspi.py` 解析轨迹
3. 使用 `sim/tools/acmi2weapon_events.py` 提取武器事件
4. 将武器事件 ID 填入 `extras.weapon_event_refs`

---

## 约束

- **不修改低层 baseline**: VQ tokenizer / OneStepTokenBC / P6DOF bridge 不在此契约范围内
- **内部使用 null**: 不使用字符串 "unknown" / "unavailable" / "N/A"
- **单位统一**: 距离 m, 速度 m/s, 角度 deg, 时间 s
- **枚举唯一来源**: 新增枚举必须先在 `enum_registry_v1.json` 注册
- **不跨平台**: 本契约仅服务当前项目 AFSIM/ACMI 场景, 不设计通用接口

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1 | 2026-03-24 | 初版, 9 个 semantic_state, 4 个 quality tier |
| v1.1 | 2026-03-25 | 一致性补丁: confidence 路径统一, enemy/ally 单对象, null 不用 "unknown", energy_state 公式, engagement 阈值唯一来源, 单一时刻粒度, conflict_flag/tier 统一规则, schema 封口 additionalProperties:false |
