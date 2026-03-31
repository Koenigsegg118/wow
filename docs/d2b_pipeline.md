# D2b: 小规模语义数据生成管线

## 概述

D2b 基于 D2 已完成的 runtime obs-sidecar，生成一小批语义标注数据，供后续 Qwen3.5-9B SFT 使用。

**不改 baseline 主链路**：VQ tokenizer / OneStepTokenBC / bridge / P6DOF 全部冻结。

## 数据流

```
AFSIM 640-value frames
         |
         v
  build_obs_sidecar()          [已有, D2 完成]
         |
         v obs-sidecar dict
         |
    +----+----------------------------+
    |                                 |
    v                                 v
 Module A                        Module A
 export_obs_samples.py           adapter_obs_to_formatA.py
    |                                 |
    v obs_samples.jsonl               v obs_formatA_requests.jsonl
    |
    +----+----------------------------+
    |                                 |
    v                                 v
 Module B                        Module C
 rule_labeler_v1.py              build_hardcase_requests.py
 (4 easy + null)                 (10 hard-case types)
    |                                 |
    v rule_labeled.jsonl              v hardcase_requests.jsonl
    |                                 |
    |                            GPT-5.4 (或 stub)
    |                                 |
    |                                 v gpt_responses.jsonl
    |                                 |
    |                            Module D
    |                            validate_gpt_output.py
    |                            adapter_ai_to_canonical.py
    |                                 |
    |                                 v canonical_from_gpt.jsonl
    |                                 |
    +----+----------------------------+
         |
         v
    Module E
    build_preference_cases.py    (规则 vs GPT 冲突)
    adapter_preference_results.py
         |
         v preference_canonical.jsonl
         |
         v
    merge_and_export.py
    (preference > gpt > rule)
         |
    +----+----+
    |         |
    v         v
 sft_standard.jsonl   sft_review.jsonl
```

## 模块说明

### Module A: Obs 采样与 format_A 转换

| 文件 | 功能 |
|------|------|
| `export_obs_samples.py` | 从 640 帧缓冲或已有 JSONL 采样 obs |
| `adapter_obs_to_formatA.py` | obs → AI 标注最小输入 |

### Module B: 规则标注 (高精度优先)

`rule_labeler_v1.py` 只自动标 5 种结果:

| 标签 | 条件 | 置信度 |
|------|------|--------|
| `merge_entry` | is_merge=True 或 range<5km+closure<-100 | 0.90-0.95 |
| `commit_intercept` | 正前方接敌, 远距, 非 merge/defensive | 0.90 |
| `extend` | 背对目标, closure>0, 拉开中 | 0.85-0.90 |
| `hold_geometry` | 低 closure, 非 merge/defensive (最保守) | 0.85 |
| `null` | 无敌方 或 信息不足 | — |

**不自动标**: press_attack, support, offensive_turn, defensive_break, energy_manage

### Module C: Hard-case 请求构建

`build_hardcase_requests.py` 检测 10 种 hard-case 来源并打包给 GPT-5.4。

### Module D: GPT 输出校验与适配

| 文件 | 功能 |
|------|------|
| `validate_gpt_output.py` | V01-V11 合规校验 |
| `adapter_ai_to_canonical.py` | format_C → 完整 canonical record |

Tier 派生规则:
- confidence >= 0.80 → silver
- 0.50 <= confidence < 0.80 → bronze
- confidence < 0.50 → weak
- conflict_flag=True → tier 上限 silver

### Module E: 偏好判断

只处理高价值冲突: 规则 vs GPT 不一致、低置信度歧义样本。

### 合并与导出

`merge_and_export.py` 三路合并 (preference > gpt > rule) 并导出:
- `sft_standard.jsonl`: 无 evidence/quality, 排除 conflict
- `sft_review.jsonl`: 含 evidence, 含 conflict

## 配置管理

```
sim/runtime/semantic_thresholds.py   ← engagement 阈值 (不改)
     ↑ import
tools/d2b/d2b_config.py              ← D2b 规则阈值 + GPT 配置
     ↑ import
tools/d2b/*.py                       ← 业务代码 (零硬编码)
```

## GPT-5.4 调用

- System prompt: `prompts/d2b_gpt_system_prompt.txt`
- User template: `prompts/d2b_gpt_user_template.txt`
- API key: 环境变量 `D2B_GPT_API_KEY` 或配置文件
- Stub 可用于管道端到端测试（无需 API）

```python
from tools.d2b.gpt_caller_stub import GPTCallerStub, GPTCallerOpenAI

# 测试模式
caller = GPTCallerStub()

# 生产模式
caller = GPTCallerOpenAI(base_url="https://api.openai.com/v1")
```

## 运行命令

```bash
# 1. 采样 obs
python -m tools.d2b.export_obs_samples --source jsonl \
    --input datasets/d2b/raw_obs.jsonl \
    --output datasets/d2b/obs_samples.jsonl

# 2. 转 format_A
python -m tools.d2b.adapter_obs_to_formatA \
    --input datasets/d2b/obs_samples.jsonl \
    --output datasets/d2b/obs_formatA_requests.jsonl

# 3. 规则标注
python -m tools.d2b.rule_labeler_v1 \
    datasets/d2b/obs_samples.jsonl \
    --output datasets/d2b/rule_labeled.jsonl

# 4. 构建 hard-case 请求
python -m tools.d2b.build_hardcase_requests \
    --obs datasets/d2b/obs_samples.jsonl \
    --rule datasets/d2b/rule_labeled.jsonl \
    --output datasets/d2b/hardcase_requests.jsonl

# 5. GPT 标注 (填入 API key 后)
# export D2B_GPT_API_KEY=sk-...
# python -m tools.d2b.gpt_caller ...

# 6. 校验 GPT 输出
python -m tools.d2b.validate_gpt_output \
    --responses datasets/d2b/gpt_responses.jsonl \
    --output datasets/d2b/gpt_validated.jsonl

# 7. 转 canonical
python -m tools.d2b.adapter_ai_to_canonical \
    --validated datasets/d2b/gpt_validated.jsonl \
    --obs datasets/d2b/obs_samples.jsonl \
    --output datasets/d2b/canonical_from_gpt.jsonl

# 8. 合并 + 导出 SFT
python -m tools.d2b.merge_and_export \
    --rule datasets/d2b/rule_labeled.jsonl \
    --gpt datasets/d2b/canonical_from_gpt.jsonl \
    --output datasets/d2b/merged_canonical.jsonl

# 9. 报告
python -m tools.d2b.report \
    --merged datasets/d2b/merged_canonical.jsonl \
    --outdir reports/
```

## 当前仍为 null 的字段

| 字段 | 原因 | 影响 |
|------|------|------|
| ego.roll_deg | 640 帧不含 roll | 无法判断 offensive_turn/defensive_break |
| ego.energy_state | q30/q70 阈值未配置 | energy_manage 无法规则判断 |
| engagement.has_shot_opportunity | 无武器包线数据 | 不影响标签判断 |
| history.prev_semantic_state | 语义层未接入 | target_switch 检测受限 |
| history.prev_token_family | 同上 | — |
| ally.is_supporting | 无法从单帧判断 | support 标签依赖 GPT |
