# Restricted Label Smoke Train

## 目标

用 6 标签 restricted-label 数据集对 Qwen3.5-4B 做 LoRA smoke train，
验证语义标注管线到 SFT 训练的完整闭环。

**当前是 smoke train，不是正式 9-label 主实验。**

## Restricted Label Set

| 标签 | 纳入 | 预期样本数 |
|------|------|-----------|
| commit_intercept | ✅ | ~53 |
| press_attack | ✅ | ~3 |
| hold_geometry | ✅ | ~28 |
| extend | ✅ | ~17 |
| merge_entry | ✅ | ~11 |
| defensive_break | ✅ | ~6 |
| support | ❌ 排除 | 0 |
| offensive_turn | ❌ 排除 | 0 |
| energy_manage | ❌ 排除 | 0 |

## 数据来源

- `datasets/trainpacks/restricted_nonnull_train.jsonl` — 训练集
- `datasets/trainpacks/restricted_nonnull_dev.jsonl` — 验证集

来源: ACMI v2 mining → D2b 规则标注 + GPT-5.4 hard-case 标注 → merge → restricted export

## null 版本

`datasets/trainpacks/restricted_with_null_review.jsonl` 包含 semantic_state=null 的记录，
仅用于后续审阅型/校准实验，不进入本轮主训练。

## 训练命令

### LLaMA-Factory

```bash
# 安装
pip install llamafactory

# 训练 (从 wow/ 目录下运行)
llamafactory-cli train configs/qwen_smoke_restricted.yaml
```

注意: `model_name_or_path` 需要改成你本地的 Qwen3.5-4B 模型路径。

### 关键超参

| 参数 | 值 | 说明 |
|------|-----|------|
| finetuning_type | lora | 单卡 4090 可跑 |
| lora_rank | 64 | 适中，4B 模型余量充足 |
| epochs | 5 | smoke train 不需要太多 |
| batch_size | 4 × 4 grad_accum = 16 effective | 适配 24GB 显存 |
| lr | 2e-4 | LoRA 标准值 |
| cutoff_len | 2048 | obs JSON 一般 ~500 token |

## 不改的部分

- VQ tokenizer (t4_cb64)
- OneStepTokenBC
- bridge / P6DOF
- runtime obs-sidecar

## 后续

1. smoke train 完成后，检查 dev loss 曲线
2. 人工审查 dev 集预测 vs 真实标签
3. 根据结果决定是否扩展到 9 标签 + 更多数据
