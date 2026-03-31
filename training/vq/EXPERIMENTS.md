# VQ-VAE 实验记录

> 日期：2026-03-21
> 数据集：`datasets/dt2hz_H2s_vqclean.npz`（500,748 windows, action=[dpsi_rad, dalt_sp_m, dspd_sp_mps]）

## 一、问题背景

初始 VQ-VAE 管线使用 stride=1 滑窗从每个 20-step window 中切出 19 个 2-step chunk，
产生 **950 万**训练样本。训练结果：

- active codes: 42 / 128（33%）
- perplexity: 12.0
- top-1 token 占比: 30%（"近似直飞"的静稳动作）
- top-10 占比: 85%

**核心问题**：低机动/直飞 chunk 占比过高 → codebook 大量 code 被浪费在静稳变体上。

## 二、改进措施

在 `chunk_dataset.py` 中引入两项改进（不修改数据集和模型结构）：

### A. Chunk 提取模式

| 模式 | 说明 | 目的 |
|------|------|------|
| `single_fixed` | 每 window 只取固定 offset 处 1 个 chunk | 消除 19× 重复放大 |
| `single_random` | 每 window 随机取 1 个 chunk（seed 可复现） | 消除重复 + 轻微数据增强 |
| `strided_all` | 按步长提取多个 chunk | 保留多 chunk，可控步长 |

### B. Motion-aware 下采样

```
motion_score = mean(|dpsi|) + λ_alt * mean(|dalt|/100) + λ_spd * mean(|dspd|/10)
```

按分位数分三档（static / light / maneuvering），对低机动 chunk 做可控下采样。

## 三、实验结果

### 实验总览

| 编号 | 配置 | token_steps | codebook | epochs | 训练样本 | 训练时间 |
|------|------|:-----------:|:--------:|:------:|:--------:|:--------:|
| E0 | baseline (stride=1, 无下采样) | 2 | 128 | 30 | 9,514,212 | 1950s |
| E1 | single_fixed + motion 下采样 | 2 | 128 | 20 | 225,338 | 29s |
| E2 | single_random, 不下采样 | 2 | 128 | 80 | 500,748 | 251s |
| E3 | single_random, 不下采样 | **4** | 128 | 100 | 500,748 | 326s |
| E4 | single_random, 不下采样 | **4** | **64** | 150 | 500,748 | 479s |

### Codebook 利用率

| 编号 | active codes | 利用率 | perplexity | top-1 | top-2 | top-10 |
|------|:------------:|:------:|:----------:|:-----:|:-----:|:------:|
| E0 | 42 / 128 | 33% | 12.0 | 30.1% | 48.8% | 84.9% |
| E1 | 12 / 128 | 9% | 5.8 | 44.5% | — | 98.6% |
| E2 | 25 / 128 | 20% | 6.8 | 39.0% | 64.4% | 95.3% |
| E3 | 28 / 128 | 22% | 8.5 | 36.6% | 59.9% | 89.3% |
| **E4** | **30 / 64** | **47%** | **8.4** | 38.9% | 60.0% | 90.3% |

### 重建质量（物理单位）

| 编号 | val MSE | dpsi RMSE (rad) | dalt RMSE (m) | dspd RMSE (m/s) |
|------|:-------:|:---------------:|:-------------:|:---------------:|
| E0 | 0.171 | 0.0483 | 46.05 | 6.35 |
| E1 | 0.397 | 0.1055 | 95.33 | 13.93 |
| E2 | 0.231 | 0.0572 | 54.98 | 7.37 |
| E3 | 0.228 | 0.0564 | 55.57 | 7.15 |
| **E4** | **0.220** | **0.0544** | **53.84** | **7.17** |

## 四、关键发现

### 1. stride=1 的重复并非主要问题

E0 的 42 active codes 反而高于去重后的 E1 (12) 和 E2 (25)。
stride=1 的 19× 重叠更像一种数据增强，帮助 codebook 见到更多微小变体。
**结论**：盲目去重不能改善 codebook 利用率。

### 2. Motion 下采样需要足够样本量配合

E1（22.5 万样本 + 20 epochs）表现最差，因为：
- 样本量骤降 42 倍，codebook 还没收敛（VQ loss 5.1 vs E0 的 0.82）
- 训练分布与评估分布严重不匹配

### 3. token_steps=4 带来时间维度语义

t=2（6 维输入）→ t=4（12 维输入）后，尾部 token 开始展现出**有意义的时间演化模式**：
转弯加速、俯冲减速等 4 步动作轨迹比 2 步更能区分不同机动意图。

### 4. 缩小 codebook 提升利用率

codebook 128 → 64 后：
- 利用率从 22% 提升到 **47%**
- 重建质量反而更好（val MSE 0.220 < 0.228）
- 更小的 codebook 更容易收敛

### 5. 静稳主导是数据的固有属性

所有实验中 top-1 token 始终占 30–45%。这反映了真实空战数据中 ~60% 的飞行时间
确实是近似直飞（dpsi≈0, dalt≈0, dspd≈0）。这是数据本身的性质，不是模型缺陷。

## 五、最佳配置（E4）

```bash
python -m training.vq.train_vqvae \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --save_dir checkpoints/vqvae_clean_t4_cb64 \
    --token_steps 4 \
    --codebook_size 64 \
    --epochs 150 \
    --batch 512 \
    --lr 3e-4 \
    --chunk_extract_mode single_random \
    --static_keep_ratio 1.0 \
    --light_keep_ratio 1.0
```

**最佳模型 checkpoint**：`checkpoints/vqvae_clean_t4_cb64/best.pt`

### E4 Token 语义示例

| Token | 占比 | dpsi_rad | dalt_sp_m | dspd_sp_mps | 语义 |
|-------|------|----------|-----------|-------------|------|
| T45 | 38.9% | ≈0 | ≈+5m | ≈+0.5 | 直飞/微爬升（静稳） |
| T25 | 21.1% | ≈0 | ≈-3m | ≈-0.5 | 直飞/微降 |
| T17 | 3.1% | ≈-0.13 | ≈0 | ≈-8 | 右转 + 减速 |
| T35 | 2.1% | ≈0 | ≈-260 | ≈+8 | 急俯冲 + 加速（dive & accelerate） |
| T55 | 1.5% | ≈+0.2 | ≈+150 | ≈-25 | 左转爬升 + 强减速（pull-up turn） |

## 六、后续方向

| 方向 | 预期收益 | 改动量 |
|------|----------|--------|
| codebook=32 | 利用率可能达 80%+，但牺牲分辨率 | 只改参数 |
| EMA codebook + dead code restart | 强制激活死 code，提升 perplexity | 需改 vqvae_model.py |
| token_steps=8 | 更长时间窗口，更丰富的机动语义 | 只改参数，但 24 维输入可能需要加大 MLP |
| 下游 GPT-style policy | 用当前 30-token vocabulary 建序列模型 | 新模块 |

## 七、文件索引

```
checkpoints/
├── vqvae_clean/           # E0 baseline (t=2, cb=128, stride=1)
├── vqvae_clean_dedup/     # E1 (single_fixed + motion 下采样)
├── vqvae_clean_r1/        # E2 (single_random, t=2, cb=128)
├── vqvae_clean_t4/        # E3 (single_random, t=4, cb=128)
└── vqvae_clean_t4_cb64/   # E4 (single_random, t=4, cb=64) ← 最佳

outputs/
├── token_vis_clean/       # E0 可视化
├── token_vis_dedup/       # E1 可视化
├── token_vis_r1/          # E2 可视化
├── token_vis_t4/          # E3 可视化
└── token_vis_t4_cb64/     # E4 可视化
```
