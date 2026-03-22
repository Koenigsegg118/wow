# VQ-VAE Action Tokeniser

将连续的 action 子序列（chunk）编码为离散 codebook token，
为后续 action-token 序列建模（如 GPT-style policy）提供基础。

## 概览

```
action [N, T, 3]
    │  滑动窗口 (stride=1)
    ▼
chunks [M, token_steps, 3]
    │  (可选) per-channel z-score 标准化
    ▼
┌────────────────────────────┐
│ Encoder  flatten → MLP → z │
│ VQ       codebook lookup    │
│ Decoder  z_q → MLP → x̂     │
└────────────────────────────┘
    │
    ▼
重建 loss + commitment loss
（日志同时报告标准化空间 & 原始物理单位误差）
```

## 为什么需要标准化

三个 action 通道的量纲和数值范围差异极大：

| 通道 | 典型值 | 量级 |
|------|--------|------|
| `dpsi_rad` | ±0.01–0.5 | ~10⁻¹ |
| `alt_sp_m` | 200–15000 | ~10⁴ |
| `spd_sp_mps` | 120–650 | ~10² |

不做标准化时，MSE loss 几乎完全由 `alt_sp_m` 主导，`dpsi_rad` 的梯度信号
被淹没。per-channel z-score 标准化使三个通道在损失函数中权重相当，
codebook 能更均匀地编码所有维度的变化。

**默认开启标准化** (`--normalize_action true`)。标准化参数（mean/std）保存在
`action_stats.json` 中，checkpoint 也内嵌了同样的信息，确保推理时可复现。

## 快速开始

```bash
# 从仓库根目录运行（需先 conda activate wow）

# 推荐：标准化训练（默认）
python -m training.vq.train_vqvae \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --save_dir checkpoints/vqvae \
    --token_steps 2 \
    --codebook_size 128 \
    --epochs 30 \
    --batch 512 \
    --lr 3e-4 \
    --seed 42

# 对比实验：关闭标准化
python -m training.vq.train_vqvae \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --save_dir checkpoints/vqvae_raw \
    --normalize_action false
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | (必填) | 训练 NPZ 路径 |
| `--save_dir` | `checkpoints/vqvae` | 输出目录 |
| `--token_steps` | 2 | 每个 chunk 的时间步数（2 = 1秒 @2Hz） |
| `--codebook_size` | 128 | 码本大小 |
| `--latent_dim` | 32 | 潜空间维度 |
| `--epochs` | 30 | 训练轮数 |
| `--batch` | 512 | batch size |
| `--lr` | 3e-4 | 学习率 |
| `--beta` | 0.25 | commitment loss 权重 |
| `--seed` | 42 | 随机种子 |
| `--val_ratio` | 0.1 | 验证集比例 |
| `--normalize_action` | `true` | per-channel z-score 标准化 (`true`/`false`) |

## 输出文件

```
checkpoints/vqvae/
├── best.pt             # 最佳验证 checkpoint（含 action_stats）
├── last.pt             # 最后一轮 checkpoint（含 action_stats）
├── action_stats.json   # 标准化参数（mean/std per channel）
└── metrics.json        # 训练历史 + 逐通道误差
```

### action_stats.json 示例

```json
{
  "normalize": true,
  "fields": ["dpsi_rad", "alt_sp_m", "spd_sp_mps"],
  "mean": {"dpsi_rad": -0.00153, "alt_sp_m": 6525.4, "spd_sp_mps": 361.2},
  "std":  {"dpsi_rad": 0.129,    "alt_sp_m": 4121.1, "spd_sp_mps": 118.3}
}
```

### metrics.json 每轮记录

**标准化空间（训练 loss 空间）：**
- `val_norm_rmse_*` / `val_norm_mae_*`

**原始物理单位（反标准化后）：**
- `val_raw_rmse_dpsi_rad` — 航向变化 RMSE (rad)
- `val_raw_mae_dpsi_rad` — 航向变化 MAE (rad)
- `val_raw_rmse_alt_sp_m` / `val_raw_mae_alt_sp_m` — 高度 (m)
- `val_raw_rmse_spd_sp_mps` / `val_raw_mae_spd_sp_mps` — 速度 (m/s)

## Token 可视化与检查

训练完成后，使用 `visualize_tokens.py` 检查 codebook 质量。

### Action 语义自动识别

`visualize_tokens.py` 自动检测 action 通道的含义，兼容两种数据集：

| 语义类型 | 字段 | 来源 |
|----------|------|------|
| **setpoint** | `[dpsi_rad, alt_sp_m, spd_sp_mps]` | BC 主线数据集 |
| **delta** | `[dpsi_rad, dalt_sp_m, dspd_sp_mps]` | VQ clean 数据集 |

识别优先级：NPZ `meta.action_semantics` > checkpoint `action_stats.fields` > 回退 setpoint。

**animate 模式下两者 rollout 逻辑不同：**
- **setpoint**：速度/高度向设定点一阶逼近（`v += alpha * (sp - v)`）
- **delta**：速度/高度直接加增量（`v += gamma * dspd`，`z += gamma * dalt`）

两种模式都只用于语义可视化，不代表真实飞行动力学。

### summary 模式

输出 token 使用统计、top-k token 的 action profile 图、CSV 汇总：

```bash
python -m training.vq.visualize_tokens \
    --mode summary \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --ckpt checkpoints/vqvae/best.pt \
    --save_dir outputs/token_vis \
    --top_k 10 \
    --max_samples_per_token 50
```

输出文件：
- `token_usage_bar.png` — 所有 token 使用频率柱状图
- `token_summary.csv` — 每个 token 的 count、usage_ratio、逐步 mean/std
- `token_XXX_summary.png` — top-k token 的 action profile（样本曲线 + mean +/- std）
- `token_gallery.png` — top-k token 平均曲线网格总览

### animate 模式

输入指定 token_id，生成该 token 对应的局部 3D 轨迹动画：

```bash
# 平均动作 + overlay + 2D 三视图
python -m training.vq.visualize_tokens \
    --mode animate \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --ckpt checkpoints/vqvae/best.pt \
    --save_dir outputs/token_vis \
    --token_id 12 \
    --anim_source mean \
    --overlay_samples true \
    --num_overlay_samples 5 \
    --show_2d_views true \
    --show_action_panel true

# 查看真实样本并保存 gif
python -m training.vq.visualize_tokens \
    --mode animate \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --ckpt checkpoints/vqvae/best.pt \
    --save_dir outputs/token_vis \
    --token_id 12 \
    --anim_source sample \
    --sample_index 3 \
    --save_path outputs/token12.gif
```

### VQ clean 数据集示例

```bash
# summary（自动识别 delta 语义）
python -m training.vq.visualize_tokens \
    --mode summary \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --ckpt checkpoints/vqvae_clean/best.pt \
    --save_dir outputs/token_vis_clean \
    --top_k 10

# animate（使用 delta rollout）
python -m training.vq.visualize_tokens \
    --mode animate \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --ckpt checkpoints/vqvae_clean/best.pt \
    --save_dir outputs/token_vis_clean \
    --token_id 12 \
    --anim_source mean \
    --overlay_samples true \
    --show_2d_views true
```

支持保存格式：`.png`（静态末帧）、`.gif`、`.mp4`、`.html`。
不提供 `--save_path` 时弹出交互窗口。

### 简化 rollout 模型

animate 模式使用简化运动学将 action chunk 转成局部 3D 轨迹，**仅用于可视化**：

- 初始状态：x=0, y=0, z=5000m, psi=0 (North), v=300 m/s
- 航向：`psi += dpsi_rad`（直接累加）
- 速度：一阶逼近 `v += alpha_v * (spd_sp - v)`，alpha_v=0.3
- 高度：一阶逼近 + 爬升率限幅 160 m/s，alpha_h=0.3
- 位置：`x += v*sin(psi)*dt, y += v*cos(psi)*dt`

**局限**：
- 不含气动约束、转弯过载、能量守恒
- 仅表示 token 的"动作意图方向"，不是真实飞行轨迹
- 适合快速判断 token 语义（左转/右转/爬升/加速等），不适合精确仿真

## VQ Clean Dataset

### 为什么 VQ 需要独立的 clean dataset

现有 BC 数据集 (`dt2hz_H2s_fighteronly.npz`) 的 action 定义是：

```
[dpsi_rad, alt_sp_m, spd_sp_mps]    ← 绝对量
```

这对 BC 回归没问题，但对 VQ tokenisation 有三个核心问题：

1. **分布过散**：`alt_sp_m` 范围 200–15000m，`spd_sp_mps` 范围 120–650 m/s。
   同样的"爬升 200m"动作，在 3000m 和 12000m 处会被编码为完全不同的 token，
   codebook 被工况主导而非动作语义主导。

2. **异常未清除**：现有 builder 检测了 accel spike / speed jump，但只统计不剔除。
   异常步会产生无意义 token（极端 dpsi、瞬移高度变化），污染 codebook。

3. **长实体支配**：单个长 sortie 可贡献数万窗口，如果不做配额控制，
   codebook 会偏向该实体的飞行模式。

### VQ clean dataset 的改进

| 维度 | BC 主线 | VQ clean |
|------|---------|----------|
| action[1] | `alt_sp_m` (绝对高度) | `dalt_sp_m` (高度变化量) |
| action[2] | `spd_sp_mps` (绝对速度) | `dspd_sp_mps` (速度变化量) |
| 异常步 | 检测但不剔除 | 真正剔除（accel + speed jump） |
| 时间 gap | 不处理 | 按 gap 切分 segment，窗口不跨 gap |
| 实体配额 | 无 | `--max_windows_per_entity` |
| 文件配额 | 无 | `--max_windows_per_file` |

### 构建命令

```bash
# 默认构建（使用 tra_data/ 下所有 ACMI）
python -m training.vq.build_vq_clean_dataset \
    --out datasets/dt2hz_H2s_vqclean.npz

# 只保留 F-16/F-15，限制每实体 2000 窗口
python -m training.vq.build_vq_clean_dataset \
    --out datasets/dt2hz_H2s_vqclean_fighter2k.npz \
    --include_regex "(?i)(F-16|F-15)" \
    --max_windows_per_entity 2000

# 调整异常阈值和 gap 切分
python -m training.vq.build_vq_clean_dataset \
    --out datasets/dt2hz_H2s_vqclean_strict.npz \
    --accel_thresh 100.0 \
    --speed_jump 150.0 \
    --gap_thresh 1.5
```

### 输出文件

```
datasets/
├── dt2hz_H2s_vqclean.npz            # 主数据集
├── dt2hz_H2s_vqclean.meta.json      # 构建参数 + 过滤统计
├── dt2hz_H2s_vqclean.stats.json     # 分布统计（各维度 percentiles）
├── dt2hz_H2s_vqclean.filelist.txt   # 使用的文件列表
└── dt2hz_H2s_vqclean.rejected.txt   # 被拒绝的文件
```

### 用 VQ clean dataset 训练

```bash
python -m training.vq.train_vqvae \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --save_dir checkpoints/vqvae_clean \
    --codebook_size 128 --epochs 30
```

注意：`train_vqvae.py` 和 `chunk_dataset.py` 不关心 action 的具体语义，
只需要 `action` shape 为 `[N, T, 3]`。VQ clean dataset 和 BC 数据集
可以无缝替换使用。

### 判断数据是否仍然过散

查看 `stats.json` 中以下指标：

| 指标 | 期望 | 说明 |
|------|------|------|
| `action.dalt_sp_m.std` | < 200 m | 越小 = 高度变化越集中 |
| `action.dspd_sp_mps.std` | < 30 m/s | 越小 = 速度变化越集中 |
| `action.dpsi_rad.p95` | < 0.3 rad | 多数转弯应在小角度 |
| `per_entity_windows.max / total` | < 20% | 无单实体过度支配 |
| `per_entity_windows.p95 / p50` | < 5x | 实体间分布均匀 |

## Reducing duplicate chunks and static dominance

### 问题描述

默认 stride=1 滑窗从每个 20-step window 中切出 19 个 2-step chunk，
产生 950 万级训练样本，其中大量高度重叠的片段被放大。同时低机动/近似直飞
chunk 占比约 50%，导致 codebook 利用率偏低（active codes 42/128，
perplexity 12，top-1 token 占 30%）。

### 解决方案

#### A. Chunk 提取模式 (`--chunk_extract_mode`)

| 模式 | 说明 | 每 window 产出 |
|------|------|:---:|
| `single_fixed`（默认） | 每个 window 只取固定 offset 处的一个 chunk | 1 |
| `single_random` | 每个 window 随机取一个 chunk（seed 可复现） | 1 |
| `strided_all` | 按 `--chunk_stride` 步长提取多个 chunk | ≥1 |

**为什么 `single_fixed` 作为默认值**：VQ clean 数据集的源 window 本身就有
stride=5 滑窗，相邻 window 已有 75% 重叠。再做 stride=1 二次切片会将
重复放大 19 倍。`single_fixed` 从每个 window 取唯一一个 chunk，把样本量
从 950 万降到 50 万，同时保留了完整的动作多样性。

#### B. Motion-aware 下采样

对每个 chunk 计算 motion score：

```
score = mean(|dpsi|) + λ_alt * mean(|dalt|/100) + λ_spd * mean(|dspd|/10)
```

按分位数将 chunk 分为三档：
- **static**（≤ p50）：保留 `--static_keep_ratio`（默认 25%）
- **light**（p50–p85）：保留 `--light_keep_ratio`（默认 50%）
- **maneuvering**（> p85）：全部保留

### 推荐第一轮实验

```bash
# 默认去重 + 低机动下采样（~22.5 万 chunks）
python -m training.vq.train_vqvae \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --save_dir checkpoints/vqvae_clean_dedup \
    --token_steps 2 \
    --codebook_size 128 \
    --epochs 20 \
    --batch 512 \
    --lr 3e-4 \
    --chunk_extract_mode single_fixed \
    --static_keep_ratio 0.25 \
    --light_keep_ratio 0.5
```

### 对照实验（strided_all）

```bash
# 保留多 chunk，但 stride=2 避免极端重叠
python -m training.vq.train_vqvae \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --save_dir checkpoints/vqvae_clean_strided \
    --token_steps 2 \
    --codebook_size 128 \
    --epochs 20 \
    --batch 512 \
    --lr 3e-4 \
    --chunk_extract_mode strided_all \
    --chunk_stride 2 \
    --static_keep_ratio 0.25 \
    --light_keep_ratio 0.5
```

### 新增 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk_extract_mode` | `single_fixed` | 提取模式 |
| `--fixed_offset` | 0 | `single_fixed` 的时间偏移 |
| `--chunk_stride` | `token_steps` | `strided_all` 的步长 |
| `--max_chunks` | 无 | 最终硬上限 |
| `--lambda_alt` | 1.0 | 高度通道的 motion score 权重 |
| `--lambda_spd` | 1.0 | 速度通道的 motion score 权重 |
| `--static_keep_ratio` | 0.25 | 静稳 chunk 保留比例 |
| `--light_keep_ratio` | 0.5 | 轻机动 chunk 保留比例 |
| `--static_q` | 0.50 | 静稳/轻机动分位数边界 |
| `--light_q` | 0.85 | 轻机动/机动分位数边界 |

### 输出文件

训练目录中新增 `sampling_stats.json`，记录完整的采样管线参数和统计。

## Tokenized Dataset for Downstream Policy

### 为什么冻结 tokenizer

经过多轮实验（详见 `EXPERIMENTS.md`），确定最优配置为
`token_steps=4, codebook_size=64`（30 个 active codes, perplexity=8.4）。
冻结 tokenizer 后，将 VQ clean 数据集一次性编码为 token 序列，
供下游 GPT-style action-token policy 使用。

### 为什么用 deterministic non-overlap tokenization

- 下游序列建模需要**稳定、可复现**的 token 序列
- 每个 20-step window 自然切成 5 个不重叠 chunk：`[0:4] [4:8] [8:12] [12:16] [16:20]`
- 不使用随机 offset，确保同一 window 总是产生相同的 5-token 序列

### 为什么要做 active code remap

原始 codebook 有 64 个 code，但只有 30 个被实际使用。
将稀疏的 raw code id（如 4, 6, 9, ...）映射为稠密的 `[0, 29]`，
让下游 embedding 层大小 = 30 而非 64，减少参数浪费。

### 输出数据集 schema

```
datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz
```

| 字段 | dtype | shape | 说明 |
|------|-------|-------|------|
| `obs_full` | float32 | `[N, 20, 8]` | 完整 observation 序列 |
| `obs_tok_start` | float32 | `[N, 5, 8]` | 每个 token chunk 起始时刻的 obs |
| `token_ids` | int64 | `[N, 5]` | 稠密 token id（下游训练用） |
| `token_ids_raw` | int64 | `[N, 5]` | 原始 codebook id（回溯用） |
| `motion_score` | float32 | `[N, 5]` | 每个 chunk 的机动强度 |
| `active_raw_codes` | int64 | `[K]` | 活跃 raw code 列表 |
| `meta` | JSON | — | 元数据（token_steps, vocab_size 等） |

附属文件：
- `.meta.json` — 完整元数据 + top-10 token 统计
- `.vocab.json` — raw ↔ dense 映射表 + 每个 token 的 count/ratio

### 运行命令

```bash
python -m training.vq.tokenize_npz \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --out datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz
```

## Token BC Baseline（下游 next-token 预测）

### 任务定义

给定 obs 上下文和前序 token，预测下一个 action token（30 类分类）。
这是 VQ 管线的首个下游验证：frozen tokenizer → token 序列 → causal Transformer → next-token policy。

与连续 BC（直接预测 [dpsi, dalt, dspd]）不同，Token BC 将控制问题转化为**离散序列建模**：
模型通过 teacher forcing 学习自回归预测 5-token 序列中每个位置的 token。

### 为什么用 active code remap 后再训练

原始 codebook 有 64 个 code，但只有 30 个被实际使用。
remap 后 embedding 层大小 = 30 而非 64，减少参数浪费，
且 dense id 连续 `[0, 29]`，便于 softmax 输出和 class weight 计算。

### 防泄漏数据划分

源数据使用 stride=5 滑窗（T=20），相邻 window 重叠 15 步。
随机 shuffle 会导致训练集/测试集包含来自同一轨迹的重叠 window（数据泄漏）。

**解决方案**：连续 block split + guard bands（默认 100 windows 隔离带）：
```
train: [0, N_train)
guard: [N_train, N_train+100)          ← 丢弃
val:   [N_train+100, N_train+100+N_val)
guard: [N_train+100+N_val, ...)         ← 丢弃
test:  [N_train+200+N_val, N)
```

如果未来 tokenized dataset 中增加 provenance（file_id / entity_id），
应优先按 group split 划分（同一 group 不能跨 split）。

### 为什么要做 Weighted Cross-Entropy

当前 token 分布长尾明显：top-1 静稳 token 占比 ~40%。
如果使用均匀 CE，模型会倾向"永远预测静稳 token"而获得 ~40% accuracy，
但对机动动作的建模能力为零。

Weighted CE 通过提升稀有 token 的 loss 权重来缓解这一问题：
- `inverse_sqrt`（默认）：`w_c = 1/√(count_c)`，均值归一化
- `effective_num`：`w_c = (1-β)/(1-β^n_c)`，β=0.999

### 为什么必须同时看两套指标

| 指标 | 覆盖 | 作用 |
|------|------|------|
| **all-token** | 全部 token | 衡量整体预测准确率，但被静稳 token 主导 |
| **maneuver-subset** | motion_score top 30% | 衡量对战术动作的建模能力，不被静稳稀释 |

如果只看 all-token，一个"永远预测静稳"的模型也能得 ~40%；
加上 maneuver-subset 才能暴露模型是否真正学到了机动意图。

### Majority-token baseline

训练脚本自动统计并报告 majority baseline（永远预测最高频 token），
同时在 all-token 和 maneuver-subset 上分别计算，
并报告模型相对 baseline 的提升（lift）。

### 模型架构

- Causal Transformer（默认 2 层, hidden_dim=128, 4 heads）
- 输入：obs_embed(obs[t]) + tok_embed(tok[t-1]) + pos_embed(t)
  - t=0 时用可学习 `<start>` embedding 替代 tok_embed
- 输出：logits over vocab_size=30
- Teacher forcing 训练，自回归预测

### 运行命令

```bash
python -m training.vq.train_token_bc \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
    --save_dir checkpoints/token_bc_t4_cb64 \
    --epochs 50 \
    --batch 256 \
    --lr 3e-4 \
    --hidden_dim 128 \
    --num_layers 2 \
    --dropout 0.1 \
    --class_weight_mode inverse_sqrt \
    --maneuver_quantile 0.7
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | (必需) | tokenized NPZ 路径 |
| `--save_dir` | `checkpoints/token_bc_t4_cb64` | 输出目录 |
| `--epochs` | 50 | 训练轮数 |
| `--batch` | 256 | batch size |
| `--lr` | 3e-4 | 学习率 |
| `--seed` | 42 | 随机种子 |
| `--hidden_dim` | 128 | Transformer hidden dim |
| `--num_layers` | 2 | Transformer 层数 |
| `--n_heads` | 4 | 注意力头数 |
| `--dropout` | 0.1 | dropout 率 |
| `--class_weight_mode` | `inverse_sqrt` | `{none, inverse_sqrt, effective_num}` |
| `--maneuver_quantile` | 0.7 | 机动子集分位数（0.7 = top 30%） |
| `--guard` | 100 | block split 隔离带大小 |
| `--patience` | 10 | 早停耐心 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `best.pt` | 最佳 val accuracy 模型 |
| `last.pt` | 最后一个 epoch 模型 |
| `metrics.json` | 完整训练指标 + 历史 |
| `eval_all.json` | 全量 test 评估（CE, top-1/3/5, majority baseline, lift）|
| `eval_maneuver.json` | 机动子集 test 评估（CE, top-1/3/5, majority baseline, lift）|
| `class_weights.json` | 每个 token 的 class weight |

## Autoregressive Evaluation and Decode-back Metrics

### 为什么 teacher-forcing 指标不够

Teacher-forcing 评估时，模型在每个位置都能看到**真实的前序 token**。
这相当于"开卷考试"——一旦前序 token 出错，真实部署时会累积误差，
但 teacher-forcing 评估完全看不到这种退化。

### 为什么要做 free-running

Autoregressive（free-running）评估让模型使用**自己预测的 token**作为下一步的输入：
- t=0: 仅看 obs，无前序 token（使用 BOS）
- t=1: 使用 t=0 的**预测结果**（而非真实 token）
- t=2~4: 依次使用前一步的预测结果

这模拟了真实部署场景，能暴露误差累积问题。

### 为什么要 decode 回连续 action

Token 级 accuracy 只衡量"离散 id 是否匹配"，但无法回答：
- 预测错误的 token 在**物理空间**偏差多大？
- 是混淆了两个相似的静稳 token（影响小），还是把机动 token 预测成了静稳（影响大）？

通过冻结的 VQ-VAE decoder 将预测 token 解码回连续 action `[dpsi, dalt, dspd]`，
并与真实 20-step action 序列比较 RMSE/MAE，得到**物理单位的重建误差**。

### 运行命令

```bash
python -m training.vq.eval_token_bc_autoreg \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
    --token_bc_ckpt checkpoints/token_bc_t4_cb64/best.pt \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --save_dir checkpoints/token_bc_t4_cb64_eval \
    --maneuver_quantile 0.7 --batch 256
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `eval_teacher_all.json` | Teacher-forcing 全量评估 |
| `eval_teacher_maneuver.json` | Teacher-forcing 机动子集评估 |
| `eval_autoreg_all.json` | Autoregressive 全量评估 |
| `eval_autoreg_maneuver.json` | Autoregressive 机动子集评估 |
| `decode_teacher_metrics.json` | Teacher-forcing decode-back 连续动作 RMSE/MAE |
| `decode_autoreg_metrics.json` | Autoregressive decode-back 连续动作 RMSE/MAE |
| `per_token_recall.csv` | Top-10 token 的 recall 和主要混淆对象 |
| `confusion_top10.csv` | Top-10 token 的混淆矩阵 |

## One-step Receding-Horizon Token Policy

### 为什么 5-token 序列模型不适合直接部署

现有 Token BC baseline（`train_token_bc.py`）在每个位置 t 使用 `obs_tok_start[:, t, :]`，
其中 t=1~4 对应的 obs 分别在仿真的第 4/8/12/16 步——这些是**未来时刻的真值观测**。
在真实 AFSIM 部署中，t=0 时刻只能看到当前 obs，无法预知 4 步后的状态。

### 为什么 one-step 更接近真实部署

One-step token policy 模拟 receding-horizon 控制循环：
1. 每 4 个仿真步（2 秒）调用一次策略
2. 输入：当前 obs + 上一步预测的 token
3. 输出：一个 token → VQ-VAE decode → 4 步连续动作
4. 执行 4 步后，用新的 obs 重复

这不依赖未来观测，计算量更小，天然适合实时控制。

### One-step 数据集 schema

从 5-token 窗口数据展开：每个窗口的 5 个 token 位置各生成 1 个训练样本。
**Block split 在窗口级别进行（展开之前）**，防止同一轨迹的 token 跨 split 泄漏。

```
datasets/dt2hz_H2s_vqclean_t4_cb64_onestep.npz
```

| 字段 | dtype | shape | 说明 |
|------|-------|-------|------|
| `obs_hist` | float32 | `[M, H, 8]` | 观测历史（H=1 时只有当前 obs）|
| `prev_token` | int64 | `[M]` | 前一个 token（t=0 时为 BOS=30）|
| `target_token` | int64 | `[M]` | 目标 token |
| `motion_score` | float32 | `[M]` | 运动强度 |
| `window_idx` | int64 | `[M]` | 源窗口编号 |
| `token_pos` | int64 | `[M]` | 窗口内位置 (0~4) |
| `split_label` | int64 | `[M]` | 0=train, 1=val, 2=test |
| `obs_mean/obs_std` | float32 | `[8]` | 从 train split 计算 |

### 运行命令

```bash
# 1. 构建 one-step 数据集
python -m training.vq.build_onestep_token_dataset \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
    --out datasets/dt2hz_H2s_vqclean_t4_cb64_onestep.npz \
    --obs_hist_len 1 --include_prev_token true

# 2. 训练 one-step token policy
python -m training.vq.train_onestep_token_bc \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_onestep.npz \
    --save_dir checkpoints/onestep_token_bc_t4_cb64 \
    --epochs 50 --batch 512 --lr 3e-4 \
    --hidden_dim 128 --dropout 0.1 \
    --class_weight_mode inverse_sqrt \
    --maneuver_quantile 0.7 \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt
```

## 设计假设与限制

1. **标准化参数来源**：从全部 chunk（训练+验证）统计 mean/std，而非仅训练集。
   对于 90/10 划分，差异可忽略；如需严格隔离，后续可改为仅用训练集统计。
2. **Chunk 提取**：默认 `single_fixed` 每个源 window 只取一个 chunk，
   避免 stride=1 造成的极端重复。可通过 `strided_all` 恢复多 chunk 模式。
3. **MLP 架构**：Encoder/Decoder 各两层全连接。chunk 尺寸小
   （token_steps=2 → 6维输入），MLP 足够。
4. **无 EMA 更新**：codebook 使用梯度更新（非 EMA），适合小 codebook。
5. **不修改现有管线**：完全独立于 BC 训练和 AFSIM bridge。

---

## One-step Replay 与 Bridge 评估

### 目的

在完整 5-token 窗口上回放 one-step 策略，对比 **oracle-prev-token**（使用真实上一步
token）和 **self-fed-prev-token**（使用模型自身预测作为上一步输入）两种模式，
量化误差累积（drift），并通过 VQ-VAE 解码回连续动作评估部署可行性。

### 两种回放模式

| 模式 | prev_token 来源 | 对应场景 |
|------|----------------|----------|
| Oracle | 真实 ground-truth token | 上界估计（无累积误差）|
| Self-fed | 模型自身上一步预测 | 真实部署（含累积误差）|

### 运行

```bash
python -m training.vq.replay_onestep_token_policy \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
    --policy_ckpt checkpoints/onestep_token_bc_t4_cb64/best.pt \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --save_dir checkpoints/onestep_token_bc_t4_cb64_replay \
    --maneuver_quantile 0.7 --batch 512 \
    --num_vis_samples 5 --export_bridge_rollout true
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `replay_oracle.json` | oracle 模式 token 指标 |
| `replay_selffed.json` | self-fed 模式 token 指标 |
| `decode_oracle.json` | oracle 解码连续动作 RMSE/MAE |
| `decode_selffed.json` | self-fed 解码连续动作 RMSE/MAE |
| `drift_summary.json` | 两模式差异摘要 |
| `per_token_recall_oracle.csv` | oracle 每 token 召回率 |
| `per_token_recall_selffed.csv` | self-fed 每 token 召回率 |
| `vis/*.png` | 样本可视化（GT vs oracle vs self-fed）|
| `bridge_rollout.npz` | AFSIM 桥接用离线 rollout 数据 |

### 关键指标（当前结果）

|  | Oracle | Self-fed | Drift |
|--|--------|----------|-------|
| ALL top-1 | 78.5% | 65.0% | +13.5% |
| MAN top-1 | 60.0% | 40.2% | +19.8% |
| dpsi RMSE (rad) | 0.108 | 0.172 | +0.064 |
| dalt RMSE (m) | 54.9 | 57.1 | +2.3 |
| dspd RMSE (m/s) | 11.7 | 15.7 | +3.9 |

### Per-position 误差累积

Self-fed 模式下各 token 位置准确率逐步下降：

```
pos0=67.1%  pos1=65.6%  pos2=64.7%  pos3=64.0%  pos4=63.5%
```

pos0 与 oracle 相同（无 prev_token 输入），pos1–4 因使用自身预测的 prev_token
而逐步退化，但退化斜率较缓（每步约 -1%）。

### Bridge Rollout 格式

`bridge_rollout.npz` 包含：
- `sample_ids [Nt]`：test 窗口索引
- `predicted_tokens [Nt, 5]`：self-fed 预测 token 序列
- `decoded_action_chunks [Nt, 5, 4, 3]`：解码后连续动作（5 token × 4 step × 3 dim）

可直接作为 AFSIM 离线桥接的输入数据源。

---

## Current-step Ego-centric History Input for One-step Policy

### 为什么 current-step ego frame 对 one-step 更合理

之前的 one-step 模型使用单帧全局坐标 obs，存在两个问题：
1. **绝对位置无意义**：不同 ACMI 文件的 x_e/y_n 范围差几百公里，但飞机执行相同机动时动作应完全一致
2. **训练/部署不对齐**：训练时 window-anchor ego transform 以窗口起点为锚，但实际部署时每步只有当前状态，无法访问窗口起点

Current-step ego transform 以**当前决策时刻**为锚点，与 runtime 环境完全一致。

### 为什么不能只用单帧 current-step ego obs

如果只用当前时刻一帧做 ego transform：
- dx'=0, dy'=0, heading'=0 → 位置和航向信息完全塌缩
- 只剩 z_u, vx', vy', vz_u, speed（实际上 vx'≈0, vy'≈speed）
- 模型无法区分"直飞" vs "正在转弯中"

### 为什么需要 history_len >= 2

引入短历史（默认 H=4 个 token boundary 时刻，约 8 秒）后：
- 历史点在 ego 坐标系中形成有意义的轨迹（转弯 → 弧线，直飞 → 直线）
- 模型可以从位置/航向变化率推断当前运动趋势
- H=4 在 token boundary 粒度（每 2 秒一个点）覆盖 6 秒历史，足以捕捉转弯

### 本轮冻结范围

- Tokenizer（t4_cb64）冻结不动
- 只重跑 one-step 相关链路：dataset builder → training → replay eval
- Tokenized dataset `dt2hz_H2s_vqclean_t4_cb64_tok.npz` 的 `obs_full` 保留全局坐标，支持按需构造任意 ego frame

### 运行

```bash
# 构建新数据集
python -m training.vq.build_onestep_token_dataset \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
    --out datasets/dt2hz_H2s_vqclean_t4_cb64_onestep_h4.npz \
    --history_len 4

# 训练
python -m training.vq.train_onestep_token_bc \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_onestep_h4.npz \
    --save_dir checkpoints/onestep_token_bc_t4_cb64_h4 \
    --epochs 50 --batch 512 --lr 3e-4 \
    --hidden_dim 128 --dropout 0.1 \
    --class_weight_mode inverse_sqrt \
    --maneuver_quantile 0.7 \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt

# Replay 评估
python -m training.vq.replay_onestep_token_policy \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
    --policy_ckpt checkpoints/onestep_token_bc_t4_cb64_h4/best.pt \
    --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --save_dir checkpoints/onestep_token_bc_t4_cb64_h4_replay \
    --maneuver_quantile 0.7 --batch 512
```

### 共享工具

`training/vq/ego_obs_utils.py` 包含：
- `ego_transform_current_anchor(obs_hist)` — current-step 锚定 ego 变换
- `extract_token_boundary_history(obs_full, token_pos, token_steps, history_len)` — 从窗口中提取 token boundary 历史

### Runtime 接入契约

详见 `training/vq/runtime_obs_contract.txt`，供 Cursor 实现 AFSIM bridge 时参考。
