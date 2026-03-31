# Token 行为克隆训练管线

本文档详细描述项目的离散化行为克隆 (Tokenized Behavioral Cloning) 训练管线，
涵盖从原始 ACMI 轨迹数据到可部署策略模型的完整流程。

---

## 1. 总体架构

```
ACMI 战术轨迹 (Tacview)
        │
        ▼
┌─────────────────────────────────────┐
│  阶段 1: 数据预处理                    │
│  build_vq_clean_dataset.py          │
│  原始轨迹 → 清洁 obs/action NPZ      │
└───────────────┬─────────────────────┘
                │  dt2hz_H2s_vqclean.npz
                ▼
┌─────────────────────────────────────┐
│  阶段 2: VQ-VAE 训练                  │
│  train_vqvae.py                     │
│  连续动作块 → 离散 codebook           │
└───────────────┬─────────────────────┘
                │  checkpoints/vqvae/best.pt
                ▼
┌─────────────────────────────────────┐
│  阶段 3: Token 化                     │
│  tokenize_npz.py                    │
│  冻结 VQ-VAE 编码全部数据 → token 序列  │
└───────────────┬─────────────────────┘
                │  *_tok.npz + vocab.json
                ▼
┌─────────────────────────────────────┐
│  阶段 4: Token BC 策略训练             │
│  train_token_bc.py (序列)            │
│  train_onestep_token_bc.py (单步)    │
│  因果 Transformer / MLP 学习预测       │
│  下一个动作 token                     │
└───────────────┬─────────────────────┘
                │  checkpoints/*/best.pt
                ▼
┌─────────────────────────────────────┐
│  阶段 5: 运行时推理                    │
│  token_policy_runtime.py            │
│  OneStepTokenBC + VQ decoder        │
│  → AFSIM 实时控制                    │
└─────────────────────────────────────┘
```

---

## 2. 阶段 1: 数据预处理

**脚本**: `training/vq/build_vq_clean_dataset.py`

### 2.1 数据源

- 输入: ACMI/Tacview 格式的空战轨迹文件
- 默认输入目录: `tra_data/`
- 递归搜索所有 `.acmi` 文件

### 2.2 时间与窗口参数

所有参数统一定义在 `ml/dataset_default_config.py` (单一真理来源):

| 参数 | 值 | 含义 |
|------|-----|------|
| `DT` | 0.5 s | 决策率 2 Hz |
| `H_SEC` | 2.0 s | 设定点前瞻时间 |
| `K` | 4 步 | 前瞻步数 (`H_SEC / DT`) |
| `SEQ_LEN` | 20 步 | 每个训练窗口的长度 (= 10 秒) |
| `STRIDE` | 5 步 | 窗口滑动步长 (50% 重叠) |

### 2.3 实体过滤

1. **类型过滤**: 仅保留 `Air+FixedWing` 类型的实体
2. **名称正则排除**: 排除 A-50, E-3, AWACS, 加油机, 轰炸机等非战斗实体
3. **轨道飞行器检测**: 基于行为启发式 (在满足最低步数 600 步 = 300 秒后):
   - 高度标准差 < 80 m
   - 速度标准差 < 7 m/s
   - |dpsi| 第 95 百分位 < 0.05 rad
   - 全部满足则判定为轨道飞行器 (AWACS 巡逻等)，予以排除

### 2.4 物理过滤与异常值清除

| 过滤条件 | 阈值 | 处理方式 |
|----------|------|---------|
| 速度下限 | 60 m/s | 排除低能量样本 (滑行) |
| 速度上限 | 800 m/s | 排除损坏轨迹 (硬截断) |
| 加速度峰值 | 150 m/s² | 移除异常步 (非标记) |
| 速度跳跃 | 200 m/s | 标记跳跃两侧步，移除 |

异常值处理策略: 不仅标记，而是真正移除异常步。移除后只保留连续有效段中长度 >= `SEQ_LEN` 的片段。

### 2.5 时间间隙感知分割

- 当实体时间轴中出现 `gap > 2s` 的断裂时，在断裂处将实体拆分为多个子段
- 防止训练窗口跨越时间不连续区域
- 每个子段独立做窗口化

### 2.6 观测与动作定义

**观测向量** (8 维):

| 索引 | 字段 | 含义 |
|------|------|------|
| 0 | `x_e_m` | 东向位置 (m) |
| 1 | `y_n_m` | 北向位置 (m) |
| 2 | `z_u_m` | 垂直高度 (m) |
| 3 | `vx_e_mps` | 东向速度 (m/s) |
| 4 | `vy_n_mps` | 北向速度 (m/s) |
| 5 | `vz_u_mps` | 垂直速度 (m/s) |
| 6 | `track_angle_rad_unwrapped` | 航迹角 (rad), atan2(vx_east, vy_north), 0=北, 顺时针正 |
| 7 | `ground_speed_mps` | 地面速度 (m/s), sqrt(vx^2 + vy^2) |

**动作向量** (3 维, 相对增量):

| 索引 | 字段 | 含义 |
|------|------|------|
| 0 | `dpsi_rad` | 航向变化量 (rad), 顺时针为正 |
| 1 | `dalt_sp_m` | 高度设定点变化量 (m) |
| 2 | `dspd_sp_mps` | 速度设定点变化量 (m/s) |

动作语义: `action[t] = state[t + K] - state[t]`，即 2 秒前瞻的状态增量。

### 2.7 输出

```
datasets/dt2hz_H2s_vqclean.npz
  obs:       [N, 20, 8]   float32
  action:    [N, 20, 3]   float32
  obs_mean:  [8]          float32
  obs_std:   [8]          float32
  act_mean:  [3]          float32
  act_std:   [3]          float32
  meta:      JSON 元数据
```

典型规模: ~50 万个窗口, 来自 82 个 ACMI 文件, 983 个战斗机实体。

---

## 3. 阶段 2: VQ-VAE 训练

**脚本**: `training/vq/train_vqvae.py`
**模型**: `training/vq/vqvae_model.py`

### 3.1 目标

将连续动作空间离散化为有限的"行为 token"码本，使下游策略学习转化为分类问题。

### 3.2 VQ-VAE 模型架构

```
ActionChunkVQVAE:

输入: [B, token_steps, 3]  (动作块)
       |
       v flatten -> [B, token_steps * 3]
+------------------+
| Encoder          |
| Linear(12, 128)  |
| ReLU             |
| Linear(128, 32)  |
+--------+---------+
         | z: [B, 32]
         v
+------------------+
| VectorQuantizer  |
| codebook: 64x32  |
| 直通梯度估计       |
+--------+---------+
         | z_q: [B, 32]
         v
+------------------+
| Decoder          |
| Linear(32, 128)  |
| ReLU             |
| Linear(128, 12)  |
+--------+---------+
         |
         v reshape -> [B, token_steps, 3]
输出: x_hat (重建动作块)
```

### 3.3 向量量化器 (VectorQuantizer)

- **码本大小**: 64 个码字
- **潜在维度**: 32
- **距离计算**: L2 距离找最近码字
- **梯度传播**: straight-through estimator -- `z_q = z + (z_q - z).detach()`
- **损失函数**:
  ```
  vq_loss = MSE(z_q, z.detach())           # 码本损失: 让码字靠近编码器输出
           + 0.25 * MSE(z_q.detach(), z)    # 承诺损失: 让编码器输出靠近码字
  ```
- 不使用 EMA 更新，采用显式梯度更新码本

### 3.4 动作块提取

每个训练窗口 (20 步) 中按 `token_steps` 分割为多个动作块:

| 模式 | 说明 |
|------|------|
| `single_fixed` | 每个窗口取固定偏移处的 1 个块 |
| `single_random` | 每个窗口取随机偏移处的 1 个块 |
| `strided_all` | 每个窗口按 stride 提取多个块 (数据增强) |

### 3.5 动作感知采样

为防止静态 token (飞行器维持巡航) 过度主导码本，对动作块按运动强度分桶采样:

**运动强度评分**:
```
motion_score = mean(|dpsi|) + lambda_alt * mean(|dalt|/100) + lambda_spd * mean(|dspd|/10)
```

**三个桶及保留比例**:

| 桶 | 条件 | 保留比例 |
|----|------|---------|
| 静态 (static) | score <= Q50 | 25% |
| 轻微 (light) | Q50 < score <= Q85 | 50% |
| 机动 (maneuvering) | score > Q85 | 100% |

### 3.6 训练参数

| 参数 | 值 |
|------|-----|
| `token_steps` | 4 (每 token 覆盖 2 秒) |
| `codebook_size` | 64 |
| `latent_dim` | 32 |
| `beta` (承诺损失权重) | 0.25 |
| `epochs` | 30 |
| `batch_size` | 512 |
| `learning_rate` | 3e-4 (Adam) |
| `normalize_action` | True (z-score per channel) |
| `val_ratio` | 0.1 |

总损失: `L = MSE(x, x_hat) + vq_loss`

### 3.7 评估指标

- **归一化空间**: 重建 RMSE / MAE (训练损失空间)
- **原始物理空间** (反归一化后):
  - `dpsi_rad` RMSE / MAE
  - `dalt_sp_m` RMSE / MAE (米)
  - `dspd_sp_mps` RMSE / MAE (米/秒)

### 3.8 输出

```
checkpoints/vqvae/
  best.pt            # 最优验证损失检查点
  last.pt            # 最后 epoch 检查点
  metrics.json       # 训练历史
  action_stats.json  # 动作归一化统计
  sampling_stats.json
```

---

## 4. 阶段 3: Token 化

**脚本**: `training/vq/tokenize_npz.py`

### 4.1 流程

1. 加载冻结的 VQ-VAE (best.pt)
2. 将数据集每个窗口 [20, 3] 的动作序列按 `token_steps` 分割为不重叠的块:
   - 20 步 / token_steps=4 = 5 个 token/窗口
3. 用 VQ-VAE 编码所有块，得到原始 codebook 索引
4. **活跃码字重映射**: `raw_id -> dense_id`
   - 64 个码字中通常仅 30 个被实际使用
   - 映射为连续的 [0, K-1] 密集 ID, K 为活跃码数
5. 提取 token 边界处的观测: `obs[t=0], obs[t=4], obs[t=8], ...`
6. 计算每个 token 的运动强度评分

### 4.2 输出

```
datasets/*_tok.npz
  obs_full:        [N, 20, 8]   完整观测序列
  obs_tok_start:   [N, 5, 8]    token 边界处的观测
  token_ids_raw:   [N, 5]       原始 codebook 索引
  token_ids:       [N, 5]       密集映射后的 token ID
  motion_score:    [N, 5]       每个 token 的运动强度
  active_raw_codes: [K]         活跃码字列表
  meta:            JSON 元数据 (含映射表)

*_tok.vocab.json                dense <-> raw 双向映射
*_tok.meta.json                 完整元数据
```

---

## 5. 阶段 4: Token BC 策略训练

提供两种模型, 分别用于序列建模和实时部署:

### 5.1 TokenBCTransformer (序列模型)

**脚本**: `training/vq/train_token_bc.py`

#### 模型架构

```
TokenBCTransformer (因果 Transformer):

位置 t 的输入: obs_embed(obs[t]) + tok_embed(tok[t-1]) + pos_embed(t)
                                    ^ t=0 时使用可学习的 <start> 嵌入

  obs_proj:     Linear(8, 128)
  tok_embed:    Embedding(V, 128)
  start_embed:  Parameter(128)       # 可学习 <start> 向量
  pos_embed:    Embedding(5, 128)

  transformer:  2 层 TransformerEncoder
                4 头注意力
                FFN = 512
                Pre-Norm, Dropout=0.1
                因果掩码 (上三角遮蔽)

  head:         Linear(128, V)

输出: [B, T, V] logits, 教师强制训练
```

#### 数据划分

采用**顺序块分割** (Block Split) 防止窗口泄漏:

```
[------ train: 80% ------][guard: 100 窗][-- val: 10% --][guard: 100 窗][-- test: 10% --]
```

Guard band = 100 窗 (500 步 = 250 秒 @2Hz), 确保无重叠窗口跨越分割边界。

#### 类别不平衡处理

静态 token (飞行器巡航) 占比极高, 使用加权交叉熵:

| 模式 | 公式 |
|------|------|
| `none` | 均匀权重 w=1 |
| `inverse_sqrt` | w_c = 1/sqrt(count_c), 归一化使 mean=1 |
| `effective_num` | w_c = (1-beta)/(1-beta^n_c), beta=0.999 |

默认使用 `inverse_sqrt`。

#### 训练参数

| 参数 | 值 |
|------|-----|
| `epochs` | 50 |
| `batch_size` | 256 |
| `learning_rate` | 3e-4 (AdamW) |
| `hidden_dim` | 128 |
| `num_layers` | 2 |
| `n_heads` | 4 |
| `dropout` | 0.1 |
| `patience` | 10 (early stopping) |
| `scheduler` | CosineAnnealingLR |
| `maneuver_quantile` | 0.7 (top 30% 为机动子集) |

#### 评估指标

- **全 token**: Top-1 / Top-3 / Top-5 准确率
- **机动子集**: motion_score > 第 70 百分位的 token 的准确率
- **逐位置准确率**: 各 token 位置 (0~4) 的独立准确率
- **多数类基线**: 始终预测最频繁 token 的准确率 (作为对比)
- **Lift**: 模型准确率 - 多数类基线

---

### 5.2 OneStepTokenBC (单步部署模型)

**脚本**: `training/vq/train_onestep_token_bc.py`

#### 设计目标

为实时 AFSIM 控制环路设计的**单步策略**:
- 输入: 当前观测历史 (H 个 token 边界步) + 前一个 token
- 输出: 下一个 token
- 部署循环: predict token -> VQ 解码为动作块 -> 执行 -> 重复

#### 模型架构

```
OneStepTokenBC (MLP):

obs_hist [B, H, 8]
    | flatten -> [B, H*8]
    v
  obs_encoder:
    Linear(H*8, 128) + ReLU + Dropout(0.1)
    -> [B, 128]

prev_token [B]
    v
  tok_embed:
    Embedding(V+1, 128)    # +1 用于 <start> 哨兵
    -> [B, 128]

    concat -> [B, 256]
    v
  head (融合 MLP):
    Linear(256, 128) + ReLU + Dropout(0.1)
    Linear(128, V)
    -> logits [B, V]
```

#### Ego-Centric 观测变换

使用 `ego_obs_utils.py` 进行观测预处理 (训练和推理共用):

**变换步骤** (锚点为历史窗口的最后一个时间步):

1. **位置相对化**: dx = x - x_anchor, dy = y - y_anchor
2. **旋转对齐**: 以当前航向为基准旋转位置和速度, 使当前 heading -> 0
   ```
   dx' =  cos(-h0) * dx + sin(-h0) * dy
   dy' = -sin(-h0) * dx + cos(-h0) * dy
   vx' =  cos(-h0) * vx + sin(-h0) * vy
   vy' = -sin(-h0) * vx + cos(-h0) * vy
   ```
3. **航向相对化**: heading' = heading - heading_anchor
4. **不变项**: 高度 z_u, 垂直速度 vz_u, 速度标量 speed

**输出观测布局**: `[dx', dy', z_u, vx', vy', vz_u, heading', speed]`

#### 训练参数

| 参数 | 值 |
|------|-----|
| `epochs` | 50 |
| `batch_size` | 512 |
| `learning_rate` | 3e-4 |
| `hidden_dim` | 128 |
| `num_layers` | 2 |
| `dropout` | 0.1 |
| `obs_hist_len (H)` | 4 (4 个 token 边界) |
| `class_weight_mode` | inverse_sqrt |
| `maneuver_quantile` | 0.7 |

---

## 6. 阶段 5: 运行时推理

**脚本**: `sim/token_policy_runtime.py`

### 6.1 推理流程

```
obs_hist [H, 8] (原始观测)
    |
    v ego_transform_current_anchor()
    |  位置/速度/航向相对化 + 旋转
    |
    v 归一化: (obs - obs_mean) / obs_std
    |
    v OneStepTokenBC(obs_norm, prev_token)
    |  -> logits [V]
    |
    v dense_token = argmax(logits)
    |  confidence = softmax(logits).max()
    |
    v raw_token = dense_to_raw[dense_token]
    |
    v z_q = vqvae.vq.embedding[raw_token]
    |
    v action_chunk = vqvae.decoder(z_q)
    |  -> [token_steps, 3]
    |
    v 反归一化: chunk * act_std + act_mean
    |
    v 输出: (dense_token, action_chunk[token_steps, 3], confidence)
```

### 6.2 部署循环

```python
prev_token = runtime.bos_token  # 初始: vocab_size 作为 <start>

while simulation_running:
    obs_hist = collect_obs_at_token_boundaries(H=4)  # [4, 8]

    dense_token, chunk, conf = runtime.predict_and_decode(obs_hist, prev_token)
    # chunk: [token_steps, 3] = [dpsi_rad, dalt_sp_m, dspd_sp_mps]

    apply_action_to_afsim(chunk[0])  # 执行第一步 (receding-horizon)
    prev_token = dense_token
```

### 6.3 关键设计

- **无状态推理**: 每次调用独立, 不维护内部缓存
- **Ego 变换**: 提供位置/方向不变性, 无需全局坐标系
- **Receding Horizon**: 仅执行 chunk 的第一步, 下一步重新预测
- **置信度输出**: softmax 最大概率, 可用于决策质量监控
- **合约版本**: `onestep_h4_current_anchor_ego_v1`

---

## 7. 方向约定

与 AFSIM 仿真引擎对齐的坐标和符号约定:

| 项目 | 约定 |
|------|------|
| 坐标系 | ENU (东-北-天) |
| 航向零点 | 正北 |
| 航向正方向 | 顺时针 |
| 航向计算 | `atan2(vx_east, vy_north)` |
| `dpsi_sign` | +1 (顺时针转 -> 正 dpsi) |
| AFSIM 指令 | dpsi(rad) -> 转换为度 -> `TurnToRelativeHeading` |

---

## 8. 训练产物清单

完成全管线后, 可部署所需的文件:

```
checkpoints/
  vqvae_clean_t4_cb64/
    best.pt              # VQ-VAE 编码器 + 解码器
    action_stats.json    # 动作归一化统计 (mean/std)
    metrics.json         # 训练日志

  token_bc_t4_cb64/
    best.pt              # TokenBCTransformer (序列, 教师强制)
    eval_all.json        # 全 token 测试集评估
    eval_maneuver.json   # 机动子集评估
    metrics.json         # 训练日志 + 多数类基线

  onestep_token_bc_t4_cb64_h4/
    best.pt              # OneStepTokenBC (部署用)
    metrics.json

datasets/
  *_tok.npz              # Token 化数据集
  *_tok.vocab.json       # Dense <-> Raw 映射 (部署必需)
  *_tok.meta.json        # Token 化元数据
```

**部署最小文件集**:
- `checkpoints/vqvae_clean_t4_cb64/best.pt` -- VQ 解码器
- `checkpoints/onestep_token_bc_t4_cb64_h4/best.pt` -- 策略模型
- `*_tok.vocab.json` -- Token 映射表

---

## 9. 完整训练命令参考

```bash
# 1. 数据预处理
python -m training.vq.build_vq_clean_dataset \
    --in tra_data \
    --out datasets/dt2hz_H2s_vqclean.npz

# 2. VQ-VAE 训练 (冻结口径: t4_cb64)
python -m training.vq.train_vqvae \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --save_dir checkpoints/vqvae_clean_t4_cb64 \
    --token_steps 4 --codebook_size 64 --latent_dim 32 \
    --epochs 30 --batch 512 --lr 3e-4 \
    --normalize_action True \
    --chunk_extract_mode single_fixed \
    --static_keep_ratio 0.25 --light_keep_ratio 0.5

# 3. Token 化
python -m training.vq.tokenize_npz \
    --data datasets/dt2hz_H2s_vqclean.npz \
    --ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
    --out datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz

# 4a. 序列 Token BC
python -m training.vq.train_token_bc \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
    --save_dir checkpoints/token_bc_t4_cb64 \
    --epochs 50 --batch 256 --lr 3e-4 \
    --hidden_dim 128 --num_layers 2 --n_heads 4 \
    --class_weight_mode inverse_sqrt \
    --maneuver_quantile 0.7

# 4b. 单步 Token BC (部署用)
python -m training.vq.train_onestep_token_bc \
    --data datasets/dt2hz_H2s_vqclean_t4_cb64_onestep_h4.npz \
    --save_dir checkpoints/onestep_token_bc_t4_cb64_h4 \
    --epochs 50 --batch 512 --lr 3e-4 \
    --hidden_dim 128 --num_layers 2 \
    --class_weight_mode inverse_sqrt \
    --maneuver_quantile 0.7
```
