# BC 训练管道使用说明

本目录包含从 Tacview ACMI 录像到 Transformer 行为克隆（BC）模型的完整训练管道。

---

## 目录结构

```
training/
├── README.md                        # 本说明文件
├── dataset_default_config.py        # 所有超参数的唯一配置源（不要随意修改）
├── entity_filter.py                 # 实体过滤器（Regex 名称过滤 + Orbiter 行为过滤）
├── heading_convention_config.py     # 航向约定（atan2 顺序、正方向）
├── acmi_to_dt_dataset_smooth.py     # ACMI 解析函数库（被其他脚本共享调用）
├── build_training_dataset.py        # 一键合成训练集入口
├── analyze_acmi_coverage.py         # ACMI 数据覆盖分析与质量报告
├── train_bc_transformer_smooth.py   # Transformer BC 模型训练脚本
└── tests/
    └── test_coverage_utils.py       # 38 个单元测试（wrap/unwrap/dpsi/entropy）
```

输入数据和输出产物位于 `wow/` 根目录：

```
wow/
├── tra_data/           # 原始 ACMI 文件（82 个）
├── datasets/           # 训练集 .npz 及配套元数据文件
└── reports/coverage/   # 覆盖分析报告、JSON、PNG 图表
```

---

## 环境准备

```bash
conda activate wow
# 依赖：numpy, torch, matplotlib（lazy import，可选）
```

---

## 快速开始（三步走）

### 第 1 步：数据覆盖分析（可选，用于摸底数据质量）

```bash
# 从 wow/ 目录运行
python training/analyze_acmi_coverage.py

# 同时生成"过滤前/后"对比图和推荐训练文件列表
python training/analyze_acmi_coverage.py --write_recommended_list 1
```

输出到 `wow/reports/coverage/`：
- `stats.json` / `stats_before.json` / `stats_after.json`
- `report.md`
- `plots/` — 高度、速度、|Δψ| 分布直方图（过滤前/后对比）

质量等级：**A**（≥500 有效窗口，<5% 异常值）/ **B**（≥100，<15%）/ **C**（≥20）/ **F**

---

### 第 2 步：合成训练集

```bash
# 默认（扫描全部 tra_data/，使用默认过滤器，输出到 datasets/）
python training/build_training_dataset.py

# 指定自定义输出路径
python training/build_training_dataset.py --out datasets/myset.npz

# 关闭 Orbiter 过滤（保留所有固定翼）
python training/build_training_dataset.py --exclude_orbiters 0

# 自定义 Regex 排除规则
python training/build_training_dataset.py \
    --exclude_regex "(?i)(A-50|E-3|AWACS|KC-135|Tanker)"
```

输出（`wow/datasets/` 下）：

| 文件 | 说明 |
|---|---|
| `dt2hz_H2s_fighteronly.npz` | 训练张量（obs + action + 归一化统计）|
| `dt2hz_H2s_fighteronly.meta.json` | 完整元数据（可读）|
| `dt2hz_H2s_fighteronly.filelist.txt` | 参与合成的 ACMI 文件路径 |
| `dt2hz_H2s_fighteronly.rejected.txt` | 排除记录（实体名、原因）|
| `dt2hz_H2s_fighteronly.stats.json` | 分布快照（高度/速度/|Δψ| 百分位 + 熵）|

NPZ 数组格式：
```
obs    [N, T=20, 8]  float32  — ENU 状态序列
action [N, T=20, 3]  float32  — [dpsi_rad, alt_sp_m, spd_sp_mps]
```

---

### 第 3 步：训练模型

```bash
# 首次训练（30 epoch，lr=3e-4）
python training/train_bc_transformer_smooth.py \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --save datasets/bc_transformer_fighteronly.pt \
    --epochs 30 --lr 3e-4

# 续训（加载最优 checkpoint，降低 lr 继续收敛）
python training/train_bc_transformer_smooth.py \
    --data datasets/dt2hz_H2s_fighteronly.npz \
    --save datasets/bc_transformer_fighteronly.pt \
    --epochs 10 --lr 1e-4
```

> **续训说明**：脚本自身不加载已有 checkpoint；若要从之前的权重继续，需在命令行传入 `--ckpt` 参数（或在脚本中手动 torch.load + 加载 state_dict）。实践中可直接以较小的 lr 重新训练以取得相似效果。

常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | — | 训练 .npz 路径（必填）|
| `--save` | `bc_transformer.pt` | 模型保存路径 |
| `--epochs` | 10 | 训练轮数 |
| `--batch` | 256 | Batch size |
| `--lr` | 3e-4 | 学习率 |
| `--d_model` | 128 | Transformer 隐层维度 |
| `--layers` | 4 | Encoder 层数 |
| `--heads` | 4 | 注意力头数 |
| `--w_dpsi` | 1.0 | dpsi 损失权重 |
| `--w_alt` | 0.01 | 高度损失权重 |
| `--w_spd` | 0.1 | 速度损失权重 |
| `--lambda_smooth` | 0.5 | 平滑正则权重 |

输出 checkpoint（`.pt`）包含：`model weights`, `obs_mean/std`, `act_mean/std`, `meta`, `args`

---

## 单元测试

```bash
# 从 wow/ 目录运行
python -m pytest training/tests/test_coverage_utils.py -v
# 预期：38 tests PASSED（< 1s）
```

---

## 超参数配置

所有数据管道超参数集中在 `dataset_default_config.py`，CLI 参数始终优先于此文件中的默认值。

**修改前必须重新运行 `sanity_heading_alignment.py` 验证。**

| 参数 | 值 | 说明 |
|---|---|---|
| `DT` | 0.5 s | 决策步长（2 Hz）|
| `H_SEC` | 2.0 s | 设定点预测时域 |
| `K` | 4 | 时域步数 = H_SEC / DT |
| `SEQ_LEN` | 20 | 每个训练窗口的步数 |
| `STRIDE` | 5 | 窗口滑动步长 |
| `MIN_SPEED` | 60 m/s | 过滤低速/地面滑行 |
| `MAX_SPEED` | 800 m/s | 硬上限（超出 = 轨迹损坏）|
| `DPSI_SIGN` | +1 | 右转→正 dpsi，与 AFSIM 验证一致 |

---

## 航向约定（勿改）

| 项目 | 值 |
|---|---|
| 航向公式 | `heading = atan2(vx_east, vy_north)` |
| 零位 | North（正北）|
| 正方向 | 顺时针（CW）|
| dpsi_sign | **+1**（右转 → 正值）|
| 训练单位 | rad / m / m/s |
| AFSIM 接口 | `TurnToRelativeHeading(deg)` → 调用前需 `math.degrees(dpsi)` |

---

## 实体过滤说明

### Regex 过滤（`entity_filter.py`）

默认排除以下非战斗机实体（AWACS / 加油机 / 运输机 / 轰炸机）：

```
(?i)(A-50|E-3|E-2|E-767|AWACS|Sentry|Hawkeye|
     KC-135|KC-10|KC-46|IL-78|Il-78|Tanker|
     C-130|C-17|C-5|An-26|An-12|Il-76|
     Tu-95|B-52|B-1|B-2|Bomber|Transport|JSTAR)
```

### Orbiter 行为过滤

飞行轨迹同时满足以下条件时判定为"盘旋/稳定巡逻"并排除：

| 条件 | 阈值 |
|---|---|
| 轨迹长度 | ≥ 600 步（= 300 s @ 2 Hz）|
| std(altitude) | < 80 m |
| std(speed) | < 7 m/s |
| p95(│Δψ│) | < 0.05 rad |

过滤效果（82 个 ACMI 文件）：共排除 164 个实体，|Δψ| p95 从 0.237 → 0.289 rad（+22%）。

---

## 已验证结果（参考基线）

| 指标 | 值 |
|---|---|
| 训练集 | 504,774 windows，10,095,480 样本 |
| 数据集大小 | 101 MB（NPZ）|
| 最佳 Val Loss | **0.1052**（epoch 39，初训 30ep @lr=3e-4 + 续训 10ep @lr=1e-4）|
| 模型大小 | 3.3 MB（`bc_transformer_fighteronly.pt`）|
