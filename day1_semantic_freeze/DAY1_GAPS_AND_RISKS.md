# Day 1 差距与风险评估

---

## 1. 当前代码/数据是否已足够支撑高层语义层

### 已具备

| 能力 | 状态 | 来源 |
|------|------|------|
| Ego 飞行状态 (8D) | 完备 | obs[t, 0:8], 速度/高度/航向/位置/垂直速度 |
| Token 解码到物理动作 | 完备 | VQ-VAE decode + denormalize, 30 个 token 全部可解码 |
| Token family 粗分类 | Day 1 产出 | `token_family_catalog.csv`, 30 个 token 分为 ~15 个 family |
| P6DOF 执行接口 | 完备 | 2 个 profile, 14+ 可覆盖参数, guardrails |
| 回归测试 | 完备 | 80 项回归, 覆盖 chunk 执行/latch/G-load/稳定性 |
| 文档-代码一致性 | 良好 | MIGRATION_HANDOFF 与代码高度对齐 |

### 部分具备

| 能力 | 状态 | 缺口 |
|------|------|------|
| 能量状态估算 | 可计算 | 需定义 E_spec = alt + v^2/(2g) 的 low/medium/high 阈值 |
| 俯仰角近似 | 可计算 | pitch ~ asin(vz/speed), 假设小侧滑 |

### 不具备 (关键缺口)

| 能力 | 缺口严重度 | 说明 |
|------|-----------|------|
| **敌我配对态势** | **严重** | 训练 NPZ 为单实体窗口, 无 enemy/ally 位置/速度/航向; 无法计算 range/bearing/aspect/closure |
| **多机交互信息** | **严重** | 640 值帧中有 80 个平台的状态, 但训练数据抽取时丢失了配对关系 |
| **武器包线** | **严重** | AFSIM 仿真中无武器发射事件数据接入; 无法判断 has_shot_opportunity |
| **Roll 角** | **中等** | AFSIM 状态帧不回传 roll; 无法直接获取实际 bank 角 |
| **威胁评估** | **中等** | 无现成规则; 需基于 bearing/aspect/range 自定义 |
| **编队协议** | **中等** | 2v2 场景无编队角色分配机制; support 标签当前无法自动判断 |

---

## 2. 最严重的字段缺失

### Top 1: 敌我相对几何 (enemy.range, bearing, aspect, closure)

- **影响**: 9 个语义标签中至少 6 个 (commit_intercept, press_attack, merge_entry, offensive_turn, defensive_break, support) 需要敌我几何才能判断
- **根因**: 训练数据管线 (`build_vq_clean_dataset.py`) 按单实体抽取窗口, 丢弃了同一场景中其他实体的信息
- **缓解路径**:
  1. **短期 (Day 2)**: 从 AFSIM 640 值帧中提取对手平台状态, 在 runtime bridge 层计算相对几何 -> 填充 state_summary
  2. **中期**: 回溯 ACMI 文件, 基于时间戳重新配对同一场景中的多个实体, 建立 ego-enemy-ally 三元组数据集

### Top 2: 武器包线数据

- **影响**: has_shot_opportunity 字段完全不可得; press_attack / wez_commit 的区分缺少关键依据
- **根因**: 当前仿真未建模武器发射; ACMI 数据中的 weapon events 已有解析工具 (`tools/acmi2weapon_events.py`) 但未集成到语义层
- **缓解路径**: 利用已有的 `acmi2weapon_events.py` 提取射击事件, 在离线标注时作为弱监督信号

### Top 3: Roll 角

- **影响**: 无法直接验证 AoA 饱和和 infeasible level turn 的实时触发条件
- **缓解路径**: 在 bridge 层从 psi_dot 和 speed 反推 `phi = atan(V * psi_dot / g)`, 作为估计值

---

## 3. 文档和代码不一致的地方

| 位置 | 文档说法 | 代码实际 | 严重度 |
|------|---------|---------|--------|
| TRAINING_PIPELINE.md | 已修正为 `t4_cb64` | checkpoint 实际 `t4_cb64` | **已修复** |
| 用户任务书 "当前主线" | token_steps=2, codebook=128 | 实际 token_steps=4, codebook=64 | **中** (任务书描述了早期版本, 以代码为准) |
| P6DOF_CONTROL_ARCHITECTURE.md | AoA flag 定义完整 | bridge 代码中未实时检测 alpha | **低** (设计文档 vs 实现优先级) |

---

## 4. 最可能阻塞 Day 2 弱监督标注的问题

### 阻塞 1: 缺少配对态势数据

弱监督标注的核心是"给定态势, 判断语义标签"。但当前样本 (`representative_state_windows.jsonl`) 只有 ego 信息, 无法判断:
- 是在接敌还是在巡航? (需要 enemy range/bearing)
- 是防御还是进攻? (需要 aspect angle)
- 是支援还是独立行动? (需要 ally position)

**建议**: Day 2 首先从 AFSIM 640 值帧中提取多平台位置, 或从 ACMI 回溯配对。

### 阻塞 2: 语义标签边界模糊

- `commit_intercept` vs `hold_geometry`: 仅从 ego 观测无法区分"朝目标飞"和"原地巡航"
- `press_attack` vs `offensive_turn`: 都涉及转弯, 区别在于距离和意图
- 没有几何信息, 标注者 (人或 LLM) 只能猜测

### 阻塞 3: 缺少标注工具

当前无任何标注 UI 或半自动标注脚本。需要:
- 一个能加载 JSONL 样本的简单 viewer
- 显示 ego 轨迹 + token 序列
- 允许标注者选择 semantic_state

---

## 5. 最小可行下一步 (不超过 5 条)

1. **提取 2v2 配对态势**: 从 AFSIM 运行时的 640 值帧中, 提取 red_1/red_2 vs blue_1/blue_2 的相对几何 (range, bearing, aspect, closure), 填充 `state_summary_schema.json` 的 enemy/ally 字段。这是**所有后续工作的前置依赖**。

2. **回溯 ACMI 建立配对数据集**: 利用 ACMI 文件中的时间戳和实体 ID, 将单实体窗口重新组装为 ego-enemy 配对样本。输出格式: `{ego_obs[20,8], enemy_obs[20,8], token_ids[5], ...}`。

3. **基于配对数据做首轮弱监督标注**: 用简单规则 (如 range 递减 + bearing < 30 deg = commit_intercept) 生成初步标签, 作为 LLM 标注的 seed。

4. **建立 bridge 层态势提取模块**: 在 `sim/` 中新增一个 `situation_extractor.py`, 从 640 值帧实时计算所有 `state_summary_schema.json` 中标为 "available_at_runtime_only" 的字段, 供语义层调用。
