# 空战语义标注指南 v1

> **适用 schema**: `semantic_record_v1`
> **枚举注册表**: `enum_registry_v1.json`
> **日期**: 2026-03-24

---

## 1. 标注目标

本项目的标注目标是为 2v2 空战仿真 (AFSIM/ACMI) 中每架飞机在每个决策时间步赋予一个**高层战术语义标签** (`semantic_state`)，并标注配套的角色 (`role`)、目标引用 (`target_ref`)、行为约束 (`constraints`)、事件标志 (`event_flags`) 和标注理由 (`rationale_short`)。

标注产出将用于:

1. **训练 token policy 网络** -- 语义标签作为 token 到连续动作的 condition 信号
2. **构建战术状态机** -- 语义标签序列揭示典型战术决策流
3. **评估仿真质量** -- 标签分布和转移矩阵作为仿真对标指标

**核心原则**:

- 每个样本标注的是**该飞机在该时刻的战术意图**，而非物理运动本身
- 标签反映"飞行员想做什么"，不是"飞机正在怎么飞"
- 信息不足时使用 `null`，**绝不**用 `hold_geometry` 充当默认值

---

## 2. semantic_state 各标签定义与边界

以下按 9 个标签逐一展开。每个标签给出**定义**、**典型态势条件**、**与相邻标签的区分准则**、**常见误标场景**。

### 2.1 commit_intercept

**类别**: OBFM (进攻性超视距机动)

**定义**: 主动接敌，建立截获或压迫态势。飞机朝目标方向稳定飞行或浅转弯，并伴有加速或维速。这是 BVR 阶段最常见的进攻意图。

**典型态势条件**:
- range > 15 km (超视距距离)
- bearing 绝对值 < 30 deg (机头大致对准目标)
- closure < 0 m/s (距离在缩短)
- 航向变化率 < 3 deg/s (稳定飞行或微调)
- 速度维持或增加

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `hold_geometry` | commit 主动缩短距离 (closure < 0)；hold 的 closure 接近 0 或距离无明显变化 |
| `press_attack` | commit 在远距离建立态势 (range > 15 km)；press 在中近距离积极压入 (range < 15 km) |
| `energy_manage` | commit 有明确的目标朝向 (bearing < 30 deg)；energy_manage 无指向性 |

**常见误标场景**:
- 飞机刚完成转弯对准目标、但尚未稳定飞行，可能只是 offensive_turn 的尾段 -- 应标 `offensive_turn` 直到航向稳定
- 远距离巡航且尚未发现目标时 -- 应标 `null` 或 `hold_geometry`，不应标 commit

---

### 2.2 hold_geometry

**类别**: OBFM

**定义**: 保持当前几何关系，既不推进也不撤离。飞行员在等待时机 (如等待友机就位、等待武器包线条件满足) 或维持阵位。

**典型态势条件**:
- closure 接近 0 m/s (绝对值 < 20 m/s)
- 航向变化率 < 2 deg/s
- 速度基本不变 (加速度接近 0)
- range 无显著变化趋势

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `commit_intercept` | hold 的距离无明显缩短；commit 距离持续缩短 |
| `extend` | hold 维持距离；extend 主动拉大距离 (closure > 0)，且通常伴有背对目标的航向 |
| `energy_manage` | hold 维持在当前位置有战术意图 (如保持阵位)；energy_manage 纯粹是调整速度/高度 |

**常见误标场景**:
- **最严重**: 信息不足时将 `hold_geometry` 作为默认标签 -- **严禁**，应填 `null`
- 飞机实际在缓慢接近目标但标注者认为"变化不大" -- 应检查 closure，若持续 < -20 m/s 则应标 `commit_intercept`

---

### 2.3 press_attack

**类别**: OBFM

**定义**: 积极进入攻击，修正方向并加速压入。这是从 BVR 向 WVR 过渡的进攻阶段，飞行员决心攻击且正在快速接近。

**典型态势条件**:
- range < 15 km 且在快速缩短
- closure < -50 m/s (快速接近)
- 有方向修正 (heading 变化指向目标)
- 速度增加或维持高速
- bearing 绝对值 < 45 deg

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `commit_intercept` | press 距离更近 (< 15 km)，closure 更大 (< -50 m/s)，进攻意图更积极 |
| `merge_entry` | press 仍在中距范围；merge_entry 在 range < 5 km 的近距过渡段 |
| `offensive_turn` | press 以直线或浅修正为主；offensive_turn 是大幅度转弯争夺角度 |

**常见误标场景**:
- 飞机虽然距离近但在转弯绕飞 -- 应视转弯幅度标 `offensive_turn`
- 飞机快速接近但在 merge 前最后 5 km -- 应标 `merge_entry`

---

### 2.4 extend

**类别**: OBFM

**定义**: 脱离当前交战，拉开距离并恢复能量。飞行员选择暂时退出交战以获得更好的态势 (距离、能量、角度) 再重新接敌。

**典型态势条件**:
- closure > 0 m/s (距离在增大)
- 航向偏离目标 (bearing 绝对值 > 90 deg，即背对目标飞)
- 通常伴有加速或平飞恢复速度
- 非紧急防御 (没有大 bank 角急转)

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `hold_geometry` | extend 距离在增大且航向背离目标；hold 距离不变 |
| `defensive_break` | extend 是主动选择脱离、机动幅度平缓 (roll < 45 deg)；defensive_break 是被动紧急脱离、大 bank 角急转 |
| `energy_manage` | extend 有脱离目标的明确意图 (bearing > 90 deg)；energy_manage 无战术性离脱 |

**常见误标场景**:
- 飞机在高速转弯后拉开距离但目的是重新获取角度 -- 可能仍属于 `offensive_turn` 的一部分
- 紧急规避标为 extend -- 若伴有大 bank 角和急速变向应标 `defensive_break`

---

### 2.5 support

**类别**: Coordination (编队协同)

**定义**: 支援友机，执行掩护、配合或拉扯敌方注意力的行为。support 的核心判据是飞行员的主要关注对象是**友机**或**友机的交战对手**，而非自己直接交战。

**典型态势条件**:
- 与友机保持适当间距 (通常 5-15 km)
- 航向与友机交战轴线有偏移 (提供侧翼或后方掩护)
- 未直接朝最近的敌方平台飞行
- 友机正在 engaged 状态

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `commit_intercept` | support 的主要关注对象是友机而非敌方；commit 直接朝敌方飞 |
| `hold_geometry` | support 的"保持位置"有明确的掩护意图；hold 是单机维持阵位 |
| `energy_manage` | support 有战术协同意图；energy_manage 纯粹是能量调整 |

**常见误标场景**:
- 编队中两架飞机都在朝同一目标飞 -- 靠前的标 `commit_intercept` 或 `press_attack`，靠后有意掩护的标 `support`
- 飞机恰好在友机附近但没有掩护意图 -- 需结合航向与友机交战轴线的关系判断

---

### 2.6 merge_entry

**类别**: Merge (超视距到近距过渡)

**定义**: 进入近距交战的几何重构阶段，BVR 向 WVR 过渡。双方飞机即将或刚刚通过最近点 (closest point of approach)，需要重新建立角度优势。

**典型态势条件**:
- range < 5 km
- closure < -100 m/s (双方高速接近) 或刚通过 CPA
- aspect 接近 0 deg (头对头) 或正在快速变化
- 可能伴有预判转弯 (pre-merge turn)

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `press_attack` | press 在 > 5 km 距离；merge_entry 在 < 5 km 且即将进入近距格斗 |
| `offensive_turn` | merge_entry 是刚通过 merge 的过渡态；offensive_turn 是在 WVR 阶段内持续的角度争夺 |
| `commit_intercept` | commit 在远距；merge_entry 在近距过渡 |

**常见误标场景**:
- 已经完成 merge 并进入稳定的转弯格斗 -- 应标 `offensive_turn` 或 `defensive_break`
- range < 5 km 但双方平行飞行没有交叉意图 -- 可能是 `extend` 或 `hold_geometry`

---

### 2.7 offensive_turn

**类别**: BFM (基本格斗机动)

**定义**: 主动争夺角度优势的转弯机动。包括追踪转弯 (pursuit curve)、引导追踪 (lead turn)、滞后追踪 (lag pursuit) 等，核心是飞行员在近距格斗中积极转弯以获取射击条件。

**典型态势条件**:
- range < 10 km (通常 < 5 km)
- 航向变化率 > 5 deg/s
- roll 绝对值 > 30 deg
- 转弯方向朝目标 (试图减小 bearing)
- 非防御性质 (不是被迫转弯)

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `merge_entry` | offensive_turn 是 merge 后稳定的角度争夺；merge_entry 是刚通过 CPA 的过渡 |
| `defensive_break` | offensive_turn 主动争取角度、转弯朝向敌方；defensive_break 被动规避、转弯远离敌方 |
| `press_attack` | offensive_turn 以大幅转弯为主；press_attack 以直线压入为主 |
| `commit_intercept` | offensive_turn 在近距有大幅航向变化；commit 在远距稳定飞行 |

**常见误标场景**:
- 远距离 (> 10 km) 的方向修正 -- 应标 `commit_intercept`，不是 offensive_turn
- 转弯是为了脱离而非争夺角度 -- 应标 `extend` 或 `defensive_break`

---

### 2.8 defensive_break

**类别**: DBFM (防御性格斗机动)

**定义**: 防御性急转或脱离，生存优先。飞行员判断自己处于被动态势 (被咬尾、被导弹锁定、能量劣势)，执行急转、俯冲、barrel roll 等防御机动。

**典型态势条件**:
- aspect < 60 deg (敌方接近尾追状态，即敌方接近从后方看到己方)
- range < 10 km
- roll 绝对值 > 60 deg (大 bank 角急转)
- 可能伴有快速高度损失 (俯冲逃逸)
- 航向变化率 > 10 deg/s

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `offensive_turn` | defensive 的转弯目的是脱离/规避；offensive 的转弯目的是获取角度 |
| `extend` | defensive 是紧急性的大幅度机动 (roll > 60 deg)；extend 是相对平缓的脱离 |
| `energy_manage` | defensive 有明确的威胁触发；energy_manage 无近距威胁 |

**常见误标场景**:
- 飞机大 bank 角转弯但目的是进攻 (追踪转弯) -- 应标 `offensive_turn`，需要结合 aspect 和转弯方向判断
- 远距离 (> 15 km) 执行规避 -- 通常标 `extend` 更合适，defensive_break 主要出现在近距

---

### 2.9 energy_manage

**类别**: Energy (能量管理)

**定义**: 纯能量状态调整，非战术机动。飞行员在没有直接威胁或交战意图的情况下调整速度和高度 (如爬升恢复位置能量、平飞加速恢复动能)。

**典型态势条件**:
- 无近距威胁 (range > 20 km 或无敌方信息)
- 航向变化率 < 2 deg/s (直线或近似直线)
- 垂直速度非零 (在爬升/下降调整高度) 或有明显加速/减速
- bearing 与任何目标无明显对准关系

**与相邻标签的区分准则**:

| 对比标签 | 区分关键 |
|---|---|
| `commit_intercept` | energy_manage 无目标指向；commit 有明确目标方位对准 |
| `hold_geometry` | energy_manage 的高度/速度在明显变化；hold 整体状态保持不变 |
| `extend` | energy_manage 没有脱离交战的战术背景；extend 在交战后拉开距离 |

**常见误标场景**:
- 飞机在交战后爬升恢复能量 -- 若仍有脱离意图 (背对目标)，应标 `extend`
- 巡航中无任何信息 -- 应标 `null` 而非 `energy_manage`，除非能确认正在调整能量

---

## 3. role 定义与填写规则

`role` 描述本机在编队中的当前角色。取值范围:

| 值 | 含义 | 典型场景 |
|---|---|---|
| `engaged` | 直接交战中 (主攻或防御) | 本机正在 commit/press/merge/offensive_turn/defensive_break |
| `support` | 掩护配合友机 | 本机 semantic_state 为 support，或虽在其他状态但主要服务于友机交战 |
| `egressing` | 正在脱离交战区域 | 本机 semantic_state 为 extend 且明确脱离 |
| `neutral` | 无明确角色 | 巡航、energy_manage、或 hold_geometry 且无交战背景 |

**填写规则**:

1. **role 与 semantic_state 有关联但不完全绑定**。例如 `commit_intercept` 通常对应 `engaged`，但若 commit 的目的是掩护友机，则 role 可以是 `support`
2. **一架飞机在同一时刻只有一个 role**
3. **role 描述的是编队协作中的职能**，不是单机战术意图 (单机意图由 semantic_state 描述)
4. 信息不足时填 `null`

**常见对应关系** (参考，非硬绑定):

| semantic_state | 最常见 role |
|---|---|
| commit_intercept | engaged |
| hold_geometry | engaged 或 neutral |
| press_attack | engaged |
| extend | egressing |
| support | support |
| merge_entry | engaged |
| offensive_turn | engaged |
| defensive_break | engaged |
| energy_manage | neutral |

---

## 4. target_ref 填写规则

`label.target_ref` 标注本机当前的主要作用目标。取值:

- `enemy_1` -- 当前主要目标为敌方第一平台
- `enemy_2` -- 当前主要目标为敌方第二平台
- `ally_1` -- 当前主要关注对象为友方平台 (仅在 support 场景使用)
- `null` -- 无特定目标

### 4.1 何时填写各值

| 场景 | target_ref |
|---|---|
| 本机正在朝某敌方飞机飞行/攻击 | 填对应的 `enemy_1` 或 `enemy_2` |
| 本机正在对某敌方飞机执行防御 | 填威胁来源的 `enemy_1` 或 `enemy_2` |
| 本机执行 support，主要配合友机 | 填 `ally_1` |
| 本机执行 support，主要牵制某敌机 | 填对应的 `enemy_1` 或 `enemy_2` |
| energy_manage，无特定目标 | `null` |
| hold_geometry，无明确关注对象 | `null` |
| 信息不足 | `null` |

### 4.2 目标分配逻辑

1. **主要交战目标优先**: 如果飞机的航向、机动明确指向某个敌方平台，则填该平台
2. **最近威胁优先**: 当无法从航向判断时，选择距离最近且构成威胁的敌方平台
3. **一致性原则**: 同一交战阶段内，除非发生明确的 `target_switch` 事件，`target_ref` 应保持不变
4. **`enemy_1` / `enemy_2` 的序号** 在每个 episode 内固定分配 (按运行时 object ID)，跨 episode 不保证对应同一物理位置

### 4.3 注意事项

- `target_ref` 填的是 **label** 层面的标注判断，不是直接复制 `obs.enemy.target_ref`
- `obs.enemy.target_ref` 记录的是 obs 中当前观测到的敌方平台；`label.target_ref` 记录的是标注者判断的主要作用目标
- 两者可能不同 (例如 obs 中记录的是最近的敌机，但标注者判断飞行员其实在关注另一架)

---

## 5. event_flags.target_switch 判定规则

### 5.1 核心概念

`target_switch` 是一个**离散事件 (discrete event)**，不是持续状态。它仅在**目标切换发生的那一帧**为 `true`，其余帧为 `false` 或 `null`。

### 5.2 判定公式

```
target_switch = (label.target_ref != history.prev_target_ref)
                AND (label.target_ref != null)
                AND (history.prev_target_ref != null)
```

即: 当本帧的 `target_ref` 与上一帧的 `prev_target_ref` **都非 null 且不同**时，`target_switch = true`。

### 5.3 特殊情况处理

| 场景 | target_switch |
|---|---|
| 首帧 (prev_target_ref = null) | `null` 或 `false` |
| 从 null 变为 enemy_1 (首次锁定) | `false` (不算"切换"，算"首次指定") |
| 从 enemy_1 变为 enemy_2 | `true` |
| 从 enemy_1 变为 null (丢失目标) | `false` (丢失不是切换) |
| 从 enemy_2 变为 ally_1 | `true` |
| target_ref 连续多帧不变 | `false` |

### 5.4 关键警告

- **target_switch 不是 semantic_state**。不要因为 semantic_state 发生变化就设 target_switch = true
- **target_switch 不是 role 变化**。role 从 engaged 变为 support 不代表 target_switch
- target_switch 仅关注 `target_ref` 字段本身的前后对比

---

## 6. constraints 填写规则

`label.constraints` 是一个包含 4 个布尔字段的对象，描述当前状态下的行为约束。每个字段可独立为 `true`、`false` 或 `null`。

### 6.1 prefer_energy_preserve

**含义**: 优先保持能量 (避免大幅消耗速度/高度)

| 值 | 条件 |
|---|---|
| `true` | 当前能量状态为 low 或 medium-low; 或 semantic_state 为 energy_manage / extend; 或飞行员正在爬升恢复 |
| `false` | 能量充足且战术意图需要消耗能量 (如 press_attack 加速、offensive_turn 大转弯) |
| `null` | 无法判断当前能量状态 |

### 6.2 allow_aggressive_turn

**含义**: 允许大 bank 角 (> 60 deg) 的激进转弯

| 值 | 条件 |
|---|---|
| `true` | semantic_state 为 offensive_turn / defensive_break / merge_entry; 或态势需要快速改变航向 |
| `false` | semantic_state 为 commit_intercept / hold_geometry / energy_manage 等稳定飞行状态; 或能量过低不适合大幅机动 |
| `null` | 无法判断 |

### 6.3 must_support_teammate

**含义**: 必须考虑友机位置 (行为受友机状态约束)

| 值 | 条件 |
|---|---|
| `true` | role 为 support; 或友机处于 defensive_break 需要掩护; 或编队战术要求保持阵型 |
| `false` | 本机独立交战，友机状态不影响当前决策 |
| `null` | 友机信息不可用 (obs.ally 为 null) |

### 6.4 should_abort_if_threatened

**含义**: 受威胁时应自动切换为防御

| 值 | 条件 |
|---|---|
| `true` | 当前非防御状态但有潜在威胁 (如正在 commit_intercept 但另一敌机在侧翼); 或能量较低 |
| `false` | 当前已在防御状态 (defensive_break); 或态势绝对优势无需考虑威胁; 或正在 press_attack 决心全力进攻 |
| `null` | 无法评估威胁等级 |

---

## 7. profile_hint 填写规则

`label.profile_hint` 为下游 P6DOF 控制器提供 profile 建议。

| 值 | 何时使用 |
|---|---|
| `p6dof_semantic` | 默认值。标准语义 profile，保留 token 高度语义 (roll_from_token + vert_speed_from_token)。适用于大多数状态: commit_intercept, hold_geometry, press_attack, extend, support, energy_manage |
| `p6dof_aggressive_turn` | 高 bank 角维持高度的实验 profile。适用于 offensive_turn 和 defensive_break 中 roll > 60 deg 的阶段 |
| `auto` | 由执行层根据当前动作自动选择。当标注者不确定哪个 profile 更合适时使用 |
| `null` | 不提供 profile 建议。信息不足或当前帧与 P6DOF 控制无关 |

**注意**: profile_hint 是**建议**而非强制，下游执行层可以覆盖。

---

## 8. rationale_short 写法

`label.rationale_short` 是一句话标注理由，限 **120 字符**，供审阅时快速理解标注逻辑。

### 8.1 写法要求

1. **必须是一个完整短句**，说明"为什么标这个标签"
2. **包含关键数值依据** (距离、角度、速度等)
3. **不超过 120 字符** (含中文、数字、标点)
4. **中文书写**

### 8.2 各 semantic_state 示例

| semantic_state | rationale_short 示例 |
|---|---|
| commit_intercept | `朝敌方稳定飞行,range=45km,bearing=12deg,closure=-180m/s,建立截获态势` |
| hold_geometry | `保持距离30km阵位,closure接近0,等待友机就位` |
| press_attack | `range=8km快速接近,closure=-220m/s,修正航向压入攻击` |
| extend | `背对目标加速脱离,range从6km拉开至12km,恢复能量` |
| support | `保持友机侧后方8km,掩护友机对enemy_1的交战` |
| merge_entry | `range=3km,双方头对头closure=-300m/s,即将进入merge` |
| offensive_turn | `近距4km右转追踪,roll=55deg,争夺尾后角度` |
| defensive_break | `被enemy_2咬尾aspect=30deg,大bank角左急转脱离` |
| energy_manage | `无近距威胁,平飞加速从M0.7恢复至M0.85` |

---

## 9. null 使用规则

### 9.1 null 的语义

`null` = **信息不足，无法判断**

### 9.2 null 不等于什么

| null 不等于 | 说明 |
|---|---|
| `false` | `null` 表示不知道，`false` 表示明确否定 |
| `"no"` | 同上 |
| `hold_geometry` | hold 是一个有明确含义的标签 ("维持几何关系")，不是"不知道选什么" |
| 空字符串 `""` | schema 中不允许空字符串，用 `null` 代替 |

### 9.3 何时应标 null

- `semantic_state = null`: 观测数据不足以判断飞行员意图 (如仅有位置无速度/航向)
- `target_ref = null`: 无特定目标 (如 energy_manage) 或无法确定目标
- `role = null`: 无法判断编队角色
- `constraints` 各字段 = `null`: 缺少对应信息 (如无友机数据则 `must_support_teammate = null`)

### 9.4 强制规定

> **严禁将 `hold_geometry` 作为"不知道标什么"时的默认值。** 如果信息不足无法判断，必须标 `null`。`hold_geometry` 只在标注者确认飞行员确实在"有意维持当前几何关系"时才使用。

---

## 10. conflict_flag 判定规则

`quality.conflict_flag` 是一个布尔值，标记**多来源标注之间是否存在冲突**。

### 10.1 何时设为 true

- 同一样本被两个及以上来源 (如 acmi_ai + afsim_rule) 标注后，`semantic_state` 不一致
- AI 标注与规则标注的 `role` 或 `target_ref` 不一致
- 人工审核时发现 AI 标注明显不合理但尚未修正

### 10.2 何时设为 false

- 只有单一来源标注 (无比较对象)
- 多来源标注结果一致
- 冲突已被人工审核解决

### 10.3 与 needs_review 的关系

- `conflict_flag = true` 时，`needs_review` **必须**同时为 `true`
- `needs_review = true` 不一定要求 `conflict_flag = true` (可能是其他原因需要复核)

---

## 11. quality.tier 判定规则

| 等级 | 条件 | 典型场景 |
|---|---|---|
| `gold` | 人工审核确认 + 信息完整 + 标签可信 | 人工标注并经第二人复核; 或 AI 标注经人工确认 |
| `silver` | AI 高置信度 (confidence > 0.8) + 与规则标注一致; 或人工初标未经复核 | acmi_ai confidence=0.9 且 afsim_rule 结果相同 |
| `bronze` | AI 中低置信度 (0.5 <= confidence <= 0.8); 或单一来源; 或部分字段缺失 | 仅 acmi_ai 一个来源, confidence=0.65 |
| `weak` | 信息严重不足 + 标签存疑 + 仅供参考 | obs 大量字段为 null; 或 confidence < 0.5; 或明显异常帧 |

**注意事项**:

1. tier 是对**整条标注记录**的质量评估，不仅仅是 semantic_state
2. 如果核心字段 (semantic_state, target_ref) 为 null 且原因是信息不足，tier 通常不高于 `bronze`
3. `conflict_flag = true` 的记录 tier 不应高于 `silver`

---

## 12. 持续状态 vs 离散事件

本 schema 中有两类时间概念，标注时必须严格区分:

### 12.1 持续状态 (Continuous State)

**代表字段**: `semantic_state`, `role`, `target_ref`, `constraints`

- 描述某个时刻飞行员**正在处于**的状态
- 状态在多个连续时间步上保持不变是正常的
- 状态可以持续数秒到数十秒
- 例: 飞机在 t=10s 到 t=25s 持续处于 `commit_intercept`

### 12.2 离散事件 (Discrete Event)

**代表字段**: `event_flags.target_switch`

- 描述某个时刻**发生了**的事件
- 事件只在发生的那一帧为 `true`，之前之后都是 `false`
- 事件是瞬时的，不会"持续"
- 例: 在 t=15s 从 enemy_1 切换到 enemy_2，只有 t=15s 的 `target_switch = true`

### 12.3 概念对比

| | 持续状态 | 离散事件 |
|---|---|---|
| 问题 | "现在是什么状态?" | "这一刻发生了什么?" |
| 时间跨度 | 多帧持续 | 单帧触发 |
| 连续为 true | 正常 | 异常 (说明误标) |
| 典型字段 | semantic_state, role | target_switch |

### 12.4 常见错误

- 把 `target_switch` 在连续多帧标为 `true` -- 错误，切换只发生在一帧
- 因为 `semantic_state` 变化就设 `target_switch = true` -- 错误，两者独立
- 因为 `target_switch = true` 就改变 `semantic_state` -- 错误，目标切换不一定改变战术意图

---

## 13. 常见标注陷阱

### 陷阱 1: 用 hold_geometry 充当默认标签

**错误**: 不确定飞行员在做什么，就标 `hold_geometry`。
**正确**: 信息不足时标 `null`。`hold_geometry` 要求标注者确认飞行员**有意维持当前态势**。

### 陷阱 2: 混淆 extend 和 defensive_break

**错误**: 只要飞机在远离敌方就标 `extend`。
**正确**: `defensive_break` 是紧急的大幅度防御机动 (roll > 60 deg, 航向变化率 > 10 deg/s)；`extend` 是相对平缓的主动脱离。关键区分: 是否在近距被威胁状态下的紧急响应。

### 陷阱 3: target_switch 连续多帧标 true

**错误**: 从 enemy_1 切到 enemy_2 后，连续 5 帧都标 `target_switch = true`。
**正确**: 只在切换发生的第一帧标 `true`，后续帧 `target_ref` 保持 enemy_2 不变时 `target_switch = false`。

### 陷阱 4: 将物理运动直接映射为语义标签

**错误**: 飞机在转弯就标 `offensive_turn`。
**正确**: 需要判断转弯的**意图** -- 是为了争夺角度 (offensive_turn)、紧急规避 (defensive_break)、还是方向修正 (commit_intercept 中的浅转弯)。同样的物理运动可以对应不同的语义标签。

### 陷阱 5: 忽略 aspect 角在攻防判断中的作用

**错误**: 只看 range 和 closure 判断 commit 还是 press。
**正确**: 需要结合 `aspect` (敌方展示角)。aspect 接近 0 deg (头对头) 和 aspect 接近 180 deg (尾追) 对应完全不同的战术态势。aspect < 60 deg 时对方可能处于被咬尾状态。

### 陷阱 6: constraints 全部填 null 或全部填 false

**错误**: 偷懒将 4 个 constraints 字段统一填同一个值。
**正确**: 每个 constraint 独立判断。例如 `offensive_turn` 状态下: `prefer_energy_preserve = false` (允许消耗能量)、`allow_aggressive_turn = true` (允许大转弯)、`must_support_teammate = null` (如果友机信息不明)、`should_abort_if_threatened = true` (如果侧翼有威胁)。

### 陷阱 7: support 状态下 target_ref 错误指向

**错误**: semantic_state = support 时 target_ref 填 null。
**正确**: support 状态下 target_ref 应填**本机主要关注的对象** -- 如果是掩护友机则填 `ally_1`，如果是牵制敌机则填对应的 `enemy_1`/`enemy_2`。只有完全无法判断时才填 null。

### 陷阱 8: 混淆 commit_intercept 和 energy_manage

**错误**: 飞机在远距直线飞行就标 `commit_intercept`。
**正确**: `commit_intercept` 要求有明确的目标朝向 (bearing < 30 deg) 和距离缩短趋势 (closure < 0)。如果只是在远距巡航调整速度/高度且无明确目标方位对准，应标 `energy_manage` 或 `null`。
