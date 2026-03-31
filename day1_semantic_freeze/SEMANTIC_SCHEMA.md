# 高层语义层输出 Schema

---

## 1. 最小 JSON Schema

```json
{
  "semantic_state": "commit_intercept",
  "target_id": "enemy_1",
  "role": "shooter",
  "constraints": {
    "prefer_energy_preserve": false,
    "allow_aggressive_turn": false,
    "must_support_teammate": false,
    "should_abort_if_threatened": false
  },
  "profile_hint": "p6dof_semantic"
}
```

---

## 2. 字段定义

### semantic_state (必填)

高层战术意图标签。

| 取值 | 类别 | 含义 |
|------|------|------|
| `commit_intercept` | obfm | 主动接敌 |
| `hold_geometry` | obfm | 保持几何关系 |
| `press_attack` | obfm | 积极进入攻击 |
| `extend` | obfm | 脱离拉开 |
| `support` | coordination | 支援友机 |
| `merge_entry` | merge | 进入近距交战 |
| `offensive_turn` | bfm | 攻击性转弯机动 |
| `defensive_break` | dbfm | 防御性急转脱离 |
| `energy_manage` | energy | 能量管理 |

- 不允许为空字符串
- 未知时使用 `"hold_geometry"` 作为**运行时安全 fallback**（不是"缺标注时的训练真值"；训练标注中不应将 hold_geometry 作为缺省填充）
- 完整定义见 `semantic_label_set_v0.json`

### target_id (可选)

当前主要作用目标的标识符。使用角色中性的命名，不绑定具体阵营。

| 取值 | 含义 |
|------|------|
| `"enemy_1"`, `"enemy_2"` | 敌方平台（按运行时分配的序号） |
| `"ally_1"` | 友方平台 |
| `"none"` | 无特定目标 (如 energy_manage, hold_geometry) |
| `"unknown"` | 有目标但无法确定 |
| 运行时对象 ID | 也可接受 AFSIM 运行时动态分配的 platform object ID |

- 默认值: `"none"`
- ID 命名是角色相对的（enemy/ally），不是阵营固定的（red/blue），以便在红蓝任一侧复用

### role (可选)

本机在编队中的角色。

| 取值 | 含义 |
|------|------|
| `"shooter"` | 主攻手, 负责直接交战 |
| `"supporter"` | 支援手, 掩护/配合 |
| `"independent"` | 独立行动 |
| `"unknown"` | 角色未定 |

- 默认值: `"independent"`
- **role 与 semantic_state 的区别**: role 描述的是编队中的分工, 是持续性的; semantic_state 描述的是当前战术意图, 可随时切换。同一 role=shooter 的飞机, semantic_state 可以从 commit_intercept 切换到 defensive_break。

### constraints (必填, 各子字段有默认值)

语义层向执行层传递的行为约束。

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `prefer_energy_preserve` | bool | false | 优先保持能量 (限制大转弯/大俯冲) |
| `allow_aggressive_turn` | bool | false | 允许使用大 bank 角 (>60 deg) 转弯 |
| `must_support_teammate` | bool | false | 必须考虑友机位置 |
| `should_abort_if_threatened` | bool | false | 受威胁时应自动切换到 defensive |

- 所有字段均可缺省, 缺省时使用默认值
- **constraints 与 profile_hint 的区别**: profile_hint 指定底层控制模式 (硬件级); constraints 是软约束, 影响 token family 偏好和决策逻辑, 不直接改变控制律。

### profile_hint (可选)

建议使用的 P6DOF 控制 profile。

| 取值 | 含义 |
|------|------|
| `"p6dof_semantic"` | 默认 profile, 保留 token 高度语义 |
| `"p6dof_aggressive_turn"` | 实验 profile, 高 bank 角时保持高度 |
| `"auto"` | 由执行层根据当前动作自动选择 |

- 默认值: `"auto"`
- 执行层保留覆盖权: 即使语义层建议 aggressive_turn, 执行层可因安全原因降级为 semantic

---

## 3. unknown / none 处理规则

| 字段 | 无信息时 | 含义 |
|------|---------|------|
| semantic_state | 用 `"hold_geometry"` | 运行时 fallback（非训练默认） |
| target_id | 用 `"none"` | 无特定目标 |
| role | 用 `"unknown"` | 未分配角色 |
| constraints | 全部用默认值 | 不施加约束 |
| profile_hint | 用 `"auto"` | 让执行层决定 |

---

## 4. 将来可扩展但当前不需要的字段

| 字段 | 类型 | 用途 | 不纳入原因 |
|------|------|------|-----------|
| `urgency` | enum(low/medium/high/critical) | 紧急程度 | 当前无实时威胁评估 |
| `weapon_state` | object | 武器状态 (剩余弹药等) | 当前仿真未建模 |
| `formation_directive` | string | 编队指令 | 当前 2v2 尚未建立编队协议 |
| `time_horizon_sec` | float | 策略时间窗口 | 当前固定 2s, 不需要变 |
| `confidence` | float | 语义判断置信度 | 等弱监督标注完成后加入 |

---

## 5. 合法样例

### 样例 1: 主攻接敌

```json
{
  "semantic_state": "commit_intercept",
  "target_id": "enemy_1",
  "role": "shooter",
  "constraints": {
    "prefer_energy_preserve": false,
    "allow_aggressive_turn": false,
    "must_support_teammate": false,
    "should_abort_if_threatened": true
  },
  "profile_hint": "p6dof_semantic"
}
```

### 样例 2: 防御急转

```json
{
  "semantic_state": "defensive_break",
  "target_id": "none",
  "role": "independent",
  "constraints": {
    "prefer_energy_preserve": false,
    "allow_aggressive_turn": true,
    "must_support_teammate": false,
    "should_abort_if_threatened": false
  },
  "profile_hint": "p6dof_aggressive_turn"
}
```

### 样例 3: 支援友机

```json
{
  "semantic_state": "support",
  "target_id": "enemy_2",
  "role": "supporter",
  "constraints": {
    "prefer_energy_preserve": true,
    "allow_aggressive_turn": false,
    "must_support_teammate": true,
    "should_abort_if_threatened": true
  },
  "profile_hint": "auto"
}
```

### 样例 4: 能量管理

```json
{
  "semantic_state": "energy_manage",
  "target_id": "none",
  "role": "independent",
  "constraints": {
    "prefer_energy_preserve": true,
    "allow_aggressive_turn": false,
    "must_support_teammate": false,
    "should_abort_if_threatened": true
  },
  "profile_hint": "p6dof_semantic"
}
```

### 样例 5: 近距攻击性转弯

```json
{
  "semantic_state": "offensive_turn",
  "target_id": "enemy_1",
  "role": "shooter",
  "constraints": {
    "prefer_energy_preserve": false,
    "allow_aggressive_turn": true,
    "must_support_teammate": false,
    "should_abort_if_threatened": false
  },
  "profile_hint": "p6dof_aggressive_turn"
}
```
