# ACMI 字段映射规范 v1

> semantic_data_contract_v1 / acmi_field_mapping_v1
>
> 本文档定义了从 Tacview ACMI 文件到 canonical semantic record 各字段的完整映射关系,
> 包括数据来源、转换公式、单位、边界条件及当前不可得字段。

---

## 1. 概述

### 1.1 目的

本规范为"ACMI -> semantic record"转换管线提供唯一事实来源(single source of truth),确保:

- 所有字段的来源可溯、公式可审、单位无歧义;
- 下游消费者(VQ-VAE tokenizer、token policy、训练 pipeline)对数据语义的理解一致;
- 当前已实现字段与尚不可得字段的边界清晰,避免下游误用 null/默认值。

### 1.2 范围

| 项目 | 说明 |
|------|------|
| 上游来源 | Tacview ACMI 2.x 文件(`.acmi` / `.zip`) |
| 解析器 | `sim/tools/acmi2tspi.py`(轨迹解析)、`sim/tools/acmi2weapon_events.py`(武器事件） |
| 训练数据构建器 | `training/vq/build_vq_clean_dataset.py` |
| 目标 schema | `semantic_data_contract_v1/semantic_record_schema_v1.json` 中定义的 canonical record |
| 坐标系约定 | ENU (East-North-Up), 原点 = ACMI ReferenceLatitude/ReferenceLongitude |

### 1.3 ACMI 轨迹点原始格式

```
T=lon|lat|alt|roll_deg|pitch_deg|yaw_deg
```

6 个管道分隔字段,后三项可为空字符串。时间戳由 `#<float>` 行给出。

### 1.4 Identity 属性键

```
Name, Type, Coalition, Country, Group, Color, Pilot
```

由 `side_from_props(props)` 推断阵营:优先级为 Color > Coalition > Group/Name 文本匹配 > 回退 `"unknown"`。

---

## 2. ego 字段映射

以下所有公式中, 下标 $t$ 表示当前重采样时间步, $\Delta t$ 为重采样间隔(默认 0.5 s)。

### 2.1 基础标识字段

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `runtime_id` | string | ACMI object ID (含 `__seg` 后缀) | 唯一标识当前实体段 |
| `callsign` | string | `props["Name"]` | ACMI Name 属性 |
| `coalition` | string | `side_from_props(props)` | `"red"` / `"blue"` / `"unknown"` |

### 2.2 运动学字段

#### 2.2.1 pos_east_m / pos_north_m (ENU 位置)

**来源**: ACMI 经纬度 -> `latlon_to_enu()` 转换

**公式**:

$$
x_e = (\lambda - \lambda_0) \cdot \cos\!\Bigl(\frac{\varphi + \varphi_0}{2}\Bigr) \cdot R_E
$$

$$
y_n = (\varphi - \varphi_0) \cdot R_E
$$

其中:

- $\varphi, \lambda$ = 当前点纬度、经度 (弧度)
- $\varphi_0, \lambda_0$ = 参考点纬度、经度 (弧度), 取自 ACMI `ReferenceLatitude` / `ReferenceLongitude`
- $R_E = 6378137.0\ \text{m}$ (WGS-84 赤道半径)

**单位**: m
**边界**: 无硬限制; 典型范围 $\pm 200\,\text{km}$

#### 2.2.2 altitude_m

**来源**: ACMI `T` 字段第 3 个分量 (index=2), 经线性插值重采样

**单位**: m (MSL)
**边界**: $\geq 0$; 仿真中通常 $[300, 15000]$ m

#### 2.2.3 heading_deg

**来源**: 优先使用 ACMI `yaw_deg` (T 字段 index=5); 若缺失则由 ENU 速度分量推导。

**公式 (速度推导路径)**:

$$
\psi = \text{atan2}(v_{x,e},\ v_{y,n})
$$

其中:

$$
v_{x,e} = \frac{d\,x_e}{dt}, \quad v_{y,n} = \frac{d\,y_n}{dt}
$$

导数采用 `numpy.gradient()` 中心差分。

**unwrap**: 内部以弧度存储, 经 `unwrap_series()` 消除 $\pm\pi$ 跳变,保持连续。

**canonical 单位**: deg
**转换**: $\psi_{\text{deg}} = \psi_{\text{rad}} \times \frac{180}{\pi}$
**约定**: 0 = 正北, 顺时针为正
**范围**: $[0, 360)$ (canonical record) 或 $(-\infty, +\infty)$ (unwrapped 内部)

#### 2.2.4 speed_mps

**来源**: 由 ENU 位置差分计算, 为三维速率:

$$
v_{3D} = \sqrt{v_{x,e}^2 + v_{y,n}^2 + v_{z,u}^2}
$$

**下限截断**: $\max(v_{3D},\ 1.0)$ (避免除零)

**单位**: m/s
**边界**: $\geq 1.0$ m/s (代码 clamp)

> **注意**: obs 8D 中的 `ground_speed_mps` 仅含水平分量 $\sqrt{v_{x,e}^2 + v_{y,n}^2}$;
> canonical `speed_mps` 使用三维速率, 与 `acmi2tspi.py` 中 `speed_from_pos` 一致。

#### 2.2.5 vertical_speed_mps

**来源**:

$$
v_{z,u} = \frac{d\,z_u}{dt}
$$

**单位**: m/s
**符号约定**: 正 = 上升, 负 = 下降

#### 2.2.6 pitch_deg

**来源**: 优先由轨迹推导 (`compute_attitude_from_trajectory`); ACMI 原始 pitch 字段被忽略 (注释: "often accumulated rotation values")。

**公式**:

$$
\theta = \text{atan2}\!\bigl(v_{z,u},\ v_H\bigr)
$$

其中 $v_H = \max\!\bigl(\sqrt{v_{x,e}^2 + v_{y,n}^2},\ 1.0\bigr)$

**后处理**:

1. 移动平均平滑, 窗口 = `smooth_window` (默认 21 步)
2. 截断: $\theta \in [-80^\circ, +80^\circ]$

**canonical 单位**: deg
**转换**: $\theta_{\text{deg}} = \theta_{\text{rad}} \times \frac{180}{\pi}$

#### 2.2.7 roll_deg

**来源**: 由协调转弯模型推导 (coordinated-turn bank angle):

$$
\phi = \text{atan2}\!\bigl(v_{3D} \cdot \dot\psi,\ g\bigr)
$$

其中:

- $v_{3D} = \max\!\bigl(\sqrt{v_{x,e}^2 + v_{y,n}^2 + v_{z,u}^2},\ 1.0\bigr)$
- $\dot\psi$ = unwrapped heading 的时间导数 (`numpy.gradient(heading_unwrap, dt)`), 经移动平均平滑
- $g = 9.80665\ \text{m/s}^2$

**后处理**:

1. 移动平均平滑, 窗口 = `smooth_window`
2. 截断: $\phi \in [-75^\circ, +75^\circ]$

**备注**: 若 ACMI 提供了可信的 roll 值, 可直接使用; 当前代码默认走推导路径。

**canonical 单位**: deg

#### 2.2.8 energy_state

**当前不可得** -- 见第 9 节。

---

## 3. enemy 字段映射

`obs.enemy` 数组中每个元素对应一个敌方实体, 需要两个实体同一时刻的状态联合计算。

设 ego 状态: 位置 $(x_e^{ego}, y_n^{ego}, z_u^{ego})$, 航向 $\psi^{ego}$ (rad), 速度矢量 $(v_x^{ego}, v_y^{ego}, v_z^{ego})$;
设 enemy 状态: 位置 $(x_e^{en}, y_n^{en}, z_u^{en})$, 航向 $\psi^{en}$ (rad), 速度矢量 $(v_x^{en}, v_y^{en}, v_z^{en})$。

### 3.1 target_ref

**类型**: string, `"enemy_1"` 或 `"enemy_2"`
**分配规则**: 见第 7 节。

### 3.2 range_m (斜距)

$$
\Delta x = x_e^{en} - x_e^{ego}, \quad \Delta y = y_n^{en} - y_n^{ego}, \quad \Delta z = z_u^{en} - z_u^{ego}
$$

$$
r = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}
$$

**单位**: m
**边界**: $\geq 0$

### 3.3 bearing_deg (相对方位角)

从 ego 航向到目标方位线的偏转角:

$$
\alpha_{abs} = \text{atan2}(\Delta x,\ \Delta y)
$$

$$
\beta = \alpha_{abs} - \psi^{ego}
$$

$$
\text{bearing\_deg} = \text{wrap}_{[-\pi,\pi]}(\beta) \times \frac{180}{\pi}
$$

其中 $\text{wrap}_{[-\pi,\pi]}(x) = ((x + \pi) \bmod 2\pi) - \pi$。

**约定**: 正 = 右舷 (顺时针), 负 = 左舷
**范围**: $[-180, 180]$ deg
**单位**: deg

> **注意**: `atan2` 参数顺序为 `atan2(east_component, north_component)`, 使 0 对应正北, 与 heading 约定一致。

### 3.4 aspect_deg (进入角)

Aspect angle 描述 ego 从 enemy 视角的进入方向:

$$
\alpha_{ego \leftarrow en} = \text{atan2}\bigl(-\Delta x,\ -\Delta y\bigr)
$$

(即从 enemy 看向 ego 的绝对方位)

$$
\text{aspect} = \text{wrap}_{[-\pi,\pi]}\!\bigl(\alpha_{ego \leftarrow en} - \psi^{en}\bigr)
$$

$$
\text{aspect\_deg} = |\text{aspect}| \times \frac{180}{\pi}
$$

**约定**:

- $0^\circ$ = 迎头 (head-on, ego 正对 enemy 机头)
- $180^\circ$ = 尾追 (tail chase, ego 从 enemy 尾部接近)

**范围**: $[0, 180]$ deg
**单位**: deg

### 3.5 closure_mps (接近速率)

沿视线方向的相对速度分量:

$$
\hat{r} = \frac{(\Delta x,\ \Delta y,\ \Delta z)}{r}
$$

$$
\Delta v = (v_x^{en} - v_x^{ego},\ v_y^{en} - v_y^{ego},\ v_z^{en} - v_z^{ego})
$$

$$
\text{closure\_mps} = \Delta v \cdot \hat{r} = \frac{\Delta x \cdot \Delta v_x + \Delta y \cdot \Delta v_y + \Delta z \cdot \Delta v_z}{r}
$$

**约定**: 负 = 接近 (closing), 正 = 远离 (opening)

> 助记: $\Delta v$ 为 enemy 减 ego, 投影到从 ego 指向 enemy 的单位矢量上;
> 当 enemy 相对 ego 在视线方向远离时为正。

**单位**: m/s
**边界**: 当 $r < \epsilon$ 时, closure 设为 0 以避免除零 (建议 $\epsilon = 1.0$ m)。

### 3.6 alt_diff_m (高度差)

$$
\text{alt\_diff\_m} = z_u^{ego} - z_u^{en}
$$

**约定**: 正 = ego 更高
**单位**: m

### 3.7 is_primary_threat

**当前不可得** -- 见第 9 节。需要威胁评估模型或规则输入。

---

## 4. ally 字段映射

### 4.1 ally_ref

**类型**: string, `"ally_1"`
**分配规则**: 见第 7 节。

### 4.2 range_m

与 enemy `range_m` 公式完全相同, 将 enemy 替换为 ally:

$$
r_{ally} = \sqrt{(x_e^{ally} - x_e^{ego})^2 + (y_n^{ally} - y_n^{ego})^2 + (z_u^{ally} - z_u^{ego})^2}
$$

**单位**: m

### 4.3 bearing_deg

与 enemy `bearing_deg` 公式完全相同, 将 enemy 替换为 ally。

**约定**: 正 = 右舷, 负 = 左舷
**范围**: $[-180, 180]$ deg

### 4.4 is_supporting

**当前不可得** -- 见第 9 节。需要战术角色判定逻辑。

---

## 5. engagement 标志映射

所有 engagement 标志均为布尔值, 基于阈值规则生成。以下为推荐阈值, 具体值应记录在 pipeline 配置中。

### 5.1 is_merge

当任一敌方的斜距低于 merge 阈值时为 true:

$$
\text{is\_merge} = \exists\ en \in \text{enemies}: r_{en} < R_{merge}
$$

**推荐阈值**: $R_{merge} = 3000\ \text{m}$ (约 1.6 nmi)

### 5.2 is_defensive

当满足以下全部条件时为 true:

1. 至少一个敌方的 aspect_deg > $A_{def}$ (对方处于我方尾部锥)
2. 该敌方的 closure_mps < 0 (正在接近)
3. 该敌方的 range_m < $R_{def}$

$$
\text{is\_defensive} = \exists\ en: \bigl(\text{aspect}_{en} > A_{def}\bigr) \land \bigl(\text{closure}_{en} < 0\bigr) \land \bigl(r_{en} < R_{def}\bigr)
$$

**推荐阈值**: $A_{def} = 120^\circ$, $R_{def} = 15000\ \text{m}$

### 5.3 has_shot_opportunity

当满足以下全部条件时为 true:

1. bearing_deg 绝对值 < $B_{shot}$ (目标在机头锥内)
2. range_m < $R_{shot}$
3. aspect_deg < $A_{shot}$ (近似正面或侧面接近)

$$
\text{has\_shot\_opportunity} = \exists\ en: \bigl(|\text{bearing}_{en}| < B_{shot}\bigr) \land \bigl(r_{en} < R_{shot}\bigr) \land \bigl(\text{aspect}_{en} < A_{shot}\bigr)
$$

**推荐阈值**: $B_{shot} = 30^\circ$, $R_{shot} = 20000\ \text{m}$, $A_{shot} = 90^\circ$

---

## 6. history 字段维护

`obs.history` 保存上一时间步的语义状态, 用于 token policy 的 Markov 依赖。

### 6.1 字段定义

| 字段 | 类型 | 说明 |
|------|------|------|
| `prev_semantic_state` | string \| null | 上一步的顶层语义状态标签 (如 `"offensive"`, `"defensive"`, `"neutral"`) |
| `prev_token_family` | string \| null | 上一步 VQ codebook 量化后的 token family 标签 |
| `prev_target_ref` | string \| null | 上一步的 `target_ref` (如 `"enemy_1"`) |

### 6.2 维护规则

1. **初始化**: 时间序列第一帧, 所有 history 字段设为 `null`。
2. **逐帧传播**: 在时间步 $t$ 完成 semantic record 构建后, 将本步结果写入 $t+1$ 的 history:
   - `prev_semantic_state` <- 当前步经规则或模型推断的状态标签
   - `prev_token_family` <- 当前步 action 经 VQ codebook 量化后的 family
   - `prev_target_ref` <- 当前步 `obs.enemy[*].target_ref` 中被选为主目标的 ref
3. **时间序列断裂处理**: 当检测到时间间隙 > `gap_thresh` (默认 2.0 s) 时, 重置 history 为 `null`。此逻辑与 `build_vq_clean_dataset.py` 中 `split_by_gaps()` 保持一致。
4. **跨实体隔离**: 不同 `runtime_id` 的 history 不共享, 各自独立维护。

---

## 7. target_ref / ally_ref 分配规则

### 7.1 背景

2v2 空战场景中, ego 面对 2 个敌方和 1 个友方。需要为每个对象分配稳定的 ref ID。

### 7.2 enemy ref 分配

**`enemy_1` / `enemy_2` 分配算法**:

1. 获取当前时刻所有存活敌方实体, 按 ACMI object ID 字典序排列。
2. 第一个为 `enemy_1`, 第二个为 `enemy_2`。
3. 一旦某架飞机在第一帧被分配了 ref, 后续帧即使距离变化也保持该 ref, 直到该实体消失。
4. 若一个敌方实体消失 (ACMI 中无后续轨迹点), 其 ref 保留但标记为 inactive; 不重新分配给新出现的实体。

**稳定性保证**: 通过固定 ACMI ID 排序而非基于距离的动态分配, 防止 ref 在帧间跳变。

### 7.3 ally ref 分配

**`ally_1` 分配算法**:

1. 从同阵营 (coalition 相同) 实体中排除 ego 本身。
2. 2v2 场景下仅有 1 个友方, 直接分配为 `ally_1`。
3. 若存在多个友方 (扩展场景), 按 ACMI object ID 字典序排列后依次分配 `ally_1`, `ally_2`, ...

### 7.4 当前限制

训练数据来自 `build_vq_clean_dataset.py` 的单实体窗口, coalition/enemy 配对关系在构建过程中丢失。重建配对需要回溯到 ACMI 原始文件, 利用 `side_from_props()` 按阵营分组后重新匹配。

---

## 8. weapon_event 接入

### 8.1 来源

`sim/tools/acmi2weapon_events.py` 的 `extract_weapon_events()` 输出 JSON, 包含:

```json
{
  "acmi": "<path>",
  "reference_latitude": <float>,
  "reference_longitude": <float>,
  "n_events": <int>,
  "events": [
    {
      "missile_id": "<acmi_oid>",
      "missile_name": "<Name>",
      "missile_type": "<Type>",
      "side": "red|blue",
      "launch_time": <float>,
      "end_time": <float>,
      "duration_s": <float>,
      "n_points": <int>,
      "shooter": {
        "id": "<acmi_oid>",
        "slot": "<blue_1|red_2|...>",
        "name": "<Name>",
        "method": "<parent|nearest_same_side_launch|...>",
        "distance_m": <float|null>
      },
      "target": {
        "id": "<acmi_oid>",
        "slot": "<blue_1|red_2|...>",
        "name": "<Name>",
        "method": "<target|closest_approach_opposite_side|...>",
        "distance_m": <float|null>
      }
    }
  ]
}
```

### 8.2 shooter 解析优先级

1. ACMI 关系属性: `Parent` > `ParentId` / `ParentID` > `Source` > `Launcher` (直接匹配 ACMI object ID)
2. 回退: 发射时刻同阵营距离最近的飞机 (`nearest_same_side_launch`)

### 8.3 target 解析优先级

1. ACMI 关系属性: `Target` > `LockedTarget`
2. 回退: 武器飞行全程中对侧阵营最小距离实体 (`closest_approach_opposite_side`)

### 8.4 映射到 semantic record

| weapon_event 字段 | semantic record 映射位置 | 说明 |
|---|---|---|
| `launch_time` | `extras.weapon_event_refs[i].launch_time` | 匹配到 ego 时间序列最近帧 |
| `shooter.id` | `extras.weapon_event_refs[i].shooter_runtime_id` | 与 ego/enemy/ally 的 `runtime_id` 关联 |
| `shooter.slot` | `extras.weapon_event_refs[i].shooter_ref` | 如 `"ego"`, `"enemy_1"`, `"ally_1"` |
| `target.id` | `extras.weapon_event_refs[i].target_runtime_id` | 同上 |
| `target.slot` | `extras.weapon_event_refs[i].target_ref` | 同上 |
| `missile_name` | `extras.weapon_event_refs[i].weapon_name` | 武器名称 |
| `duration_s` | `extras.weapon_event_refs[i].tof_s` | 飞行时间 |
| `shooter.method` | `extras.weapon_event_refs[i].shooter_resolve_method` | 解析方法, 用于置信度评估 |
| `target.method` | `extras.weapon_event_refs[i].target_resolve_method` | 解析方法 |

### 8.5 时间对齐

武器事件的 `launch_time` 使用 ACMI 绝对时间。映射到 semantic record 时需减去 ego 轨迹的 `source_start_time`, 对齐到重采样后的时间网格, 取最近帧索引:

$$
i_{frame} = \text{round}\!\Bigl(\frac{t_{launch} - t_{start}^{ego}}{\Delta t}\Bigr)
$$

---

## 9. 当前不可得字段

以下字段在 canonical record schema 中已定义, 但当前 ACMI 解析管线无法提供, 必须填 `null`。

| 字段 | 所属 section | 不可得原因 |
|------|-------------|-----------|
| `energy_state` | `obs.ego` | ACMI 不记录飞机比能量; 需要机型性能包络数据 (推重比、阻力极线) 才能计算 $E_s = h + v^2/2g$ 相对性能边界的百分比 |
| `is_primary_threat` | `obs.enemy` | 需要威胁评估模型 (综合 range、aspect、closure、武器状态); ACMI 无此信息 |
| `is_supporting` | `obs.ally` | 需要战术角色推断 (engaged vs supporting); ACMI 无编队角色标注 |
| `prev_semantic_state` | `obs.history` | 首帧必然为 null; 后续帧依赖语义状态分类器, 目前未集成 |
| `prev_token_family` | `obs.history` | 首帧必然为 null; 后续帧依赖 VQ codebook 量化, 仅在 token pipeline 运行时可得 |

> **处理约定**: 下游消费者在遇到 `null` 字段时, 必须使用该字段对应的默认值 (schema 中定义) 或跳过依赖该字段的逻辑分支, 不得假设任何具体数值。

---

## 10. 单位对照表

| 量纲 | canonical record 单位 | ACMI 原始单位 | acmi2tspi 内部单位 | 转换公式 |
|------|----------------------|--------------|-------------------|---------|
| 经度 | -- (转换为 ENU m) | deg | deg -> rad | $\lambda_{rad} = \lambda_{deg} \times \frac{\pi}{180}$ |
| 纬度 | -- (转换为 ENU m) | deg | deg -> rad | 同上 |
| 高度 | m (MSL) | m | m | 直传 |
| 距离 | m | -- | m | ENU 欧氏距离 |
| 航向 | deg, 0=N, CW+ | deg (yaw) | rad, unwrapped | $\psi_{deg} = \psi_{rad} \times \frac{180}{\pi}$ |
| 速度 | m/s | -- | m/s | 由位置差分计算 |
| 垂直速度 | m/s | -- | m/s | $dz/dt$ |
| 俯仰角 | deg | deg (ACMI, 不使用) | rad | $\theta_{deg} = \theta_{rad} \times \frac{180}{\pi}$ |
| 滚转角 | deg | deg (ACMI, 不使用) | rad | $\phi_{deg} = \phi_{rad} \times \frac{180}{\pi}$ |
| 方位角 (bearing) | deg, +右 -左 | -- | rad | $\beta_{deg} = \beta_{rad} \times \frac{180}{\pi}$ |
| 进入角 (aspect) | deg, 0=迎头, 180=尾追 | -- | rad | 同上 |
| 接近速率 | m/s, 负=接近 | -- | m/s | 视线投影 |
| 高度差 | m, 正=ego更高 | -- | m | $z^{ego} - z^{en}$ |
| 时间 | s | s | s | 直传 |
| 地球半径常量 | -- | -- | m | $R_E = 6378137.0$ |
| 重力加速度 | -- | -- | m/s^2 | $g = 9.80665$ |

### obs 8D 内部表示对照

| index | 字段名 | 单位 | 说明 |
|-------|--------|------|------|
| 0 | `x_e_m` | m | ENU 东向坐标 |
| 1 | `y_n_m` | m | ENU 北向坐标 |
| 2 | `z_u_m` | m | ENU 天向坐标 (= altitude_m) |
| 3 | `vx_e_mps` | m/s | 东向速度分量 |
| 4 | `vy_n_mps` | m/s | 北向速度分量 |
| 5 | `vz_u_mps` | m/s | 天向速度分量 |
| 6 | `track_angle_rad_unwrapped` | rad | 航迹角 (unwrapped, 非 wrap 到 $[0, 2\pi)$) |
| 7 | `ground_speed_mps` | m/s | 地面速率 $\sqrt{v_x^2 + v_y^2}$ (不含垂直分量) |

---

*文档版本: v1 | 生成日期: 2026-03-24*
