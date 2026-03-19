# Units  (schema 0.2.0)

All contract fields use SI / metric units.

---

## Episode state fields  (`shared/schema/episode_schema.py`)

These are the fields stored in episode JSON produced by `sim/export_episode.py`.
They are the **sim ↔ inference deployment contract**, not the direct model input.

| Field | Unit | Notes |
|-------|------|-------|
| `x_e_m`, `y_n_m`, `z_u_m` | m | ENU position |
| `vx_e_mps`, `vy_n_mps`, `vz_u_mps` | m/s | ENU velocity |
| `heading_rad` | rad | Wrapped track angle in `[-π, π]`. Computed as `atan2(vx_east, vy_north)`. This is **not** body heading from INS/IMU — it is the velocity-derived track direction. Value is wrapped; runtime adapters must unwrap before feeding to the model (see coordinate_frames.md). |
| `alive` | bool | — |

---

## Model observation fields  (`ml/dataset_default_config.py :: OBS_FIELDS`)

These are the features the Transformer model was trained on.
They appear in `policy.yaml :: obs_spec.field_order`.

| Field | Unit | Notes |
|-------|------|-------|
| `x_e_m` | m | ENU east position |
| `y_n_m` | m | ENU north position |
| `z_u_m` | m | ENU up (altitude) |
| `vx_e_mps` | m/s | ENU east velocity |
| `vy_n_mps` | m/s | ENU north velocity |
| `vz_u_mps` | m/s | ENU up velocity |
| `track_angle_rad_unwrapped` | rad | **Continuously unwrapped** velocity-derived track angle. Source: `atan2(vx_east, vy_north)`; 0 = North, CW positive. Must be unwrapped across the observation window — do **not** pass wrapped `[-π, π]` values. See coordinate_frames.md §Track Angle. |
| `ground_speed_mps` | m/s | **2-D horizontal ground speed** = `sqrt(vx_east² + vy_north²)`. Does NOT include vertical component `vz`. |

---

## Episode action fields  (`shared/schema/action_schema.py`)

Used in episode JSON `actions[].commands[]`. Represents **one-step** deltas / setpoints
at the `time_step_s` interval.

| Field | Unit | Notes |
|-------|------|-------|
| `dpsi_rad` | rad | Track-angle change over **one** control step (= `time_step_s`). CW positive. |
| `alt_sp_m` | m | Altitude setpoint (AGL metres) |
| `spd_sp_mps` | m/s | Ground speed setpoint |

---

## Model action fields  (artifact `policy.yaml :: action_spec`)

The model outputs a **lookahead setpoint**, not a one-step delta.
These fields appear in `policy.yaml :: action_spec.field_order`.

| Field | Unit | Notes |
|-------|------|-------|
| `dpsi_rad` | rad | Track-angle change from **now** to `horizon_steps` control steps ahead (`horizon_steps / control_frequency_hz` seconds). CW positive. `angle_reference: track_angle`. |
| `alt_sp_m` | m | Altitude setpoint at lookahead horizon |
| `spd_sp_mps` | m/s | Ground speed setpoint at lookahead horizon |

`horizon_steps` and `control_frequency_hz` are mandatory fields in every `policy.yaml`
(introduced in schema 0.2.0). The runtime must consume them when mapping `dpsi_rad`
to each control tick: the current tick should apply `dt / (horizon_steps / control_frequency_hz)`
of the lookahead delta, capped at the full delta when `dt` exceeds the lookahead horizon.

---

## Time fields

| Field | Unit | Notes |
|-------|------|-------|
| `time_step_s` | s | Fixed control interval |
| `frames[i].t_s` | s | Time since episode start |
| `actions[i].t_s` | s | Action timestamp since episode start |
