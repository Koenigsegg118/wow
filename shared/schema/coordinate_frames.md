# Coordinate Frames  (schema 0.2.0)

---

## ENU convention

All position and velocity fields use **ENU** (East-North-Up):

| Axis | Positive direction |
|------|--------------------|
| `x_e_m` / `vx_e_mps` | East |
| `y_n_m` / `vy_n_mps` | North |
| `z_u_m` / `vz_u_mps` | Up |

When consuming data from NED systems:
- `z_u_m   = -z_d_m`
- `vz_u_mps = -vz_d_mps`

---

## Track angle vs body heading

> **Critical distinction** (schema 0.2.0+)

### Track angle  (`track_angle_rad_unwrapped`)

This project uses **velocity-derived track angle** as the angular observation feature,
**not** the body (fuselage) heading from an INS or gyro.

- **Definition**: `track_angle = atan2(vx_east, vy_north)`
- **Reference**: 0 rad → North
- **Positive rotation**: clockwise (North → East is positive)
- **Field name in model input**: `track_angle_rad_unwrapped`
- **Field name in episode JSON**: `heading_rad` (wrapped, same formula; runtime must unwrap)

### Body heading  (`body_heading_deg`)

Raw AFSIM telemetry field (degrees, INS-derived).
- This value is **not** a direct model input feature.
- It may be used by a sim adapter internally to cross-check track angle,
  but must not be passed to `PolicyRuntime.predict()` as a substitute for
  `track_angle_rad_unwrapped`.

---

## Track angle unwrap convention

The model was trained on a **continuously unwrapped** track angle sequence.
Wrapping at the `[-π, π]` boundary would create artificial discontinuities
that degrade inference quality.

### Runtime requirement

Every runtime adapter (e.g. `sim/bc_policy_socket_server.py`) must maintain
a running unwrap state across the observation window:

```
heading_now  = atan2(vx_east, vy_north)          # wrapped, in [-π, π]
heading_unwrapped = unwrap_with_prev(heading_now, prev_heading_unwrapped)
```

Where `unwrap_with_prev` corrects jumps larger than `π`:

```python
def unwrap_with_prev(curr, prev):
    if prev is None:
        return curr
    d = curr - prev
    if   d >  math.pi: curr -= 2.0 * math.pi
    elif d < -math.pi: curr += 2.0 * math.pi
    return curr
```

**Never** replace `track_angle_rad_unwrapped` with the raw `heading_rad` field
from an episode JSON without unwrapping it across the window first.

---

## Ground speed

`ground_speed_mps` = **2-D horizontal ground speed only**:

```
ground_speed_mps = sqrt(vx_east² + vy_north²)
```

The vertical component `vz_u_mps` is excluded. This matches the training
pipeline in `ml/acmi_to_dt_dataset_smooth.py`.

---

## NED interop

When consuming data from systems that report NED:
- `z_u_m = -z_d_m`
- `vz_u_mps = -vz_d_mps`

The shared episode contract stores only ENU-aligned values.
