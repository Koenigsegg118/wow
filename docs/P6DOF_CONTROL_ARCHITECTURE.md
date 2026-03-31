# P6DOF Control Architecture for WOW

## Overview

WOW uses a three-layer control architecture for P6DOF aircraft in AFSIM:

```
Token (dpsi, dalt, dspd)
    |
    +-- lateral_adapter   : dpsi -> bank angle / heading command
    +-- vertical_adapter  : dalt -> vertical speed / G-load command
    +-- energy_adapter    : dspd -> speed command
    |
    v
P6DOF Autopilot (3 independent channels: Lateral / Vertical / Speed)
    |
    v
Flight Control System -> Aerodynamics -> Aircraft Response
```

All three channels are **independent** in the P6DOF autopilot. Setting one does not
interfere with the others (confirmed by code audit of `P6DofAutopilotAction`).

## Profiles

### `p6dof_semantic` (DEFAULT)

```
lateral:   roll_from_token        SetAutopilotRollAngle(phi_cmd)
vertical:  vert_speed_from_token  SetAutopilotVerticalSpeed(vz_cmd)
energy:    GoToSpeed              platform.GoToSpeed(spd_target)
bank_max:  80 deg                 SetBankAngleMax(80) at runtime
```

**Rationale:** Preserves the token's 2s-lookahead `dalt` semantics. The token encodes an
intended altitude change over `H_action_sec = 2s`. Mapping this to vertical speed
(`vz_cmd = (alt_target - alt_current) / H_action_sec`) respects that intent.

### `p6dof_aggressive_turn` (EXPERIMENTAL)

```
lateral:   roll_from_token        SetAutopilotRollAngle(phi_cmd)
vertical:  gload_feedforward      SetPitchGLoad(1/cos(phi))
energy:    GoToSpeed              platform.GoToSpeed(spd_target)
bank_max:  80 deg                 SetBankAngleMax(80) at runtime
```

**Rationale:** Commands the exact G-load needed for a coordinated level turn.
Maximizes altitude hold during high-bank maneuvers. However, under current
FA-LGT constraints this does NOT eliminate altitude loss (see Known Constraints).

## Usage

```bash
# Default profile (p6dof_semantic)
python -m sim.token_sweep_server --vq_ckpt ... --vocab ... --single_token 29

# Explicit profile
python -m sim.token_sweep_server --vq_ckpt ... --vocab ... --profile p6dof_semantic

# Aggressive turn profile
python -m sim.token_sweep_server --vq_ckpt ... --vocab ... --profile p6dof_aggressive_turn

# Override individual settings (profile values as base, CLI overrides)
python -m sim.token_sweep_server --vq_ckpt ... --vocab ... --profile p6dof_semantic --bank_angle_max 70
```

## Lateral Adapter Modes

| Mode | ID | Command | Description |
|------|----|---------|-------------|
| heading_fraction | 0 | TurnToRelativeHeading(fractional_dpsi) | Legacy. PID sees ~2 deg error, produces only ~0.45 deg/s turn. **Not recommended.** |
| heading_lookahead | 1 | TurnToRelativeHeading(full_dpsi) | Better (~3.5 deg/s at bank60). PID saturates, bank limit becomes bottleneck. |
| **roll_from_token** | **2** | SetAutopilotRollAngle(phi_cmd) | **Default.** Directly computes bank from token: `phi = atan(V * psi_dot / g)`. Achieves ~10 deg/s at T29. |

### Why roll_from_token is the default

Experimental results (A1/B1/B2/C1):

| Experiment | Mode | Actual turn rate | vs T29 target (10.3 deg/s) |
|-----------|------|-----------------|---------------------------|
| A1 | heading_fraction + bank60 | 0.45 deg/s | 4% |
| B1 | heading_lookahead + bank60 | 3.5 deg/s | 34% |
| B2 | heading_lookahead + bank80 | 4.4 deg/s | 43% |
| **C1** | **roll_from_token + bank80** | **10.9 deg/s** | **106%** |

## Vertical Adapter Modes

| Mode | ID | Command | Description |
|------|----|---------|-------------|
| alt_hold | 0 | GoToAltitude(target) | Standard altitude hold. Insufficient for high-bank turns. |
| **gload_feedforward** | **1** | SetPitchGLoad(1/cos(phi)) | Coordinated turn G-load. Best altitude hold but still AoA-limited. |
| **vert_speed_from_token** | **2** | SetAutopilotVerticalSpeed(vz_cmd) | Preserves token dalt semantics. Default for `p6dof_semantic`. |
| gload_ff_vz | 3 | SetPitchGLoad(nz_base + Kz*alt_err) | Combined. No significant improvement over mode 1/2. |

### Experimental results (L1-L4, all with roll_from_token + bank80, T29)

| Experiment | Vertical Mode | nz_cmd | g_load actual | Alt loss @30s | Speed loss @30s |
|-----------|--------------|--------|--------------|--------------|----------------|
| L1 | alt_hold | 1.0 | 0.8G | ~2400m | ~0 kt |
| **L2** | **gload_ff** | **4.5G** | **2.9G** | **1764m** | **-75 kt** |
| **L3** | **vert_speed** | **1.0G** | **3.0G** | **1764m** | **-74 kt** |
| L4 | gload+vz | 5.0G | 3.0G | 1818m | -75 kt |

## Telemetry Flags

### `aoa_saturation_flag`

Set to 1 when `|alpha_deg| > 9.5 deg`. Indicates the aircraft is at or near its AoA limit
and cannot produce additional lift/G regardless of autopilot commands.

### `infeasible_level_turn_flag`

Set to 1 when `|phi_cmd| > arccos(1/nz_max_est)` where `nz_max_est = 3.0G`.
This means the commanded bank angle requires more G than the aircraft can produce
for a level (altitude-maintaining) turn. Altitude loss is physically unavoidable.

Current threshold: `arccos(1/3.0) = 70.5 deg`. Any `phi_cmd > 70.5 deg` is flagged.

**Interpretation:** When this flag is set AND the token's dalt does not indicate a
deliberate descent, the lateral and vertical semantics are **mutually infeasible**.
The aircraft will prioritize the lateral command (bank angle) and accept altitude loss.

## Known Constraints (FA-LGT at 30 kft)

1. **AoA saturation at ~10 deg** limits achievable normal G to ~3G
2. **Level turn infeasible above ~70.5 deg bank** (requires > 3G)
3. **phi_cmd ~78 deg** (T29-class token) always produces significant altitude loss (~60 m/s sink rate)
4. **Speed bleeds ~60-90 kt** over 30s of sustained high-bank turn
5. These are **reachability limits of the FA-LGT aircraft model**, not implementation bugs

## Risk Isolation

- `bank_angle_max` is set at **runtime** via `SetBankAngleMax()` — shared FA-LGT config files are NOT modified
- All profile settings flow through `p6dof_profiles.py` and `--profile` CLI argument
- The draw_processor in `2v2_p6dof_lateral_test.txt` is a WOW-specific scenario, not a shared demo
