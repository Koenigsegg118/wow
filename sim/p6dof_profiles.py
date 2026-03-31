"""
P6DOF control profiles for WOW/AFSIM integration.

Each profile defines a complete control architecture:
  - lateral_adapter:   how token dpsi maps to P6DOF lateral command
  - vertical_adapter:  how token dalt maps to P6DOF vertical command
  - energy_adapter:    how token dspd maps to P6DOF speed command
  - bank_angle_max_deg: runtime SetBankAngleMax override (does NOT modify shared FA-LGT config)

Profiles
--------
p6dof_semantic (DEFAULT)
    Lateral:  roll_from_token   — phi_cmd = atan(V * psi_dot / g)
    Vertical: vert_speed_from_token — vz_cmd = (alt_target - alt_current) / H_action_sec
    Energy:   GoToSpeed
    Rationale: preserves token's 2s-lookahead dalt semantics.

p6dof_aggressive_turn (EXPERIMENTAL)
    Lateral:  roll_from_token   — same
    Vertical: gload_feedforward — nz_cmd = 1/cos(phi) for coordinated level turn
    Energy:   GoToSpeed
    Rationale: maximises altitude hold during high-bank turns.
               Under current AoA ~10 deg / nz_actual ~3G constraint,
               turns at phi > ~70 deg still lose altitude; this is a
               reachability limit, not an implementation bug.

Known constraints (FA-LGT at 30 kft)
-------------------------------------
- AoA saturation at ~10 deg limits achievable nz to ~3G
- Level turn at phi > arccos(1/3) ≈ 70.5 deg is physically infeasible
- phi_cmd ~78 deg (T29-class token) will always produce significant altitude loss
- Speed bleeds ~60-90 kt over 30 s of sustained high-bank turn
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class P6DofProfile:
    name: str
    lateral_mode: int       # 0=heading_fraction, 1=heading_lookahead, 2=roll_from_token
    vertical_mode: int      # 0=alt_hold, 1=gload_ff, 2=vert_speed, 3=gload+vz
    bank_angle_max_deg: float
    description: str


PROFILES: dict[str, P6DofProfile] = {
    "p6dof_semantic": P6DofProfile(
        name="p6dof_semantic",
        lateral_mode=2,         # roll_from_token
        vertical_mode=2,        # vert_speed_from_token
        bank_angle_max_deg=80,
        description="Default: roll_from_token + vert_speed_from_token + GoToSpeed",
    ),
    "p6dof_aggressive_turn": P6DofProfile(
        name="p6dof_aggressive_turn",
        lateral_mode=2,         # roll_from_token
        vertical_mode=1,        # gload_feedforward
        bank_angle_max_deg=80,
        description="Experimental: roll_from_token + gload_feedforward + GoToSpeed",
    ),
}

DEFAULT_PROFILE = "p6dof_semantic"


def get_profile(name: str) -> P6DofProfile:
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILES[name]
