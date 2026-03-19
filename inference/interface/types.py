"""Runtime IO types aligned with shared/schema episode contract fields."""

from __future__ import annotations

from typing import TypedDict


class ObservationState(TypedDict):
    entity_id: str
    alive: bool
    x_e_m: float
    y_n_m: float
    z_u_m: float
    vx_e_mps: float
    vy_n_mps: float
    vz_u_mps: float
    heading_rad: float


class ActionCommand(TypedDict):
    entity_id: str
    dpsi_rad: float
    alt_sp_m: float
    spd_sp_mps: float
