#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AFSIM policy bridge (metric): clamp + physically-consistent slew limiter.

Action vector semantics
-----------------------
The model outputs three physical quantities per control tick:

  dpsi_rad    Relative track-angle change from **now** to the lookahead horizon
              ``t + horizon_steps``.  Units are radians, positive = clockwise /
              right turn when viewed from above.

  alt_sp_m    Altitude setpoint in meters at the same lookahead horizon.

  spd_sp_mps  Speed setpoint in meters per second at the same lookahead horizon.

AFSIM emit:
  TurnToRelativeHeading(deg(dpsi_cmd_tick))  -- positive = clockwise / right turn
  GoToAltitude(alt_cmd_m)
  GoToSpeed(spd_cmd_mps)

Runtime interpretation of dpsi_rad
----------------------------------
The producer labels ``dpsi_rad`` as a **lookahead delta** over
``horizon_steps / control_frequency_hz`` seconds.  The bridge therefore must
convert that horizon-wide delta into a single control-tick command before
sending it to AFSIM.

Current interpretation (locked by runtime tests):

  * ``dpsi_rad`` is relative to the current velocity-derived track angle.
  * ``horizon_steps`` and ``control_frequency_hz`` are active runtime inputs,
    not metadata-only fields.
  * The bridge consumes a ``dt / horizon_seconds`` fraction of the lookahead
    delta on each tick, capped at the full delta when ``dt >= horizon_seconds``.
  * ``alt_sp_m`` and ``spd_sp_mps`` remain absolute lookahead setpoints and are
    slewed directly in physical units.

Slew parameters are defined as **per-second** rates.  The caller MUST supply
both the actual control interval ``dt`` and the artifact lookahead horizon to
``step()`` so the control mapping stays physically consistent across sim tick
frequencies.

Default rates are calibrated so that ``dt=0.5`` (2 Hz) with a 2.0 s lookahead
(``horizon_steps=4`` at 2 Hz) reproduces the legacy per-tick turn magnitude:
lookahead ``dpsi_rad=0.35`` maps to a per-tick command of about ``0.087`` rad.
"""

import math
from dataclasses import dataclass


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


@dataclass
class Limits:
    dpsi_max_rad: float = 1.05
    alt_min_m: float = 200.0
    alt_max_m: float = 20000.0
    spd_min_mps: float = 120.0
    spd_max_mps: float = 650.0


@dataclass
class Slew:
    """Maximum change rates in physical units **per second**."""
    dpsi_rate_rad_per_s: float = 0.52     # ~30 deg/s
    alt_rate_m_per_s: float = 160.0       # m/s
    spd_rate_mps_per_s: float = 30.0      # (m/s)/s


@dataclass
class Cmd:
    dpsi_rad: float
    alt_m: float
    spd_mps: float


@dataclass
class BridgeConfig:
    # +1 keep sign, -1 flip sign before emitting TurnToRelativeHeading(deg).
    turn_sign: float = 1.0


class SmoothSetpointController:
    """Clamp + slew-rate limiter for AFSIM P6DOF mover commands.

    Pipeline for each tick (called from ``step()``):

    1. **Absolute clamp** -- clamp the lookahead ``dpsi_rad`` and the absolute
       alt / spd setpoints within safe bounds.
    2. **Lookahead-to-tick map** -- consume ``dt / horizon_seconds`` of the
       lookahead ``dpsi_rad`` on this tick.
    3. **Slew-rate limit** -- per-second rates * ``dt`` gives per-tick budget.
    4. **Sign mapping** -- ``turn_sign * dpsi_cmd_tick`` converted to degrees.
    5. **Emit** -- ``(turn_deg, alt_cmd_m, spd_cmd_mps)``.

    ``dpsi_rad`` is always interpreted as a lookahead track-angle delta,
    relative to the current velocity-derived track angle.
    """

    def __init__(self, limits: Limits = Limits(), slew: Slew = Slew(),
                 config: BridgeConfig = BridgeConfig()):
        self.lim = limits
        self.slew = slew
        self.cfg = config
        self.prev: Cmd | None = None
        if self.cfg.turn_sign not in (-1.0, 1.0):
            raise ValueError("BridgeConfig.turn_sign must be +1 or -1")

    def reset(self, alt_m: float, spd_mps: float) -> None:
        self.prev = Cmd(0.0, alt_m, spd_mps)

    def step(self, dpsi_rad: float, alt_sp_m: float, spd_sp_mps: float,
             alt_m: float, spd_mps: float, *, dt: float,
             lookahead_horizon_s: float
             ) -> tuple[Cmd, tuple[float, float, float]]:
        """Apply clamp + slew for one tick.

        ``dpsi_rad`` is the lookahead delta over ``lookahead_horizon_s``.
        ``dt`` is the current control interval in seconds.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if lookahead_horizon_s <= 0.0:
            raise ValueError(
                f"lookahead_horizon_s must be positive, got {lookahead_horizon_s}"
            )
        if self.prev is None:
            self.reset(alt_m, spd_mps)

        dpsi_lookahead = clamp(
            dpsi_rad, -self.lim.dpsi_max_rad, +self.lim.dpsi_max_rad,
        )
        alt_sp = clamp(alt_sp_m, self.lim.alt_min_m, self.lim.alt_max_m)
        spd_sp = clamp(spd_sp_mps, self.lim.spd_min_mps, self.lim.spd_max_mps)

        horizon_fraction = min(dt / lookahead_horizon_s, 1.0)
        dpsi_tick_target = dpsi_lookahead * horizon_fraction

        d_dpsi = self.slew.dpsi_rate_rad_per_s * dt
        d_alt = self.slew.alt_rate_m_per_s * dt
        d_spd = self.slew.spd_rate_mps_per_s * dt

        dpsi_cmd = clamp(
            dpsi_tick_target,
            self.prev.dpsi_rad - d_dpsi,
            self.prev.dpsi_rad + d_dpsi,
        )
        alt_cmd = clamp(alt_sp, self.prev.alt_m - d_alt,
                             self.prev.alt_m + d_alt)
        spd_cmd = clamp(spd_sp, self.prev.spd_mps - d_spd,
                             self.prev.spd_mps + d_spd)

        self.prev = Cmd(dpsi_cmd, alt_cmd, spd_cmd)
        turn_deg = rad2deg(self.cfg.turn_sign * dpsi_cmd)
        return self.prev, (turn_deg, alt_cmd, spd_cmd)
