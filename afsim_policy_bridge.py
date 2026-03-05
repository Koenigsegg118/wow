#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AFSIM policy bridge (metric): clamp + slew limiter.

Model output (metric/radians):
  dpsi_rad, alt_sp_m, spd_sp_mps

AFSIM emit (common convention):
  TurnToRelativeHeading(deg(dpsi_cmd))
  GoToAltitude(alt_cmd_m)
  GoToSpeed(spd_cmd_mps)

Call at 2 Hz (dt=0.5s) for the default slew parameters.
"""

import math
from dataclasses import dataclass
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def rad2deg(r):
    return r * 180.0 / math.pi

@dataclass
class Limits:
    dpsi_max_rad: float = 0.35    # ~20 deg per tick
    alt_min_m: float = 200.0
    alt_max_m: float = 20000.0
    spd_min_mps: float = 120.0
    spd_max_mps: float = 400.0

@dataclass
class Slew:
    dpsi_rate_rad: float = 0.087  # ~5 deg per tick
    alt_rate_m: float = 80.0
    spd_rate_mps: float = 7.0

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
    def __init__(self, limits: Limits = Limits(), slew: Slew = Slew(), config: BridgeConfig = BridgeConfig()):
        self.lim = limits
        self.slew = slew
        self.cfg = config
        self.prev: Cmd | None = None
        if self.cfg.turn_sign not in (-1.0, 1.0):
            raise ValueError("BridgeConfig.turn_sign must be +1 or -1")

    def reset(self, alt_m: float, spd_mps: float):
        self.prev = Cmd(0.0, alt_m, spd_mps)

    def step(self, dpsi_rad: float, alt_sp_m: float, spd_sp_mps: float,
             alt_m: float, spd_mps: float):
        if self.prev is None:
            self.reset(alt_m, spd_mps)

        dpsi = clamp(dpsi_rad, -self.lim.dpsi_max_rad, +self.lim.dpsi_max_rad)
        alt_sp = clamp(alt_sp_m, self.lim.alt_min_m, self.lim.alt_max_m)
        spd_sp = clamp(spd_sp_mps, self.lim.spd_min_mps, self.lim.spd_max_mps)

        dpsi_cmd = clamp(dpsi, self.prev.dpsi_rad - self.slew.dpsi_rate_rad,
                              self.prev.dpsi_rad + self.slew.dpsi_rate_rad)
        alt_cmd  = clamp(alt_sp, self.prev.alt_m - self.slew.alt_rate_m,
                               self.prev.alt_m + self.slew.alt_rate_m)
        spd_cmd  = clamp(spd_sp, self.prev.spd_mps - self.slew.spd_rate_mps,
                               self.prev.spd_mps + self.slew.spd_rate_mps)

        self.prev = Cmd(dpsi_cmd, alt_cmd, spd_cmd)
        turn_deg = rad2deg(self.cfg.turn_sign * dpsi_cmd)
        return self.prev, (turn_deg, alt_cmd, spd_cmd)
