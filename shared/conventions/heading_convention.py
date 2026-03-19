#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central heading/turn convention config for dataset + bridge alignment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadingConvention:
    # Dataset heading from ENU velocity: heading_rad = atan2(vx_east, vy_north)
    # This means 0=North, +90=East, and positive rotation is clockwise.
    heading_frame: str = "ENU_velocity_to_heading"
    heading_zero_reference: str = "north"
    heading_positive_direction: str = "clockwise"
    heading_atan2_order: str = "atan2(vx_east, vy_north)"

    # +1 means keep dpsi sign as-is from dataset/model; -1 flips sign.
    # Use this if integration reveals opposite TurnToRelativeHeading semantics.
    turn_sign: float = 1.0
    dpsi_sign: float = 1.0


CONVENTION = HeadingConvention()
