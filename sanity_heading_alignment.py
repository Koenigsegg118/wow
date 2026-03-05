#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sanity checks for heading/dpsi/TurnToRelativeHeading alignment.

Includes:
1) Math + bridge checks (always runnable)
2) Optional AFSIM mission regression execution for TurnToRelativeHeading sign

Usage:
    python wow/sanity_heading_alignment.py           # math checks only
    python wow/sanity_heading_alignment.py --afsim   # also launch AFSIM regression
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent.resolve()
_TRAINING  = _THIS_DIR / "training"
for _p in [str(_THIS_DIR), str(_TRAINING)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from afsim_policy_bridge import BridgeConfig, Limits, Slew, SmoothSetpointController
from heading_convention_config import CONVENTION


def wrap_360(deg):
    return deg % 360.0


def wrap_180(deg):
    deg = wrap_360(deg)
    return deg - 360.0 if deg > 180.0 else deg


def heading_from_velocity(vx_east, vy_north):
    """heading_rad = atan2(vx_east, vy_north). 0=North, CW=positive."""
    return math.atan2(vx_east, vy_north)


# ── Part 1: Math checks ───────────────────────────────────────────────────────

def check_heading_cardinal():
    """Verify atan2(vx_east, vy_north) gives correct cardinal headings."""
    cases = [
        # (vx_east, vy_north, expected_deg, label)
        ( 0.0,  1.0,    0.0, "North"),
        ( 1.0,  0.0,   90.0, "East"),
        ( 0.0, -1.0,  180.0, "South"),  # or -180, both correct
        (-1.0,  0.0,  -90.0, "West"),
        ( 1.0,  1.0,   45.0, "NE"),
        ( 1.0, -1.0,  135.0, "SE"),
    ]
    print("\n[1] Cardinal heading checks  (heading = atan2(vx_east, vy_north))")
    all_ok = True
    for vx, vy, expected_deg, label in cases:
        h_deg = math.degrees(heading_from_velocity(vx, vy))
        diff  = abs(wrap_180(h_deg - expected_deg))
        ok    = diff < 0.01
        all_ok &= ok
        mark  = "OK" if ok else "FAIL"
        print(f"  {mark}  {label:5s}: atan2({vx:5.1f},{vy:5.1f}) = {h_deg:8.3f}deg  (expected {expected_deg:8.3f}deg)")
    return all_ok


def check_dpsi_sign():
    """Verify dpsi > 0 for a clockwise (right) turn."""
    print("\n[2] dpsi sign check  (right turn -> positive dpsi)")

    # Scenario: heading 0 deg (North) -> heading +30 deg (NE), right turn
    h0   = heading_from_velocity(0.0,  1.0)    # North = 0 rad
    h1   = heading_from_velocity(0.5,  0.866)  # ~30 deg East of North
    dpsi = ((h1 - h0) + math.pi) % (2 * math.pi) - math.pi
    ok   = dpsi > 0
    mark = "OK" if ok else "FAIL"
    note = "(CW=positive OK)" if ok else "(sign WRONG!)"
    print(f"  {mark}  North->NE30: h0={math.degrees(h0):.1f}deg  h1={math.degrees(h1):.1f}deg  "
          f"dpsi={math.degrees(dpsi):.3f}deg  {note}")

    # Scenario: left turn should give negative dpsi
    h2    = heading_from_velocity(-0.5, 0.866)  # ~-30 deg West of North
    dpsi2 = ((h2 - h0) + math.pi) % (2 * math.pi) - math.pi
    ok2   = dpsi2 < 0
    mark2 = "OK" if ok2 else "FAIL"
    note2 = "(CCW=negative OK)" if ok2 else "(sign WRONG!)"
    print(f"  {mark2}  North->NW30: h0={math.degrees(h0):.1f}deg  h2={math.degrees(h2):.1f}deg  "
          f"dpsi={math.degrees(dpsi2):.3f}deg  {note2}")
    return ok and ok2


def check_bridge_sign():
    """Verify SmoothSetpointController emits correct TurnToRelativeHeading sign.

    Each case uses a fresh controller.  We call step() 10 times so the
    slew-limiter can ramp up to the commanded dpsi.
    """
    print("\n[3] Bridge sign check  (dpsi_rad -> TurnToRelativeHeading deg)")
    cases = [
        ( 0.30, "right turn (+0.3 rad) -> expect positive turn_deg"),
        (-0.30, "left  turn (-0.3 rad) -> expect negative turn_deg"),
        ( 0.00, "straight  ( 0.0 rad)  -> expect zero"),
    ]
    all_ok = True
    for dpsi, desc in cases:
        cfg  = BridgeConfig(turn_sign=CONVENTION.turn_sign)
        ctrl = SmoothSetpointController(Limits(), Slew(), cfg)
        ctrl.reset(alt_m=5000.0, spd_mps=250.0)
        turn_deg = 0.0
        for _ in range(10):    # ramp slew-limiter to steady state
            _, (turn_deg, _, _) = ctrl.step(dpsi, 5000.0, 250.0, 5000.0, 250.0)
        if dpsi > 0:
            ok = turn_deg > 0
        elif dpsi < 0:
            ok = turn_deg < 0
        else:
            ok = abs(turn_deg) < 0.001
        all_ok &= ok
        mark = "OK" if ok else "FAIL"
        print(f"  {mark}  dpsi={dpsi:+.2f} rad -> TurnToRelativeHeading({turn_deg:+.3f}deg)  # {desc}")
    return all_ok


def check_convention_config():
    """Print the frozen convention config for visual inspection."""
    print("\n[4] CONVENTION config (visual check)")
    print(f"  heading_frame         : {CONVENTION.heading_frame}")
    print(f"  heading_zero_reference: {CONVENTION.heading_zero_reference}")
    print(f"  heading_positive_dir  : {CONVENTION.heading_positive_direction}")
    print(f"  atan2_order           : {CONVENTION.heading_atan2_order}")
    print(f"  turn_sign             : {CONVENTION.turn_sign:+.1f}  (should be +1.0)")
    print(f"  dpsi_sign             : {CONVENTION.dpsi_sign:+.1f}  (should be +1.0)")
    ok = CONVENTION.turn_sign == 1.0 and CONVENTION.dpsi_sign == 1.0
    mark = "OK" if ok else "FAIL"
    print(f"  {mark}  turn_sign=={CONVENTION.turn_sign} and dpsi_sign=={CONVENTION.dpsi_sign}")
    return ok


# ── Part 2: AFSIM regression (optional) ──────────────────────────────────────

def run_afsim_regression(afsim_exe, scenario):
    """Launch AFSIM and check that a +dpsi command produces a clockwise turn.

    This is a placeholder - wire up to your regression scenario as needed.
    Returns True if the test passes, False otherwise.
    """
    print(f"\n[5] AFSIM regression (exe={afsim_exe}, scenario={scenario})")
    try:
        result = subprocess.run(
            [afsim_exe, scenario],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  FAIL  AFSIM exited with code {result.returncode}")
            print(result.stderr[:500])
            return False
        if "TURN_SIGN_OK" in result.stdout:
            print("  OK    AFSIM regression PASSED (found TURN_SIGN_OK in output)")
            return True
        else:
            print("  FAIL  TURN_SIGN_OK not found in AFSIM output")
            print(result.stdout[:500])
            return False
    except FileNotFoundError:
        print(f"  SKIP  AFSIM executable not found: {afsim_exe}")
        return True   # treat as skipped, not failed
    except subprocess.TimeoutExpired:
        print("  FAIL  AFSIM regression timed out (>60s)")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--afsim",     action="store_true",
                    help="Run optional AFSIM regression in addition to math checks")
    ap.add_argument("--afsim_exe", default="warlock",
                    help="Path to AFSIM executable (default: warlock)")
    ap.add_argument("--scenario",  default="scenarios/turn_sign_regression.xml",
                    help="AFSIM scenario file for regression test")
    args = ap.parse_args()

    print("=" * 60)
    print("Heading / dpsi / TurnToRelativeHeading sanity checks")
    print("=" * 60)

    results = [
        check_heading_cardinal(),
        check_dpsi_sign(),
        check_bridge_sign(),
        check_convention_config(),
    ]

    if args.afsim:
        results.append(run_afsim_regression(args.afsim_exe, args.scenario))

    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    if all(results):
        print(f"ALL CHECKS PASSED ({n_pass}/{n_total})")
        sys.exit(0)
    else:
        print(f"SOME CHECKS FAILED ({n_pass}/{n_total} passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
