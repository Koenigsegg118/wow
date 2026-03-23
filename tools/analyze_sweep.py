#!/usr/bin/env python3
"""Analyze token_sweep.jsonl — transient vs steady-state turn rate analysis.

Reads the structured JSONL log from token_sweep_server.py and produces:
  1. Per-token summary with before/after state changes
  2. Transient (0-2s) vs steady-state (2s+) turn rate breakdown
  3. Comparison with theoretical G-limited turn rate
  4. Cross-check: token intent vs AFSIM actual response

Usage:
    python analyze_sweep.py [path/to/token_sweep.jsonl]
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

GRAVITY = 9.81
LOG = Path(__file__).resolve().parent.parent / "logs" / "token_sweep.jsonl"


def _wrap_heading_diff(h1, h0):
    d = h1 - h0
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(LOG)
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]

    if not recs:
        print("No records found.")
        return

    print(f"Log: {path}")
    print(f"Total records: {len(recs)}")
    print(f"Sim time: {recs[0]['sim_time']:.1f}s ~ {recs[-1]['sim_time']:.1f}s")

    # detect exec_mode and g_mode from first record
    r0 = recs[0]
    exec_mode = r0.get("exec_mode", "unknown")
    g_mode = r0.get("g_mode", "unknown")
    h_action_sec = r0.get("H_action_sec", 2.0)
    print(f"Exec mode: {exec_mode}")
    print(f"G mode: {g_mode}")
    print(f"H_action_sec: {h_action_sec}s")
    print()

    # filter to platform 0 only
    p0 = [r for r in recs if r.get("platform_idx") == 0]
    if not p0:
        p0 = recs

    by_token = defaultdict(list)
    for r in p0:
        by_token[r["dense_token"]].append(r)

    # ── Section 1: Per-token summary ──
    print("=" * 120)
    print("Section 1: Per-token state change summary")
    print("=" * 120)

    hdr = (f"{'Tok':>4} {'raw':>3}  "
           f"{'N_act':>5} {'N_stl':>5}  "
           f"{'HdgBef':>8} {'HdgAft':>8} {'dHdg':>7}  "
           f"{'AltBef':>8} {'AltAft':>8} {'dAlt':>7}  "
           f"{'SpdBef':>7} {'SpdAft':>7} {'dSpd':>6}  "
           f"{'AvgTurn':>8}")
    print(hdr)
    print("-" * len(hdr))

    for did in sorted(by_token.keys()):
        rows = by_token[did]
        act = [r for r in rows if r["event"] == "token_action"]
        stl = [r for r in rows if r["event"] == "settle"]
        if not act:
            continue

        first = act[0]
        final = rows[-1]

        hb = first["afsim_state"]["heading_deg"]
        ha = final["afsim_state"]["heading_deg"]
        dh = _wrap_heading_diff(ha, hb)

        ab = first["afsim_state"]["alt_m"]
        aa = final["afsim_state"]["alt_m"]

        sb = first["afsim_state"]["speed_mps"]
        sa = final["afsim_state"]["speed_mps"]

        cmds = [r.get("abs_command", r.get("abs_command", [0, 0, 0])) for r in act]
        avg_t = sum(c[0] for c in cmds) / max(len(cmds), 1)

        raw = first["raw_token"]
        print(f" T{did:>2d}  r{raw:>2d}  "
              f"{len(act):>5d} {len(stl):>5d}  "
              f"{hb:>8.1f} {ha:>8.1f} {dh:>+7.1f}  "
              f"{ab:>8.1f} {aa:>8.1f} {aa-ab:>+7.1f}  "
              f"{sb:>7.1f} {sa:>7.1f} {sa-sb:>+6.1f}  "
              f"{avg_t:>+8.3f}")

    # ── Section 2: Transient vs Steady-state turn rate ──
    print()
    print("=" * 120)
    print("Section 2: Transient (0-2s) vs Steady-state (2s+) turn rate analysis")
    print("=" * 120)

    hdr2 = (f"{'Tok':>4}  "
            f"{'dpsi°/2s':>8}  "
            f"{'ψ̇_cmd°/s':>9}  "
            f"{'n_cmd':>5}  "
            f"{'ψ̇_theory':>9}  "
            f"{'ψ̇_trans':>8}  "
            f"{'ψ̇_steady':>9}  "
            f"{'Δhdg_10s':>8}  "
            f"{'ratio_th':>8}  "
            f"{'ratio_tok':>9}  "
            f"{'diag':>12}")
    print(hdr2)
    print("-" * len(hdr2))

    for did in sorted(by_token.keys()):
        rows = by_token[did]
        act = [r for r in rows if r["event"] == "token_action"]
        if not act:
            continue

        t0 = act[0]["sim_time"]
        first_hdg = act[0]["afsim_state"]["heading_deg"]

        dpsi_deg_2s = act[0].get("psi_dot_cmd_deg_s", 0.0) * h_action_sec
        psi_dot_cmd = act[0].get("psi_dot_cmd_deg_s", 0.0)
        n_cmd_val = act[0].get("n_cmd", 3.0)
        theory_rate = act[0].get("turn_rate_theory_deg_s", 0.0)

        # compute measured turn rates in transient and steady windows
        transient_rates = []
        steady_rates = []
        prev_hdg = first_hdg
        prev_t = t0

        cumulative_hdg = 0.0
        hdg_at_10s = None

        for i in range(1, len(act)):
            r = act[i]
            t = r["sim_time"]
            hdg = r["afsim_state"]["heading_deg"]
            dt_tick = t - prev_t

            if dt_tick > 0.001:
                dh = _wrap_heading_diff(hdg, prev_hdg)
                rate = dh / dt_tick

                elapsed = t - t0
                if elapsed <= 2.0:
                    transient_rates.append(rate)
                else:
                    steady_rates.append(rate)

                cumulative_hdg += dh
                if hdg_at_10s is None and elapsed >= 10.0:
                    hdg_at_10s = cumulative_hdg

            prev_hdg = hdg
            prev_t = t

        if hdg_at_10s is None:
            hdg_at_10s = cumulative_hdg

        avg_trans = (sum(transient_rates) / len(transient_rates)
                     if transient_rates else 0.0)
        avg_steady = (sum(steady_rates) / len(steady_rates)
                      if steady_rates else 0.0)

        # ratios
        ratio_theory = (abs(avg_steady) / abs(theory_rate)
                        if abs(theory_rate) > 0.01 else float('nan'))
        ratio_token = (abs(avg_steady) / abs(psi_dot_cmd)
                       if abs(psi_dot_cmd) > 0.01 else float('nan'))

        # diagnosis
        if abs(psi_dot_cmd) < 0.5:
            diag = "straight"
        elif math.isnan(ratio_theory):
            diag = "no_theory"
        elif ratio_theory > 0.85:
            diag = "TRACKING_OK"
        elif ratio_theory > 0.5 and abs(avg_trans) < abs(avg_steady) * 0.7:
            diag = "ROLL_IN_SLOW"
        elif ratio_theory < 0.5 and abs(avg_steady) < abs(avg_trans):
            diag = "TASK_RESET?"
        elif ratio_theory < 0.5:
            diag = "G_LIMIT"
        else:
            diag = "MIXED"

        print(f" T{did:>2d}  "
              f"{dpsi_deg_2s:>+8.1f}  "
              f"{psi_dot_cmd:>+9.2f}  "
              f"{n_cmd_val:>5.1f}  "
              f"{theory_rate:>+9.2f}  "
              f"{avg_trans:>+8.2f}  "
              f"{avg_steady:>+9.2f}  "
              f"{hdg_at_10s:>+8.1f}  "
              f"{ratio_theory:>8.2f}  "
              f"{ratio_token:>9.2f}  "
              f"{diag:>12}")

    # ── Section 3: Cross-check intent vs actual ──
    print()
    print("=" * 120)
    print("Section 3: Cross-check — token action intent vs AFSIM actual response")
    print("=" * 120)
    print(f"{'Tok':>4}  "
          f"{'IntentΔHdg':>10} {'ActualΔHdg':>10} {'Ratio':>6}  "
          f"{'IntentΔAlt':>10} {'ActualΔAlt':>10} {'Ratio':>6}  "
          f"{'IntentΔSpd':>10} {'ActualΔSpd':>10} {'Ratio':>6}")
    print("-" * 105)

    for did in sorted(by_token.keys()):
        rows = by_token[did]
        act = [r for r in rows if r["event"] == "token_action"]
        if not act:
            continue

        # intent from first row's action (single lookahead value, not accumulated)
        first_rel = act[0].get("rel_action", [0, 0, 0])
        intent_turn = math.degrees(first_rel[0])
        intent_dalt = first_rel[1]
        intent_dspd = first_rel[2]

        first_st = act[0]["afsim_state"]
        last_st = act[-1]["afsim_state"]

        duration = act[-1]["sim_time"] - act[0]["sim_time"]

        actual_dhdg = _wrap_heading_diff(
            last_st["heading_deg"], first_st["heading_deg"])
        actual_dalt = last_st["alt_m"] - first_st["alt_m"]
        actual_dspd = last_st["speed_mps"] - first_st["speed_mps"]

        def ratio_str(intent, actual):
            if abs(intent) < 0.5 and abs(actual) < 0.5:
                return "  ~0"
            if abs(intent) < 0.01:
                return " n/a"
            r = actual / intent
            return f"{r:>5.2f}x"

        t_r = ratio_str(intent_turn, actual_dhdg)
        a_r = ratio_str(intent_dalt, actual_dalt)
        s_r = ratio_str(intent_dspd, actual_dspd)

        print(f" T{did:>2d}  "
              f"{intent_turn:>+10.2f} {actual_dhdg:>+10.2f} {t_r}  "
              f"{intent_dalt:>+10.1f} {actual_dalt:>+10.1f} {a_r}  "
              f"{intent_dspd:>+10.2f} {actual_dspd:>+10.2f} {s_r}")

    # ── Section 4: Execution config summary ──
    print()
    print("=" * 80)
    print("Section 4: Execution configuration")
    print("=" * 80)
    print(f"  exec_mode      : {exec_mode}")
    print(f"  g_mode         : {g_mode}")
    print(f"  H_action_sec   : {h_action_sec}s")
    dt_sample = recs[1]["sim_time"] - recs[0]["sim_time"] if len(recs) > 1 else 0
    print(f"  dt_cmd (meas.) : {dt_sample:.3f}s")
    print(f"  horizon_frac   : {dt_sample / h_action_sec:.3f}" if h_action_sec > 0 else "")
    n_tokens = len(set(r["dense_token"] for r in p0))
    print(f"  tokens tested  : {n_tokens}")


if __name__ == "__main__":
    main()
