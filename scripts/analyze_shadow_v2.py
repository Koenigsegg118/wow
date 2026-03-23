#!/usr/bin/env python3
"""Analyze token_shadow_v2.jsonl — quick statistics."""

import json
import sys
from pathlib import Path

import numpy as np

log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/token_shadow_v2.jsonl"
records = [json.loads(l) for l in Path(log_path).read_text().strip().split("\n")]

tokens = [r for r in records if r["event"] == "new_token"]
ticks = [r for r in records if r["event"] == "tick"]

print(f"={'=' * 60}")
print(f"  Shadow v2 Analysis: {log_path}")
print(f"={'=' * 60}")
print(f"Total log lines     : {len(records)}")
print(f"  new_token events  : {len(tokens)}")
print(f"  tick events       : {len(ticks)}")
print(f"Sim time range      : {records[0]['sim_time']:.1f}s — {records[-1]['sim_time']:.1f}s")
print(f"Contract version    : {records[0].get('contract_version', 'N/A')}")
print(f"History len         : {records[0].get('history_len', 'N/A')}")

platforms = sorted(set(r["platform_idx"] for r in records))
print(f"Platforms observed  : {platforms}")

print(f"\n--- Confidence ---")
confs = [r["confidence"] for r in tokens]
print(f"  mean              : {np.mean(confs):.4f}")
print(f"  median            : {np.median(confs):.4f}")
print(f"  min / max         : {np.min(confs):.4f} / {np.max(confs):.4f}")
print(f"  std               : {np.std(confs):.4f}")
for thresh in [0.35, 0.25, 0.15]:
    pct = sum(c < thresh for c in confs) / len(confs) * 100
    print(f"  < {thresh:.2f}             : {pct:.1f}%")

print(f"\n--- History readiness ---")
ready_count = sum(1 for r in tokens if r.get("history_ready", False))
print(f"  history_ready     : {ready_count}/{len(tokens)} "
      f"({ready_count/len(tokens)*100:.1f}%)")

print(f"\n--- Guardrail dry-run (on new_token events) ---")
gr_trigger = sum(1 for r in tokens if r.get("guardrail_would_trigger", False))
print(f"  would trigger     : {gr_trigger}/{len(tokens)} "
      f"({gr_trigger/len(tokens)*100:.1f}%)")
reasons = {}
for r in tokens:
    if r.get("guardrail_would_trigger"):
        reason = r.get("guardrail_reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
if reasons:
    for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {cnt}")
else:
    print(f"    (none triggered)")

print(f"\n--- Token distribution (top 10) ---")
dense_tokens = [r["predicted_dense_token"] for r in tokens]
unique, counts = np.unique(dense_tokens, return_counts=True)
order = np.argsort(-counts)
for i in order[:10]:
    pct = counts[i] / len(dense_tokens) * 100
    print(f"  token {unique[i]:2d}: {counts[i]:6d} ({pct:5.1f}%)")
print(f"  unique tokens used: {len(unique)}/{30}")

print(f"\n--- Delta: token suggested vs primary actual ---")
dt_all = [abs(r["delta_turn_deg"]) for r in records]
da_all = [abs(r["delta_alt_m"]) for r in records]
ds_all = [abs(r["delta_spd_mps"]) for r in records]
print(f"  |delta_turn|  mean={np.mean(dt_all):7.2f}°   "
      f"median={np.median(dt_all):7.2f}°   max={np.max(dt_all):7.2f}°")
print(f"  |delta_alt|   mean={np.mean(da_all):7.1f}m   "
      f"median={np.median(da_all):7.1f}m   max={np.max(da_all):7.1f}m")
print(f"  |delta_spd|   mean={np.mean(ds_all):7.2f}m/s "
      f"median={np.median(ds_all):7.2f}m/s max={np.max(ds_all):7.2f}m/s")

print(f"\n--- Per-platform breakdown ---")
for plat in platforms:
    plat_tokens = [r for r in tokens if r["platform_idx"] == plat]
    plat_all = [r for r in records if r["platform_idx"] == plat]
    if not plat_tokens:
        print(f"  platform {plat}: no token predictions")
        continue
    pc = [r["confidence"] for r in plat_tokens]
    pdt = [abs(r["delta_turn_deg"]) for r in plat_all]
    pda = [abs(r["delta_alt_m"]) for r in plat_all]
    pds = [abs(r["delta_spd_mps"]) for r in plat_all]
    pgr = sum(1 for r in plat_tokens if r.get("guardrail_would_trigger"))
    print(f"  platform {plat}: {len(plat_tokens)} predictions, "
          f"conf={np.mean(pc):.3f}, "
          f"|dt|={np.mean(pdt):.2f}° |da|={np.mean(pda):.1f}m |ds|={np.mean(pds):.2f}m/s, "
          f"guardrail={pgr}")

print(f"\n--- Temporal phases (first/mid/last third) ---")
sim_times = [r["sim_time"] for r in tokens]
t_min, t_max = min(sim_times), max(sim_times)
t_third = (t_max - t_min) / 3
for phase_name, lo, hi in [
    ("early", t_min, t_min + t_third),
    ("mid", t_min + t_third, t_min + 2 * t_third),
    ("late", t_min + 2 * t_third, t_max + 1),
]:
    phase = [r for r in tokens if lo <= r["sim_time"] < hi]
    if not phase:
        continue
    pc = [r["confidence"] for r in phase]
    pdense = [r["predicted_dense_token"] for r in phase]
    unique_t = len(set(pdense))
    print(f"  {phase_name:5s} ({lo:.0f}-{hi:.0f}s): "
          f"n={len(phase)}, conf={np.mean(pc):.3f}, "
          f"unique_tokens={unique_t}")

print(f"\n{'=' * 60}")
