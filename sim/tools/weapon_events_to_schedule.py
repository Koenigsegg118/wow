#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile extracted weapon events into an AFSIM include file with timed fire commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def map_weapon_name(missile_name: str) -> str | None:
    name = (missile_name or "").upper()
    if name.startswith("AIM_120"):
        return "fox3"
    if name.startswith("AIM_9"):
        return "fox2"
    return None


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def load_slot_start_times(slots_meta_path: str | Path | None) -> dict[str, float]:
    if not slots_meta_path:
        return {}
    payload = json.loads(Path(slots_meta_path).read_text(encoding="utf-8"))
    slot_times = {}
    for slot, info in payload.get("slots", {}).items():
        if "source_start_time" in info:
            slot_times[slot] = float(info["source_start_time"])
    return slot_times


def compile_schedule_payload(payload: dict, slot_start_times: dict[str, float] | None = None) -> dict:
    slot_start_times = slot_start_times or {}
    scheduled = []
    skipped = []
    for event in payload.get("events", []):
        weapon_name = map_weapon_name(event.get("missile_name", ""))
        shooter_slot = ((event.get("shooter") or {}).get("slot") or "").strip()
        target_slot = ((event.get("target") or {}).get("slot") or "").strip()
        if weapon_name is None or not shooter_slot or not target_slot:
            skipped.append(
                {
                    "missile_id": event.get("missile_id"),
                    "missile_name": event.get("missile_name"),
                    "reason": "unsupported_weapon_or_missing_slot",
                }
            )
            continue
        launch_time = float(event.get("launch_time", 0.0))
        replay_time = launch_time - slot_start_times.get(shooter_slot, 0.0)
        if replay_time < 0.0:
            replay_time = 0.0
        scheduled.append(
            {
                "missile_id": event.get("missile_id"),
                "missile_name": event.get("missile_name"),
                "weapon_name": weapon_name,
                "launch_time": replay_time,
                "source_launch_time": launch_time,
                "shooter_slot": shooter_slot,
                "target_slot": target_slot,
            }
        )
    scheduled.sort(key=lambda item: (item["launch_time"], item["missile_id"]))
    return {"scheduled": scheduled, "skipped": skipped}


def render_schedule_include(
    payload: dict,
    host_name: str = "weapon_schedule_host",
    slot_start_times: dict[str, float] | None = None,
) -> str:
    compiled = compile_schedule_payload(payload, slot_start_times=slot_start_times)
    scheduled = compiled["scheduled"]
    skipped = compiled["skipped"]

    lines = [
        "# ****************************************************************************",
        "# Auto-generated AFSIM weapon launch schedule include.",
        "# Source payload: weapon_events.json",
        f"# scheduled_events {len(scheduled)}",
        f"# skipped_events {len(skipped)}",
        "# ****************************************************************************",
        "",
        "script bool FireReplayWeapon(string shooterName, string targetName, string weaponName, string missileEventId)",
        "   WsfPlatform shooter = WsfSimulation.FindPlatform(shooterName);",
        "   if (!shooter.IsValid())",
        "   {",
        '      writeln("Weapon schedule: invalid shooter <" , shooterName , "> for event <" , missileEventId , ">.");',
        "      return false;",
        "   }",
        "",
        "   WsfPlatform target = WsfSimulation.FindPlatform(targetName);",
        "   if (!target.IsValid())",
        "   {",
        '      writeln("Weapon schedule: invalid target <" , targetName , "> for event <" , missileEventId , ">.");',
        "      return false;",
        "   }",
        "",
        "   WsfWeapon weapon = shooter.Weapon(weaponName);",
        "   if (!weapon.IsValid())",
        "   {",
        '      writeln("Weapon schedule: shooter <" , shooterName , "> has no weapon <" , weaponName , "> for event <" , missileEventId , ">.");',
        "      return false;",
        "   }",
        "   if (weapon.QuantityRemaining() <= 0)",
        "   {",
        '      writeln("Weapon schedule: shooter <" , shooterName , "> is out of <" , weaponName , "> for event <" , missileEventId , ">.");',
        "      return false;",
        "   }",
        "",
        "   foreach (WsfLocalTrack ltrk in shooter.MasterTrackList())",
        "   {",
        "      if (ltrk.IsValid() && ltrk.TargetName() == target.Name())",
        "      {",
        "         bool fired = weapon.Fire(ltrk);",
        "         if (fired)",
        "         {",
        '            writeln("Weapon schedule: T=" , TIME_NOW , " " , shooterName , " FIRE " , weaponName , " AT " , targetName , " via live track, event=" , missileEventId);',
        "            return true;",
        "         }",
        "      }",
        "   }",
        "",
        "   WsfTrack trk = WsfTrack();",
        "   trk.SetTarget(target.Name());",
        "   trk.SetLocationLLA(target.Latitude(), target.Longitude(), target.Altitude());",
        "   bool fired = weapon.Fire(trk);",
        "   if (fired)",
        "   {",
        '      writeln("Weapon schedule: T=" , TIME_NOW , " " , shooterName , " FIRE " , weaponName , " AT " , targetName , " via synthetic track, event=" , missileEventId);',
        "   }",
        "   else",
        "   {",
        '      writeln("Weapon schedule: FAILED fire attempt from <" , shooterName , "> to <" , targetName , "> using <" , weaponName , ">, event=<" , missileEventId , ">.");',
        "   }",
        "   return fired;",
        "end_script",
        "",
        f"platform {host_name} WSF_PLATFORM",
        "   side green",
        "   commander SELF",
        "   position 80:10:00.00n 00:00:00.00e altitude 30000.00 ft",
        "",
    ]

    for event in scheduled:
        lines.extend(
            [
                f"   execute at_time {event['launch_time']:.3f} sec absolute",
                f'      FireReplayWeapon("{_quote(event["shooter_slot"])}", "{_quote(event["target_slot"])}", "{event["weapon_name"]}", "{_quote(str(event["missile_id"]))}");',
                "   end_execute",
                "",
            ]
        )

    lines.append("end_platform")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compile weapon_events.json into an AFSIM include schedule")
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots-meta", default="")
    ap.add_argument("--host-name", default="weapon_schedule_host")
    args = ap.parse_args()

    events_path = Path(args.events)
    out_path = Path(args.out)
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    slot_start_times = load_slot_start_times(args.slots_meta)
    text = render_schedule_include(payload, host_name=args.host_name, slot_start_times=slot_start_times)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    compiled = compile_schedule_payload(payload, slot_start_times=slot_start_times)
    print(
        f"[DONE] wrote schedule include -> {out_path} "
        f"(scheduled={len(compiled['scheduled'])}, skipped={len(compiled['skipped'])})"
    )


if __name__ == "__main__":
    main()
