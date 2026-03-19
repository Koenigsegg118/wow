#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export deployable policy artifacts for inference runtime.

Schema version: 0.2.0
Breaking changes vs 0.1.x:
  - obs_spec.field_order: heading_rad_unwrapped -> track_angle_rad_unwrapped
  - action_spec: added horizon_steps, control_frequency_hz, angle_reference
  - obs_spec: added ground_speed_convention
  - seq_len is now REQUIRED in checkpoint (--checkpoint path).
    Export fails with a clear error if seq_len is absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from shared.schema import SCHEMA_VERSION
import dataset_default_config as CFG


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _export_numpy_linear(
    artifact_dir: Path,
    obs_dim: int,
    act_dim: int,
    act_mean: np.ndarray,
) -> str:
    model_file = "model.json"
    weights = np.zeros((obs_dim, act_dim), dtype=np.float32)
    bias = act_mean.astype(np.float32)
    _write_json(
        artifact_dir / model_file,
        {
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "note": "Fallback linear policy: action = obs_last @ W + b",
        },
    )
    return model_file


def _export_onnx(
    artifact_dir: Path,
    checkpoint: Dict[str, Any],
    seq_len: int,
    obs_dim: int,
    act_dim: int,
) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for ONNX export.") from exc
    from shared.models.transformer_bc import TransformerBC

    args = checkpoint.get("args", {})
    model = TransformerBC(
        obs_dim=obs_dim,
        act_dim=act_dim,
        d_model=int(args.get("d_model", 128)),
        nhead=int(args.get("heads", 4)),
        num_layers=int(args.get("layers", 4)),
        dropout=float(args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    model_file = "model.onnx"
    out_path = artifact_dir / model_file
    dummy = torch.zeros((1, seq_len, obs_dim), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["obs_seq"],
        output_names=["pred_action_seq"],
        dynamic_axes={"obs_seq": {1: "seq"}, "pred_action_seq": {1: "seq"}},
        opset_version=17,
    )
    return model_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Export model artifact for inference runtime (schema 0.2.0)")
    ap.add_argument("--checkpoint", default="", help="Input .pt checkpoint (optional for numpy_linear)")
    ap.add_argument("--out", required=True, help="Artifact output directory")
    ap.add_argument(
        "--backend",
        default="numpy_linear",
        choices=["numpy_linear", "onnxruntime"],
        help="Target runtime backend",
    )
    ap.add_argument(
        "--seq_len", type=int, default=None,
        help="Override seq_len. If --checkpoint is given, seq_len is read from checkpoint "
             "and this flag is forbidden. For numpy_linear without checkpoint, defaults to CFG.SEQ_LEN.",
    )
    ap.add_argument("--obs_dim", type=int, default=len(CFG.OBS_FIELDS))
    ap.add_argument("--act_dim", type=int, default=len(CFG.ACT_FIELDS))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_mean = np.zeros(args.obs_dim, dtype=np.float32)
    obs_std = np.ones(args.obs_dim, dtype=np.float32)
    act_mean = np.zeros(args.act_dim, dtype=np.float32)
    act_std = np.ones(args.act_dim, dtype=np.float32)
    checkpoint: Dict[str, Any] | None = None
    seq_len: int

    if args.checkpoint:
        if args.seq_len is not None:
            raise SystemExit(
                "--seq_len is not allowed when --checkpoint is provided. "
                "seq_len is read exclusively from the checkpoint to prevent silent mismatches."
            )
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required to read checkpoint.") from exc

        checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)

        # seq_len is REQUIRED in checkpoint (persisted since train_bc_transformer_smooth.py 0.2.0)
        if "seq_len" not in checkpoint:
            raise SystemExit(
                f"Checkpoint '{args.checkpoint}' is missing the 'seq_len' key.\n"
                "This means it was produced by the old training script.\n"
                "Re-train with the updated ml/train_bc_transformer_smooth.py, "
                "or use --backend numpy_linear without --checkpoint."
            )
        seq_len = int(checkpoint["seq_len"])

        obs_mean = np.asarray(checkpoint["obs_mean"], dtype=np.float32)
        obs_std = np.asarray(checkpoint["obs_std"], dtype=np.float32)
        act_mean = np.asarray(checkpoint["act_mean"], dtype=np.float32)
        act_std = np.asarray(checkpoint["act_std"], dtype=np.float32)
        args.obs_dim = int(obs_mean.shape[0])
        args.act_dim = int(act_mean.shape[0])

        # Sanity-check obs_dim against schema OBS_FIELDS count
        if args.obs_dim != len(CFG.OBS_FIELDS):
            print(
                f"[export] WARNING: checkpoint obs_dim={args.obs_dim} != "
                f"len(CFG.OBS_FIELDS)={len(CFG.OBS_FIELDS)}. "
                "field_order in policy.yaml may be stale."
            )
    else:
        # No checkpoint — only numpy_linear fallback is meaningful
        if args.backend == "onnxruntime":
            raise SystemExit("--backend onnxruntime requires --checkpoint")
        seq_len = args.seq_len if args.seq_len is not None else CFG.SEQ_LEN
        print(f"[export] No checkpoint provided; using seq_len={seq_len} (fallback/CFG).")

    if args.backend == "onnxruntime":
        model_file = _export_onnx(
            artifact_dir=out_dir,
            checkpoint=checkpoint,
            seq_len=seq_len,
            obs_dim=args.obs_dim,
            act_dim=args.act_dim,
        )
        normalize_obs = True
        denormalize_action = True
    else:
        model_file = _export_numpy_linear(
            artifact_dir=out_dir,
            obs_dim=args.obs_dim,
            act_dim=args.act_dim,
            act_mean=act_mean,
        )
        normalize_obs = False
        denormalize_action = False

    _write_json(
        out_dir / "norm_stats.json",
        {
            "obs_mean": obs_mean.astype(np.float32).tolist(),
            "obs_std": obs_std.astype(np.float32).tolist(),
            "act_mean": act_mean.astype(np.float32).tolist(),
            "act_std": act_std.astype(np.float32).tolist(),
        },
    )

    policy = {
        "schema_version": SCHEMA_VERSION,
        "backend": args.backend,
        "model_file": model_file,
        "obs_spec": {
            "dim": args.obs_dim,
            "seq_len": seq_len,
            "field_order": CFG.OBS_FIELDS,
            "ground_speed_convention": "horizontal_2d",
            "track_angle_convention": "atan2(vx_east, vy_north)_unwrapped",
        },
        "action_spec": {
            "dim": args.act_dim,
            "field_order": CFG.ACT_FIELDS,
            "horizon_steps": CFG.HORIZON_STEPS,
            "control_frequency_hz": CFG.CONTROL_FREQUENCY_HZ,
            "angle_reference": "track_angle",
            "note": (
                "dpsi_rad is the track-angle delta from t to t+horizon_steps control steps "
                f"(={CFG.HORIZON_STEPS / CFG.CONTROL_FREQUENCY_HZ:.1f}s). "
                "CW positive. alt_sp_m and spd_sp_mps are setpoints at the same horizon."
            ),
        },
        "units": {
            "distance": "m",
            "speed": "m/s",
            "angle": "rad",
        },
        "coordinate_frame": "ENU",
        "preprocess": {
            "normalize_obs": normalize_obs,
            "denormalize_action": denormalize_action,
            "norm_stats_file": "norm_stats.json",
        },
        "input_tensor_names": ["obs_seq"],
        "output_tensor_names": ["pred_action_seq"],
        "exported_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source_checkpoint": args.checkpoint or "",
    }
    (out_dir / "policy.yaml").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme_lines = [
        "# Policy Artifact\n",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- backend: `{args.backend}`",
        f"- model_file: `{model_file}`",
        f"- seq_len: `{seq_len}`",
        f"- horizon_steps: `{CFG.HORIZON_STEPS}` ({CFG.HORIZON_STEPS / CFG.CONTROL_FREQUENCY_HZ:.1f}s lookahead)",
        f"- control_frequency_hz: `{CFG.CONTROL_FREQUENCY_HZ}`",
        "- obs_fields: " + ", ".join(f"`{f}`" for f in CFG.OBS_FIELDS),
        "- act_fields: " + ", ".join(f"`{f}`" for f in CFG.ACT_FIELDS),
        "- files:",
        "  - `policy.yaml`",
        "  - `norm_stats.json`",
        f"  - `{model_file}`",
        "",
        "## Runtime adapter requirements",
        "- `track_angle_rad_unwrapped` must be continuously unwrapped across the obs window.",
        "  Do NOT pass the raw wrapped `heading_rad` from an episode JSON.",
        "- `ground_speed_mps` = sqrt(vx_east² + vy_north²) (2-D horizontal only).",
        f"- `dpsi_rad` output spans {CFG.HORIZON_STEPS / CFG.CONTROL_FREQUENCY_HZ:.1f}s "
        f"({CFG.HORIZON_STEPS} steps at {CFG.CONTROL_FREQUENCY_HZ} Hz). "
        "Scale accordingly before sending to actuator.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"[OK] artifact exported: {out_dir}")
    print(f"     schema_version={SCHEMA_VERSION}  backend={args.backend}  seq_len={seq_len}")
    print(f"     horizon_steps={CFG.HORIZON_STEPS}  control_frequency_hz={CFG.CONTROL_FREQUENCY_HZ}")


if __name__ == "__main__":
    main()
