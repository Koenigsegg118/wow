#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime contract tests.

Organised into six categories:

1. Heading regression (track angle from velocity)
2. Artifact contract validation (strict mode defaults)
3. Bridge dt-invariance
4. Sign convention
5. Seq-len / action-field-order / legacy gate (A/B/C)
6. Horizon-steps interpretation and lookahead-delta mapping (D)
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.afsim_policy_bridge import (
    BridgeConfig,
    Limits,
    Slew,
    SmoothSetpointController,
)
from sim.bc_policy_socket_server import (
    SUPPORTED_FIELD_ORDER,
    BCPolicyServer,
    PlatformTrack,
    unwrap_with_prev,
)
from inference.runtime import (
    PolicyRuntime,
    REQUIRED_ACTION_FIELD_ORDER,
)
from shared.schema import SCHEMA_VERSION


# ====================================================================
# Helpers
# ====================================================================

def _make_artifact(
    tmp_path: Path,
    policy_overrides: dict | None = None,
    *,
    drop_keys: list[str] | None = None,
) -> Path:
    """Create a minimal valid 0.2.0 artifact directory.

    ``policy_overrides`` is a shallow-merge dict.  ``drop_keys`` is a
    dot-separated list of keys to remove after merge (e.g.
    ``["obs_spec.seq_len", "action_spec.field_order"]``).
    """
    policy: dict = {
        "schema_version": SCHEMA_VERSION,
        "backend": "numpy_linear",
        "model_file": "model.json",
        "obs_spec": {
            "dim": 8,
            "seq_len": 20,
            "field_order": list(SUPPORTED_FIELD_ORDER),
            "ground_speed_convention": "horizontal_2d",
            "track_angle_convention": "atan2(vx_east, vy_north)_unwrapped",
        },
        "action_spec": {
            "dim": 3,
            "field_order": list(REQUIRED_ACTION_FIELD_ORDER),
            "control_frequency_hz": 2.0,
            "horizon_steps": 4,
            "angle_reference": "track_angle",
        },
        "preprocess": {
            "normalize_obs": False,
            "denormalize_action": False,
            "norm_stats_file": "norm_stats.json",
        },
        "input_tensor_names": ["obs_seq"],
        "output_tensor_names": ["pred_action_seq"],
    }
    if policy_overrides:
        for k, v in policy_overrides.items():
            if isinstance(v, dict) and k in policy and isinstance(policy[k], dict):
                policy[k].update(v)
            else:
                policy[k] = v

    for dk in (drop_keys or []):
        parts = dk.split(".", 1)
        if len(parts) == 2:
            outer, inner = parts
            if outer in policy and isinstance(policy[outer], dict):
                policy[outer].pop(inner, None)
        else:
            policy.pop(parts[0], None)

    (tmp_path / "policy.yaml").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    (tmp_path / "model.json").write_text(
        json.dumps({"weights": [[0] * 3] * 8, "bias": [0, 0, 0]}),
        encoding="utf-8",
    )
    (tmp_path / "norm_stats.json").write_text(
        json.dumps({
            "obs_mean": [0] * 8, "obs_std": [1] * 8,
            "act_mean": [0] * 3, "act_std": [1] * 3,
        }),
        encoding="utf-8",
    )
    return tmp_path


def _make_server(art: Path, **kw) -> BCPolicyServer:
    defaults = dict(
        artifact_dir=str(art),
        control_indices=[0],
        host="localhost",
        port=65432,
    )
    defaults.update(kw)
    return BCPolicyServer(**defaults)


# ====================================================================
# 1) Heading regression tests
# ====================================================================

class TestTrackAngleFromVelocity:
    """Verify that we use atan2(vx_east, vy_north) for track angle."""

    def test_north(self):
        assert abs(math.atan2(0.0, 220.0) - 0.0) < 1e-9

    def test_east(self):
        assert abs(math.atan2(220.0, 0.0) - math.pi / 2) < 1e-9

    def test_south(self):
        assert abs(abs(math.atan2(0.0, -220.0)) - math.pi) < 1e-9

    def test_west(self):
        assert abs(math.atan2(-220.0, 0.0) - (-math.pi / 2)) < 1e-9

    def test_unwrap_crosses_pi(self):
        angles_deg = [170.0, 175.0, 180.0, 185.0, 190.0]
        prev = None
        unwrapped = []
        for deg in angles_deg:
            rad = math.radians(deg)
            raw = math.atan2(math.sin(rad), math.cos(rad))
            u = unwrap_with_prev(raw, prev)
            unwrapped.append(u)
            prev = u
        for i in range(1, len(unwrapped)):
            assert unwrapped[i] > unwrapped[i - 1]

    def test_build_obs_uses_velocity_not_heading_deg(self, tmp_path):
        art = _make_artifact(tmp_path)
        srv = _make_server(art)
        sim_data = np.zeros(640, dtype=np.float32)
        sim_data[0] = 1       # live
        sim_data[1] = 63.0    # lat
        sim_data[2] = 12.0    # lon
        sim_data[3] = 5000.0  # alt
        sim_data[4] = 0.0     # velN = 0
        sim_data[5] = 220.0   # velE = 220  -> track angle = +pi/2
        sim_data[6] = 0.0     # velD
        sim_data[7] = 0.0     # heading_deg = 0 (body heading north)

        st = srv._state_of(sim_data, 0)
        track = srv.tracks[0]
        track.ensure(srv.seq_len)
        obs = srv._build_obs(st, track)

        track_angle_in_obs = float(obs[6])
        expected = math.pi / 2.0
        assert abs(track_angle_in_obs - expected) < 1e-6, (
            f"Expected track_angle~={expected:.4f}, got {track_angle_in_obs:.4f}. "
            f"Obs must NOT use body heading_deg."
        )


# ====================================================================
# 2) Artifact contract validation tests (strict mode)
# ====================================================================

class TestArtifactContractStrict:
    """All tests here use the default strict mode (allow_legacy=False)."""

    def test_valid_artifact_loads(self, tmp_path):
        art = _make_artifact(tmp_path)
        rt = PolicyRuntime.from_artifact_dir(art)
        assert rt.field_order == list(SUPPORTED_FIELD_ORDER)
        assert rt.action_field_order == list(REQUIRED_ACTION_FIELD_ORDER)
        assert rt.control_frequency_hz == 2.0
        assert rt.horizon_steps == 4
        assert rt.obs_dim == 8
        assert rt.seq_len == 20

    def test_field_order_dim_mismatch_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "obs_spec": {
                "dim": 8, "seq_len": 20,
                "field_order": ["x_e_m", "y_n_m", "z_u_m"],
            },
        })
        with pytest.raises(ValueError, match="field_order has 3 fields but dim=8"):
            PolicyRuntime.from_artifact_dir(art)

    def test_different_major_version_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {"schema_version": "99.0.0"})
        with pytest.raises(ValueError, match="does not match"):
            PolicyRuntime.from_artifact_dir(art)

    def test_newer_version_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {"schema_version": "0.99.0"})
        with pytest.raises(ValueError, match="does not match"):
            PolicyRuntime.from_artifact_dir(art)

    def test_server_rejects_wrong_obs_field_order(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "obs_spec": {
                "dim": 8, "seq_len": 20,
                "field_order": [
                    "z_u_m", "x_e_m", "y_n_m",
                    "vx_e_mps", "vy_n_mps", "vz_u_mps",
                    "track_angle_rad_unwrapped", "ground_speed_mps",
                ],
            },
        })
        with pytest.raises(ValueError, match="field_order mismatch"):
            _make_server(art)

    # --- strict mode REJECTS missing/legacy fields ---

    def test_strict_rejects_old_schema(self, tmp_path):
        art = _make_artifact(tmp_path, {"schema_version": "0.1.1"})
        with pytest.raises(ValueError, match="does not match"):
            PolicyRuntime.from_artifact_dir(art)

    def test_strict_rejects_heading_rad_alias(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "obs_spec": {
                "dim": 8, "seq_len": 20,
                "field_order": [
                    "x_e_m", "y_n_m", "z_u_m",
                    "vx_e_mps", "vy_n_mps", "vz_u_mps",
                    "heading_rad", "ground_speed_mps",
                ],
            },
        })
        with pytest.raises(ValueError, match="deprecated field name"):
            PolicyRuntime.from_artifact_dir(art)

    def test_strict_rejects_missing_obs_field_order(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["obs_spec.field_order"])
        with pytest.raises(ValueError, match="obs_spec.field_order"):
            PolicyRuntime.from_artifact_dir(art)

    def test_strict_rejects_missing_seq_len(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["obs_spec.seq_len"])
        with pytest.raises(ValueError, match="obs_spec.seq_len"):
            PolicyRuntime.from_artifact_dir(art)

    def test_strict_rejects_missing_action_field_order(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["action_spec.field_order"])
        with pytest.raises(ValueError, match="action_spec.field_order"):
            PolicyRuntime.from_artifact_dir(art)

    def test_strict_rejects_missing_control_frequency(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["action_spec.control_frequency_hz"])
        with pytest.raises(ValueError, match="control_frequency_hz"):
            PolicyRuntime.from_artifact_dir(art)

    def test_strict_rejects_missing_horizon_steps(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["action_spec.horizon_steps"])
        with pytest.raises(ValueError, match="horizon_steps"):
            PolicyRuntime.from_artifact_dir(art)

    def test_strict_rejects_legacy_checkpoint(self):
        with pytest.raises(ValueError, match="requires explicit opt-in"):
            PolicyRuntime.from_legacy_checkpoint("nonexistent.pt")


# ====================================================================
# 2b) Artifact contract in legacy mode
# ====================================================================

class TestArtifactContractLegacy:
    """Tests that verify legacy compat works when explicitly opted in."""

    def test_heading_rad_alias_warns_and_maps(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "obs_spec": {
                "dim": 8, "seq_len": 20,
                "field_order": [
                    "x_e_m", "y_n_m", "z_u_m",
                    "vx_e_mps", "vy_n_mps", "vz_u_mps",
                    "heading_rad", "ground_speed_mps",
                ],
            },
        })
        with pytest.warns(match="LEGACY COMPAT.*heading_rad"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert "track_angle_rad_unwrapped" in rt.field_order

    def test_heading_rad_unwrapped_alias(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "obs_spec": {
                "dim": 8, "seq_len": 20,
                "field_order": [
                    "x_e_m", "y_n_m", "z_u_m",
                    "vx_e_mps", "vy_n_mps", "vz_u_mps",
                    "heading_rad_unwrapped", "ground_speed_mps",
                ],
            },
        })
        with pytest.warns(match="LEGACY COMPAT.*heading_rad_unwrapped"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert "track_angle_rad_unwrapped" in rt.field_order

    def test_missing_field_order_warns_and_defaults(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["obs_spec.field_order"])
        with pytest.warns(match="LEGACY COMPAT.*obs_spec.field_order"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt.field_order is not None
        assert "track_angle_rad_unwrapped" in rt.field_order

    def test_older_minor_version_compat_warns(self, tmp_path):
        art = _make_artifact(tmp_path, {"schema_version": "0.1.1"})
        with pytest.warns(match="LEGACY COMPAT.*schema_version"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt is not None

    def test_missing_control_frequency_warns(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["action_spec.control_frequency_hz"])
        with pytest.warns(match="LEGACY COMPAT.*control_frequency_hz"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt.control_frequency_hz is None

    def test_missing_horizon_steps_warns(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["action_spec.horizon_steps"])
        with pytest.warns(match="LEGACY COMPAT.*horizon_steps"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt.horizon_steps is None

    def test_missing_action_field_order_warns(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["action_spec.field_order"])
        with pytest.warns(match="LEGACY COMPAT.*action_spec.field_order"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt.action_field_order == REQUIRED_ACTION_FIELD_ORDER

    def test_missing_seq_len_warns(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["obs_spec.seq_len"])
        with pytest.warns(match="LEGACY COMPAT.*obs_spec.seq_len"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt.seq_len == 20  # default


# ====================================================================
# 3) Bridge dt-invariance tests
# ====================================================================

class TestBridgeDtInvariance:
    def test_same_physical_rate_across_dt(self):
        slew = Slew(
            dpsi_rate_rad_per_s=0.2,
            alt_rate_m_per_s=100.0,
            spd_rate_mps_per_s=10.0,
        )
        target_dpsi, target_alt, target_spd = 0.5, 6000.0, 300.0
        init_alt, init_spd = 5000.0, 250.0

        ctrl_a = SmoothSetpointController(slew=slew)
        ctrl_a.reset(init_alt, init_spd)
        total_a = 0.0
        for _ in range(2):
            cmd_a, _ = ctrl_a.step(
                target_dpsi, target_alt, target_spd,
                init_alt, init_spd, dt=0.5, lookahead_horizon_s=2.0,
            )
            total_a += float(cmd_a.dpsi_rad)

        ctrl_b = SmoothSetpointController(slew=slew)
        ctrl_b.reset(init_alt, init_spd)
        total_b = 0.0
        for _ in range(10):
            cmd_b, _ = ctrl_b.step(
                target_dpsi, target_alt, target_spd,
                init_alt, init_spd, dt=0.1, lookahead_horizon_s=2.0,
            )
            total_b += float(cmd_b.dpsi_rad)

        assert ctrl_a.prev is not None and ctrl_b.prev is not None
        assert abs(total_a - total_b) < 1e-6
        assert abs(ctrl_a.prev.alt_m - ctrl_b.prev.alt_m) < 1e-3
        assert abs(ctrl_a.prev.spd_mps - ctrl_b.prev.spd_mps) < 1e-3

    def test_half_dt_half_slew(self):
        slew = Slew(dpsi_rate_rad_per_s=0.2, alt_rate_m_per_s=100.0,
                     spd_rate_mps_per_s=10.0)

        ctrl_half = SmoothSetpointController(slew=slew)
        ctrl_half.reset(5000.0, 250.0)
        ctrl_half.step(
            1.0, 10000.0, 400.0, 5000.0, 250.0,
            dt=0.25, lookahead_horizon_s=2.0,
        )

        ctrl_full = SmoothSetpointController(slew=slew)
        ctrl_full.reset(5000.0, 250.0)
        ctrl_full.step(
            1.0, 10000.0, 400.0, 5000.0, 250.0,
            dt=0.5, lookahead_horizon_s=2.0,
        )

        assert ctrl_half.prev is not None and ctrl_full.prev is not None
        ratio_dpsi = ctrl_half.prev.dpsi_rad / ctrl_full.prev.dpsi_rad
        assert abs(ratio_dpsi - 0.5) < 1e-6

    def test_dt_zero_raises(self):
        ctrl = SmoothSetpointController()
        ctrl.reset(5000.0, 250.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            ctrl.step(
                0.0, 5000.0, 250.0, 5000.0, 250.0,
                dt=0.0, lookahead_horizon_s=2.0,
            )

    def test_dt_negative_raises(self):
        ctrl = SmoothSetpointController()
        ctrl.reset(5000.0, 250.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            ctrl.step(
                0.0, 5000.0, 250.0, 5000.0, 250.0,
                dt=-0.1, lookahead_horizon_s=2.0,
            )


# ====================================================================
# 4) Sign convention tests
# ====================================================================

class TestSignConvention:
    def test_positive_dpsi_positive_turn(self):
        ctrl = SmoothSetpointController(config=BridgeConfig(turn_sign=1.0))
        ctrl.reset(5000.0, 250.0)
        _cmd, (turn_deg, _, _) = ctrl.step(
            dpsi_rad=0.1, alt_sp_m=5000.0, spd_sp_mps=250.0,
            alt_m=5000.0, spd_mps=250.0, dt=0.5, lookahead_horizon_s=0.5,
        )
        assert turn_deg > 0

    def test_negative_dpsi_negative_turn(self):
        ctrl = SmoothSetpointController(config=BridgeConfig(turn_sign=1.0))
        ctrl.reset(5000.0, 250.0)
        _cmd, (turn_deg, _, _) = ctrl.step(
            dpsi_rad=-0.1, alt_sp_m=5000.0, spd_sp_mps=250.0,
            alt_m=5000.0, spd_mps=250.0, dt=0.5, lookahead_horizon_s=0.5,
        )
        assert turn_deg < 0

    def test_flipped_turn_sign(self):
        ctrl = SmoothSetpointController(config=BridgeConfig(turn_sign=-1.0))
        ctrl.reset(5000.0, 250.0)
        _cmd, (turn_deg, _, _) = ctrl.step(
            dpsi_rad=0.1, alt_sp_m=5000.0, spd_sp_mps=250.0,
            alt_m=5000.0, spd_mps=250.0, dt=0.5, lookahead_horizon_s=0.5,
        )
        assert turn_deg < 0

    def test_rad2deg_identity(self):
        from sim.afsim_policy_bridge import rad2deg
        assert abs(rad2deg(math.pi) - 180.0) < 1e-9
        assert abs(rad2deg(-math.pi / 2) - (-90.0)) < 1e-9


# ====================================================================
# A) seq_len contract tests
# ====================================================================

class TestSeqLenContract:
    """Verify that seq_len is strongly enforced end-to-end."""

    def test_predict_exact_seq_len_passes(self, tmp_path):
        art = _make_artifact(tmp_path, {"obs_spec": {"seq_len": 10, "dim": 8}})
        rt = PolicyRuntime.from_artifact_dir(art)
        obs = np.zeros((10, 8), dtype=np.float32)
        pred = rt.predict(obs)
        assert pred.shape == (3,)

    def test_predict_too_short_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {"obs_spec": {"seq_len": 20, "dim": 8}})
        rt = PolicyRuntime.from_artifact_dir(art)
        obs = np.zeros((19, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match.*seq_len=20"):
            rt.predict(obs)

    def test_predict_too_long_fails_no_truncation(self, tmp_path):
        art = _make_artifact(tmp_path, {"obs_spec": {"seq_len": 20, "dim": 8}})
        rt = PolicyRuntime.from_artifact_dir(art)
        obs = np.zeros((21, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match.*seq_len=20"):
            rt.predict(obs)

    def test_strict_missing_seq_len_fails(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["obs_spec.seq_len"])
        with pytest.raises(ValueError, match="obs_spec.seq_len"):
            PolicyRuntime.from_artifact_dir(art)

    def test_legacy_missing_seq_len_defaults(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["obs_spec.seq_len"])
        with pytest.warns(match="LEGACY COMPAT.*obs_spec.seq_len"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt.seq_len == 20

    def test_server_rejects_seq_len_override(self, tmp_path):
        art = _make_artifact(tmp_path, {"obs_spec": {"seq_len": 20, "dim": 8}})
        with pytest.raises(ValueError, match="conflicts with artifact"):
            _make_server(art, seq_len=15)


# ====================================================================
# B) action_spec.field_order contract tests
# ====================================================================

class TestActionFieldOrderContract:
    """Verify that action_spec.field_order is strongly validated."""

    def test_correct_order_passes(self, tmp_path):
        art = _make_artifact(tmp_path)
        rt = PolicyRuntime.from_artifact_dir(art)
        assert rt.action_field_order == REQUIRED_ACTION_FIELD_ORDER

    def test_swapped_order_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "action_spec": {
                "dim": 3,
                "field_order": ["alt_sp_m", "dpsi_rad", "spd_sp_mps"],
                "control_frequency_hz": 2.0,
                "horizon_steps": 1,
            },
        })
        with pytest.raises(ValueError, match="action_spec.field_order must be exactly"):
            PolicyRuntime.from_artifact_dir(art)

    def test_wrong_field_names_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "action_spec": {
                "dim": 3,
                "field_order": ["heading_rad", "alt_m", "speed_mps"],
                "control_frequency_hz": 2.0,
                "horizon_steps": 1,
            },
        })
        with pytest.raises(ValueError, match="action_spec.field_order must be exactly"):
            PolicyRuntime.from_artifact_dir(art)

    def test_missing_field_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "action_spec": {
                "dim": 3,
                "field_order": ["dpsi_rad", "alt_sp_m"],
                "control_frequency_hz": 2.0,
                "horizon_steps": 1,
            },
        })
        with pytest.raises(ValueError, match="action_spec.field_order must be exactly"):
            PolicyRuntime.from_artifact_dir(art)

    def test_extra_field_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "action_spec": {
                "dim": 3,
                "field_order": ["dpsi_rad", "alt_sp_m", "spd_sp_mps", "extra"],
                "control_frequency_hz": 2.0,
                "horizon_steps": 1,
            },
        })
        with pytest.raises(ValueError, match="action_spec.field_order must be exactly"):
            PolicyRuntime.from_artifact_dir(art)

    def test_strict_missing_action_field_order_fails(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["action_spec.field_order"])
        with pytest.raises(ValueError, match="action_spec.field_order"):
            PolicyRuntime.from_artifact_dir(art)

    def test_legacy_missing_action_field_order_warns(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["action_spec.field_order"])
        with pytest.warns(match="LEGACY COMPAT.*action_spec.field_order"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt.action_field_order == REQUIRED_ACTION_FIELD_ORDER

    def test_server_validates_action_field_order(self, tmp_path):
        art = _make_artifact(tmp_path)
        srv = _make_server(art)
        assert srv.policy.action_field_order == REQUIRED_ACTION_FIELD_ORDER


# ====================================================================
# C) legacy gate tests
# ====================================================================

class TestLegacyGate:
    """Strict mode rejects; legacy mode accepts + warns."""

    def test_old_schema_strict_fails(self, tmp_path):
        art = _make_artifact(tmp_path, {"schema_version": "0.1.1"})
        with pytest.raises(ValueError, match="does not match"):
            PolicyRuntime.from_artifact_dir(art, allow_legacy=False)

    def test_old_schema_legacy_warns(self, tmp_path):
        art = _make_artifact(tmp_path, {"schema_version": "0.1.1"})
        with pytest.warns(match="LEGACY COMPAT.*schema_version"):
            rt = PolicyRuntime.from_artifact_dir(art, allow_legacy=True)
        assert rt is not None

    def test_legacy_pt_strict_fails(self):
        with pytest.raises(ValueError, match="requires explicit opt-in"):
            PolicyRuntime.from_legacy_checkpoint(
                "nonexistent.pt", allow_legacy=False,
            )

    def test_major_version_mismatch_fails_even_legacy(self, tmp_path):
        art = _make_artifact(tmp_path, {"schema_version": "99.0.0"})
        with pytest.raises(ValueError, match="major version mismatch"):
            PolicyRuntime.from_artifact_dir(art, allow_legacy=True)

    def test_newer_version_fails_even_legacy(self, tmp_path):
        art = _make_artifact(tmp_path, {"schema_version": "0.99.0"})
        with pytest.raises(ValueError, match="newer than runtime"):
            PolicyRuntime.from_artifact_dir(art, allow_legacy=True)

    def test_server_strict_mode_rejects_incomplete_artifact(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=["obs_spec.seq_len"])
        with pytest.raises(ValueError, match="obs_spec.seq_len"):
            _make_server(art, allow_legacy_artifact=False)

    def test_server_legacy_mode_accepts_incomplete_artifact(self, tmp_path):
        art = _make_artifact(tmp_path, drop_keys=[
            "obs_spec.seq_len",
            "action_spec.control_frequency_hz",
            "action_spec.horizon_steps",
        ])
        with pytest.warns(match="LEGACY COMPAT"):
            srv = _make_server(art, allow_legacy_artifact=True)
        assert srv.seq_len == 20

    def test_server_legacy_mode_accepts_heading_alias(self, tmp_path):
        art = _make_artifact(tmp_path, {
            "obs_spec": {
                "dim": 8, "seq_len": 20,
                "field_order": [
                    "x_e_m", "y_n_m", "z_u_m",
                    "vx_e_mps", "vy_n_mps", "vz_u_mps",
                    "heading_rad", "ground_speed_mps",
                ],
            },
        })
        with pytest.warns(match="LEGACY COMPAT.*heading_rad"):
            srv = _make_server(art, allow_legacy_artifact=True)
        assert srv.policy.field_order is not None
        assert "track_angle_rad_unwrapped" in srv.policy.field_order


# ====================================================================
# D) horizon_steps interpretation test
# ====================================================================

class TestHorizonStepsInterpretation:
    """Lock the producer/consumer contract: ``dpsi_rad`` is a lookahead
    track-angle delta over ``horizon_steps / control_frequency_hz`` seconds.
    The bridge must consume that metadata when building each control tick.
    """

    def test_longer_horizon_reduces_per_tick_turn(self):
        ctrl_h1 = SmoothSetpointController()
        ctrl_h1.reset(5000.0, 250.0)
        cmd_h1, (turn_h1, alt_h1, spd_h1) = ctrl_h1.step(
            dpsi_rad=0.08, alt_sp_m=5500.0, spd_sp_mps=280.0,
            alt_m=5000.0, spd_mps=250.0, dt=0.5, lookahead_horizon_s=0.5,
        )

        ctrl_h4 = SmoothSetpointController()
        ctrl_h4.reset(5000.0, 250.0)
        cmd_h4, (turn_h4, alt_h4, spd_h4) = ctrl_h4.step(
            dpsi_rad=0.08, alt_sp_m=5500.0, spd_sp_mps=280.0,
            alt_m=5000.0, spd_mps=250.0, dt=0.5, lookahead_horizon_s=2.0,
        )

        assert abs(cmd_h1.dpsi_rad / cmd_h4.dpsi_rad - 4.0) < 1e-6
        assert abs(turn_h1 / turn_h4 - 4.0) < 1e-6
        assert alt_h1 == alt_h4
        assert spd_h1 == spd_h4

    def test_lookahead_delta_accumulates_over_horizon(self):
        ctrl = SmoothSetpointController()
        ctrl.reset(5000.0, 250.0)
        emitted = []
        for _ in range(4):
            cmd, _ = ctrl.step(
                dpsi_rad=0.2, alt_sp_m=5000.0, spd_sp_mps=250.0,
                alt_m=5000.0, spd_mps=250.0, dt=0.5, lookahead_horizon_s=2.0,
            )
            emitted.append(cmd.dpsi_rad)

        assert abs(sum(emitted) - 0.2) < 1e-6

    def test_server_uses_artifact_lookahead_horizon(self):
        srv = BCPolicyServer(
            artifact_dir=str(_ROOT / 'artifacts' / 'policy_demo' / 'v0.2.0'),
            control_indices=[0],
            host='localhost',
            port=65432,
        )
        assert srv.policy.horizon_steps == 4
        assert srv.policy.control_frequency_hz == 2.0
        assert abs(srv.policy.lookahead_horizon_s - 2.0) < 1e-9
        assert abs(srv._lookahead_horizon_s - 2.0) < 1e-9

    def test_dpsi_is_lookahead_delta_not_single_tick_delta(self):
        ctrl = SmoothSetpointController(config=BridgeConfig(turn_sign=1.0))
        ctrl.reset(5000.0, 250.0)
        _, (turn_deg, _, _) = ctrl.step(
            dpsi_rad=0.1, alt_sp_m=5000.0, spd_sp_mps=250.0,
            alt_m=5000.0, spd_mps=250.0, dt=0.5, lookahead_horizon_s=2.0,
        )
        expected_deg = math.degrees(0.1 * (0.5 / 2.0))
        assert abs(turn_deg - expected_deg) < 0.01, (
            f"turn_deg={turn_deg:.4f}, expected={expected_deg:.4f}"
        )
