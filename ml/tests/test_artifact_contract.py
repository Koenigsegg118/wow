#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact contract tests (schema 0.2.0).

Verifies that a policy.yaml produced by ml/export_artifact.py contains all
mandatory fields introduced in schema 0.2.0.  Run with pytest or unittest.

Usage:
    pytest ml/tests/test_artifact_contract.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ML_DIR   = _THIS_DIR.parent
_ROOT_DIR = _ML_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

import dataset_default_config as CFG
from shared.schema import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_export(extra_args=None, checkpoint_path=None, expect_success=True):
    """Run export_artifact.py in a temp directory and return (policy, returncode)."""
    extra_args = extra_args or []
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, str(_ML_DIR / "export_artifact.py"),
            "--backend", "numpy_linear",
            "--out", tmpdir,
        ]
        if checkpoint_path:
            cmd += ["--checkpoint", str(checkpoint_path)]
        cmd += extra_args

        result = subprocess.run(cmd, capture_output=True, text=True)
        if expect_success:
            assert result.returncode == 0, (
                f"export_artifact.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            policy_path = Path(tmpdir) / "policy.yaml"
            assert policy_path.exists(), "policy.yaml not produced"
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        else:
            policy = None
        return policy, result.returncode


# ---------------------------------------------------------------------------
# Contract tests — numpy_linear (no checkpoint)
# ---------------------------------------------------------------------------

class TestArtifactContractNumpyLinear(unittest.TestCase):

    def setUp(self):
        self.policy, _ = _run_export()

    def test_schema_version(self):
        self.assertEqual(self.policy["schema_version"], SCHEMA_VERSION,
                         "schema_version must equal shared/schema/VERSION")

    def test_schema_version_is_020(self):
        self.assertEqual(self.policy["schema_version"], "0.2.0",
                         "Expected schema 0.2.0 after this PR")

    def test_obs_spec_seq_len_present(self):
        obs_spec = self.policy.get("obs_spec", {})
        self.assertIn("seq_len", obs_spec, "obs_spec must contain seq_len")
        self.assertIsInstance(obs_spec["seq_len"], int)
        self.assertGreater(obs_spec["seq_len"], 0)

    def test_obs_spec_field_order_present(self):
        obs_spec = self.policy.get("obs_spec", {})
        self.assertIn("field_order", obs_spec)
        self.assertIsInstance(obs_spec["field_order"], list)
        self.assertGreater(len(obs_spec["field_order"]), 0)

    def test_obs_field_track_angle(self):
        field_order = self.policy["obs_spec"]["field_order"]
        self.assertIn("track_angle_rad_unwrapped", field_order,
                      "track_angle_rad_unwrapped must be in obs_spec.field_order")
        self.assertNotIn("heading_rad_unwrapped", field_order,
                         "old name heading_rad_unwrapped must not appear")
        self.assertNotIn("heading_rad", field_order,
                         "heading_rad must not appear in model obs field_order")

    def test_obs_ground_speed_convention(self):
        obs_spec = self.policy.get("obs_spec", {})
        self.assertIn("ground_speed_convention", obs_spec)
        self.assertEqual(obs_spec["ground_speed_convention"], "horizontal_2d")

    def test_action_spec_horizon_steps(self):
        action_spec = self.policy.get("action_spec", {})
        self.assertIn("horizon_steps", action_spec,
                      "action_spec must contain horizon_steps (schema 0.2.0)")
        self.assertIsInstance(action_spec["horizon_steps"], int)
        self.assertEqual(action_spec["horizon_steps"], CFG.HORIZON_STEPS)

    def test_action_spec_control_frequency_hz(self):
        action_spec = self.policy.get("action_spec", {})
        self.assertIn("control_frequency_hz", action_spec,
                      "action_spec must contain control_frequency_hz (schema 0.2.0)")
        self.assertAlmostEqual(
            float(action_spec["control_frequency_hz"]),
            CFG.CONTROL_FREQUENCY_HZ, places=4,
        )

    def test_action_spec_angle_reference(self):
        action_spec = self.policy.get("action_spec", {})
        self.assertIn("angle_reference", action_spec,
                      "action_spec must contain angle_reference")
        self.assertEqual(action_spec["angle_reference"], "track_angle")

    def test_action_spec_field_order(self):
        action_spec = self.policy.get("action_spec", {})
        self.assertIn("field_order", action_spec)
        self.assertEqual(action_spec["field_order"], CFG.ACT_FIELDS)

    def test_units_block(self):
        units = self.policy.get("units", {})
        self.assertEqual(units.get("distance"), "m")
        self.assertEqual(units.get("speed"), "m/s")
        self.assertEqual(units.get("angle"), "rad")

    def test_norm_stats_file_present(self):
        # Indirect: preprocess block references norm_stats.json
        preprocess = self.policy.get("preprocess", {})
        self.assertIn("norm_stats_file", preprocess)

    def test_input_output_tensor_names(self):
        self.assertIn("input_tensor_names", self.policy)
        self.assertIn("output_tensor_names", self.policy)
        self.assertIn("obs_seq", self.policy["input_tensor_names"])
        self.assertIn("pred_action_seq", self.policy["output_tensor_names"])


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------

class TestArtifactExportNegative(unittest.TestCase):

    def test_onnxruntime_without_checkpoint_fails(self):
        _, rc = _run_export(extra_args=["--backend", "onnxruntime"], expect_success=False)
        self.assertNotEqual(rc, 0, "onnxruntime without checkpoint must fail")

    def test_seq_len_with_checkpoint_forbidden(self):
        """--seq_len must be rejected when --checkpoint is also given."""
        import tempfile, torch
        # Create a minimal fake checkpoint without seq_len to exercise the path
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            torch.save({"model": {}, "obs_mean": [0]*8, "obs_std": [1]*8,
                        "act_mean": [0]*3, "act_std": [1]*3, "args": {},
                        "seq_len": 20}, ckpt_path)
            _, rc = _run_export(
                extra_args=["--seq_len", "20", "--checkpoint", ckpt_path],
                expect_success=False,
            )
            self.assertNotEqual(rc, 0, "--seq_len with --checkpoint must be rejected")
        finally:
            Path(ckpt_path).unlink(missing_ok=True)

    def test_checkpoint_without_seq_len_fails(self):
        """Checkpoint missing 'seq_len' must cause export to fail."""
        import tempfile, torch
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            # Old-style checkpoint: no seq_len key
            torch.save({"model": {}, "obs_mean": [0]*8, "obs_std": [1]*8,
                        "act_mean": [0]*3, "act_std": [1]*3, "args": {}}, ckpt_path)
            _, rc = _run_export(
                checkpoint_path=ckpt_path,
                expect_success=False,
            )
            self.assertNotEqual(rc, 0, "Checkpoint without seq_len must cause export failure")
        finally:
            Path(ckpt_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
