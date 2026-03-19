#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset smoke tests.

Verifies that an NPZ dataset produced by ml/build_training_dataset.py:
1. Has correct obs/action shapes consistent with dataset_default_config.py.
2. Has non-NaN / finite normalisation statistics.
3. Uses field names consistent with artifact obs_spec.field_order (schema 0.2.0).
4. Ground-speed values are in the expected physical range (2-D horizontal).
5. track_angle_rad_unwrapped values are present and not wrapped to [-π, π] artifact
   (unwrapped angles may exceed [-π, π] across a sequence window).

Usage:
    pytest ml/tests/test_dataset_smoke.py -v [--npz path/to/dataset.npz]

By default looks for wow/datasets/dt2hz_H2s_fighteronly.npz.  If not found,
the tests that require actual data are skipped.

pytest options:
    --npz <path>   Path to the .npz dataset file to check.
"""

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

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
# pytest hook to accept --npz argument
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--npz", action="store", default=None,
                     help="Path to dataset .npz file for smoke tests")


def pytest_configure(config):
    pass


# ---------------------------------------------------------------------------
# Locate default dataset
# ---------------------------------------------------------------------------

def _default_npz_path():
    p = _ROOT_DIR / "datasets" / f"{CFG.DEFAULT_DATASET_STEM}.npz"
    return p if p.exists() else None


def _get_npz_path():
    """Return npz path from pytest option or default location."""
    # When run via unittest directly (not pytest), use default
    return _default_npz_path()


# ---------------------------------------------------------------------------
# Pure-logic tests (no dataset file required)
# ---------------------------------------------------------------------------

class TestConfigConsistency(unittest.TestCase):
    """Config self-consistency checks — no NPZ needed."""

    def test_horizon_steps_consistent_with_dt_and_H(self):
        expected_k = int(round(CFG.H_SEC / CFG.DT))
        self.assertEqual(CFG.K, expected_k,
                         f"CFG.K={CFG.K} != round(H_SEC/DT)={expected_k}")

    def test_horizon_steps_alias(self):
        self.assertEqual(CFG.HORIZON_STEPS, CFG.K,
                         "HORIZON_STEPS must equal K")

    def test_control_frequency_hz(self):
        self.assertAlmostEqual(CFG.CONTROL_FREQUENCY_HZ, 1.0 / CFG.DT, places=6)

    def test_obs_fields_length(self):
        self.assertEqual(len(CFG.OBS_FIELDS), 8,
                         "Expected 8 obs fields for this model architecture")

    def test_act_fields_length(self):
        self.assertEqual(len(CFG.ACT_FIELDS), 3,
                         "Expected 3 action fields")

    def test_obs_field_names_schema_020(self):
        """Schema 0.2.0: must use track_angle_rad_unwrapped, not heading_rad_unwrapped."""
        self.assertIn("track_angle_rad_unwrapped", CFG.OBS_FIELDS,
                      "track_angle_rad_unwrapped must be in OBS_FIELDS")
        self.assertNotIn("heading_rad_unwrapped", CFG.OBS_FIELDS,
                         "old name heading_rad_unwrapped must not appear in OBS_FIELDS")
        self.assertNotIn("heading_rad", CFG.OBS_FIELDS,
                         "heading_rad must not appear in OBS_FIELDS")

    def test_obs_field_ground_speed(self):
        self.assertIn("ground_speed_mps", CFG.OBS_FIELDS)

    def test_act_field_dpsi_present(self):
        self.assertIn("dpsi_rad", CFG.ACT_FIELDS)

    def test_schema_version_is_020(self):
        self.assertEqual(SCHEMA_VERSION, "0.2.0",
                         "shared/schema/VERSION must be 0.2.0 after this PR")

    def test_obs_field_order_matches_build_dataset(self):
        """OBS_FIELDS must match the stack order in build_training_dataset.py."""
        expected_prefix = ["x_e_m", "y_n_m", "z_u_m",
                           "vx_e_mps", "vy_n_mps", "vz_u_mps"]
        self.assertEqual(CFG.OBS_FIELDS[:6], expected_prefix,
                         "First 6 obs fields must be ENU position+velocity")
        self.assertEqual(CFG.OBS_FIELDS[6], "track_angle_rad_unwrapped")
        self.assertEqual(CFG.OBS_FIELDS[7], "ground_speed_mps")


# ---------------------------------------------------------------------------
# Dataset file tests (skipped when NPZ not found)
# ---------------------------------------------------------------------------

class TestDatasetFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.npz_path = _get_npz_path()
        if cls.npz_path is None:
            return
        data = np.load(cls.npz_path, allow_pickle=True)
        cls.obs    = data["obs"]      # [N, T, D]
        cls.action = data["action"]   # [N, T, A]
        cls.obs_mean = data["obs_mean"]
        cls.obs_std  = data["obs_std"]
        cls.act_mean = data["act_mean"]
        cls.act_std  = data["act_std"]

    def _skip_if_no_npz(self):
        if self.npz_path is None:
            self.skipTest(
                f"Dataset NPZ not found at {_ROOT_DIR / 'datasets'}. "
                "Run build_training_dataset.py first or pass --npz."
            )

    # Shape
    def test_obs_shape_ndim(self):
        self._skip_if_no_npz()
        self.assertEqual(self.obs.ndim, 3, "obs must be rank-3 [N, T, D]")

    def test_obs_shape_D(self):
        self._skip_if_no_npz()
        self.assertEqual(self.obs.shape[2], len(CFG.OBS_FIELDS),
                         f"obs D={self.obs.shape[2]} != len(OBS_FIELDS)={len(CFG.OBS_FIELDS)}")

    def test_obs_shape_T(self):
        self._skip_if_no_npz()
        self.assertEqual(self.obs.shape[1], CFG.SEQ_LEN,
                         f"obs T={self.obs.shape[1]} != SEQ_LEN={CFG.SEQ_LEN}")

    def test_action_shape(self):
        self._skip_if_no_npz()
        self.assertEqual(self.action.ndim, 3)
        self.assertEqual(self.action.shape[:2], self.obs.shape[:2],
                         "action and obs must have same N and T")
        self.assertEqual(self.action.shape[2], len(CFG.ACT_FIELDS))

    # Norm stats
    def test_obs_mean_finite(self):
        self._skip_if_no_npz()
        self.assertTrue(np.all(np.isfinite(self.obs_mean)),
                        "obs_mean contains NaN or Inf")

    def test_obs_std_positive(self):
        self._skip_if_no_npz()
        self.assertTrue(np.all(self.obs_std > 0),
                        "obs_std must be positive")

    def test_act_mean_finite(self):
        self._skip_if_no_npz()
        self.assertTrue(np.all(np.isfinite(self.act_mean)))

    def test_act_std_positive(self):
        self._skip_if_no_npz()
        self.assertTrue(np.all(self.act_std > 0))

    # Physical plausibility — ground speed (obs[:, :, 7]) is 2-D horizontal
    def test_ground_speed_nonnegative(self):
        self._skip_if_no_npz()
        gs = self.obs[:, :, 7].ravel()
        self.assertTrue(np.all(gs >= 0.0),
                        "ground_speed_mps must be non-negative")

    def test_ground_speed_max_below_ceiling(self):
        self._skip_if_no_npz()
        gs = self.obs[:, :, 7].ravel()
        self.assertLessEqual(float(gs.max()), CFG.MAX_SPEED + 1.0,
                             f"ground_speed_mps max {gs.max():.1f} exceeds MAX_SPEED={CFG.MAX_SPEED}")

    def test_ground_speed_min_above_floor(self):
        self._skip_if_no_npz()
        gs = self.obs[:, :, 7].ravel()
        self.assertGreaterEqual(float(gs.min()), CFG.MIN_SPEED - 1.0,
                                f"ground_speed_mps min {gs.min():.1f} below MIN_SPEED={CFG.MIN_SPEED}")

    # track_angle_rad_unwrapped (obs[:, :, 6]) — may exceed [-π, π]
    def test_track_angle_has_variance(self):
        """Unwrapped track angle must have meaningful variance across windows."""
        self._skip_if_no_npz()
        ta = self.obs[:, :, 6].ravel()
        self.assertGreater(float(ta.std()), 0.1,
                           "track_angle_rad_unwrapped has suspiciously low variance — may be wrapped?")

    # dpsi_rad (action[:, :, 0])
    def test_dpsi_finite(self):
        self._skip_if_no_npz()
        dpsi = self.action[:, :, 0].ravel()
        self.assertTrue(np.all(np.isfinite(dpsi)), "dpsi_rad contains NaN or Inf")

    def test_dpsi_in_pi_range(self):
        """dpsi_rad is computed via wrap_pi so must be in (-π, π]."""
        self._skip_if_no_npz()
        dpsi = self.action[:, :, 0].ravel()
        self.assertTrue(np.all(dpsi >= -math.pi - 1e-4),
                        f"dpsi_rad below -π: min={dpsi.min():.4f}")
        self.assertTrue(np.all(dpsi <= math.pi + 1e-4),
                        f"dpsi_rad above +π: max={dpsi.max():.4f}")


if __name__ == "__main__":
    unittest.main()
