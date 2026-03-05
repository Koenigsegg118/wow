#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for wrap/unwrap and dpsi utilities used in coverage analysis.

Run with:
    pytest wow/training/tests/test_coverage_utils.py -v
or from the wow/ directory:
    pytest training/tests/test_coverage_utils.py -v
"""

import math
import os
import sys
import unittest

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WOW_DIR  = os.path.dirname(_THIS_DIR)
if _WOW_DIR not in sys.path:
    sys.path.insert(0, _WOW_DIR)

from acmi_to_dt_dataset_smooth import (
    wrap_pi,
    unwrap_angle,
    compute_dpsi_series,
    latlon_to_enu_m,
)
from analyze_acmi_coverage import histogram_entropy, quality_grade


# ── wrap_pi ──────────────────────────────────────────────────────────────────
class TestWrapPi(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(wrap_pi(0.0), 0.0)

    def test_positive_small(self):
        self.assertAlmostEqual(wrap_pi(1.0), 1.0)

    def test_negative_small(self):
        self.assertAlmostEqual(wrap_pi(-1.0), -1.0)

    def test_pi_maps_to_minus_pi(self):
        # wrap_pi uses (x+pi)%(2pi)-pi → range is [-pi, pi)
        # so +pi maps to -pi
        self.assertAlmostEqual(wrap_pi(math.pi), -math.pi, places=10)

    def test_beyond_pi(self):
        # 4.0 rad > π — should wrap to 4.0 - 2π ≈ -2.283
        expected = 4.0 - 2 * math.pi
        self.assertAlmostEqual(wrap_pi(4.0), expected, places=10)

    def test_beyond_minus_pi(self):
        # -4.0 rad < -π — should wrap to -4.0 + 2π ≈ +2.283
        expected = -4.0 + 2 * math.pi
        self.assertAlmostEqual(wrap_pi(-4.0), expected, places=10)

    def test_in_range_unchanged(self):
        # 3.0 rad is already in [-π, π) — wrap_pi must leave it unchanged
        self.assertAlmostEqual(wrap_pi(3.0),  3.0,  places=10)
        self.assertAlmostEqual(wrap_pi(-3.0), -3.0, places=10)

    def test_two_pi(self):
        self.assertAlmostEqual(wrap_pi(2 * math.pi), 0.0, places=10)

    def test_output_in_range(self):
        for deg in range(-540, 541, 15):
            x = math.radians(deg)
            w = wrap_pi(x)
            self.assertGreaterEqual(w, -math.pi - 1e-9,
                                    msg=f"wrap_pi({deg}°) = {w} < -π")
            self.assertLess(w, math.pi + 1e-9,
                            msg=f"wrap_pi({deg}°) = {w} >= π")

    def test_idempotent(self):
        for deg in range(-180, 181, 10):
            x = math.radians(deg)
            w = wrap_pi(x)
            self.assertAlmostEqual(wrap_pi(w), w, places=10,
                                   msg=f"wrap_pi not idempotent at {deg}°")


# ── unwrap_angle ─────────────────────────────────────────────────────────────
class TestUnwrapAngle(unittest.TestCase):
    def test_empty(self):
        result = unwrap_angle(np.array([]))
        self.assertEqual(len(result), 0)

    def test_single(self):
        result = unwrap_angle(np.array([1.0]))
        self.assertAlmostEqual(result[0], 1.0)

    def test_constant(self):
        arr = np.full(10, 0.5)
        result = unwrap_angle(arr)
        np.testing.assert_allclose(result, arr)

    def test_no_jumps(self):
        arr = np.linspace(0.0, math.pi * 0.9, 20)
        result = unwrap_angle(arr)
        np.testing.assert_allclose(result, arr, atol=1e-10)

    def test_crossing_pi_upward(self):
        # Sequence that wraps from just-below π to just-above π (crosses 2π boundary).
        # In wrapped representation this looks like a -2π jump; unwrap should remove it.
        n = 40
        true_heading = np.linspace(math.pi - 0.5, math.pi + 0.5, n)
        wrapped = (true_heading + math.pi) % (2 * math.pi) - math.pi
        unwrapped = unwrap_angle(wrapped)
        # After unwrapping, differences should be small
        diffs = np.abs(np.diff(unwrapped))
        self.assertTrue(np.all(diffs < 0.1),
                        msg=f"Large jump after unwrap: max diff = {diffs.max():.4f}")

    def test_multiple_revolutions(self):
        # 0 → 4π — large continuous rotation
        true_heading = np.linspace(0.0, 4 * math.pi, 100)
        wrapped = (true_heading + math.pi) % (2 * math.pi) - math.pi
        unwrapped = unwrap_angle(wrapped)
        # Unwrapped should follow the original up to a constant offset
        drift = unwrapped - true_heading
        # The drift should be nearly constant (same unwrap branch)
        self.assertAlmostEqual(np.ptp(drift), 0.0, places=9,
                               msg="unwrap drift varies — branch inconsistency")

    def test_decreasing_through_minus_pi(self):
        # Sequence going from -π+ε downward through the boundary
        true_heading = np.linspace(-math.pi + 0.2, -math.pi - 0.2, 20)
        wrapped = (true_heading + math.pi) % (2 * math.pi) - math.pi
        unwrapped = unwrap_angle(wrapped)
        diffs = np.abs(np.diff(unwrapped))
        self.assertTrue(np.all(diffs < 0.1),
                        msg=f"Large jump in decreasing unwrap: {diffs.max():.4f}")


# ── compute_dpsi_series ───────────────────────────────────────────────────────
class TestComputeDpsiSeries(unittest.TestCase):
    def test_constant_heading_zero_dpsi(self):
        heading_u = np.full(20, 0.5)
        dpsi = compute_dpsi_series(heading_u, horizon_steps=4, dpsi_sign=1.0)
        np.testing.assert_allclose(dpsi, 0.0, atol=1e-10)

    def test_output_length(self):
        n = 30
        k = 4
        heading_u = np.linspace(0.0, 1.0, n)
        dpsi = compute_dpsi_series(heading_u, horizon_steps=k, dpsi_sign=1.0)
        self.assertEqual(len(dpsi), n - k)

    def test_linear_heading_uniform_dpsi(self):
        n = 24
        k = 4
        heading_u = np.linspace(0.0, 1.0, n)   # already unwrapped
        dpsi = compute_dpsi_series(heading_u, horizon_steps=k, dpsi_sign=1.0)
        # Each step = 1/(n-1), over k steps = k/(n-1)
        expected = k / (n - 1)
        np.testing.assert_allclose(dpsi, expected, atol=1e-6)

    def test_sign_flip(self):
        heading_u = np.linspace(0.0, 1.0, 20)
        dpsi_pos = compute_dpsi_series(heading_u, horizon_steps=4, dpsi_sign=+1.0)
        dpsi_neg = compute_dpsi_series(heading_u, horizon_steps=4, dpsi_sign=-1.0)
        np.testing.assert_allclose(dpsi_pos, -dpsi_neg, atol=1e-10)

    def test_output_in_minus_pi_to_pi(self):
        """dpsi must stay in [-π, π] after wrap_pi."""
        rng = np.random.default_rng(42)
        heading_u = np.cumsum(rng.uniform(-0.5, 0.5, 100))
        dpsi = compute_dpsi_series(heading_u, horizon_steps=4, dpsi_sign=1.0)
        self.assertTrue(np.all(dpsi >= -math.pi - 1e-9),  "dpsi < -π found")
        self.assertTrue(np.all(dpsi <   math.pi + 1e-9),  "dpsi >= π found")

    def test_wrap_across_pi_boundary(self):
        # Heading near π crossing; wrap_pi should give small delta, not ≈ ±2π
        eps = 0.05
        # Approach π from below, then cross it
        heading_u = np.array([math.pi - eps] * 6 + [math.pi + eps] * 6)
        # We use already-wrapped values as input; unwrap first to be consistent
        heading_u = unwrap_angle(heading_u)
        dpsi = compute_dpsi_series(heading_u, horizon_steps=4, dpsi_sign=1.0)
        for d in dpsi:
            self.assertLess(abs(d), math.pi + 1e-9,
                            msg=f"dpsi={d:.4f} exceeds π — wrap failure")

    def test_horizon_steps_1(self):
        heading_u = np.deg2rad(np.array([0.0, 10.0, 20.0, 30.0], dtype=float))
        dpsi = compute_dpsi_series(heading_u, horizon_steps=1, dpsi_sign=1.0)
        np.testing.assert_allclose(np.rad2deg(dpsi), [10.0, 10.0, 10.0], atol=1e-5)


# ── latlon_to_enu_m ───────────────────────────────────────────────────────────
class TestLatlonToEnuM(unittest.TestCase):
    def test_origin_zero(self):
        x, y = latlon_to_enu_m(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(x, 0.0, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)

    def test_east_positive(self):
        # Moving 1° east at equator ≈ 111195 m
        x, y = latlon_to_enu_m(0.0, 1.0, 0.0, 0.0)
        self.assertGreater(x, 0, "East should be positive x")
        self.assertAlmostEqual(x / 1000, 111.195, delta=1.0)

    def test_north_positive(self):
        # Moving 1° north ≈ 111195 m
        x, y = latlon_to_enu_m(1.0, 0.0, 0.0, 0.0)
        self.assertGreater(y, 0, "North should be positive y")
        self.assertAlmostEqual(y / 1000, 111.195, delta=1.0)

    def test_symmetry(self):
        x1, y1 = latlon_to_enu_m(42.0,  1.0, 42.0, 0.0)
        x2, y2 = latlon_to_enu_m(42.0, -1.0, 42.0, 0.0)
        self.assertAlmostEqual(x1, -x2, places=1)
        self.assertAlmostEqual(y1,  y2,  places=1)


# ── histogram_entropy ─────────────────────────────────────────────────────────
class TestHistogramEntropy(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(histogram_entropy(np.array([])), 0.0)

    def test_uniform_higher_than_peaked(self):
        uniform = np.random.default_rng(0).uniform(0, 1, 1000)
        peaked  = np.random.default_rng(0).normal(0.5, 0.01, 1000)
        self.assertGreater(histogram_entropy(uniform), histogram_entropy(peaked))

    def test_constant_array_low_entropy(self):
        const = np.ones(100)
        self.assertLess(histogram_entropy(const), 0.01)

    def test_non_negative(self):
        arr = np.random.default_rng(7).exponential(1.0, 500)
        self.assertGreaterEqual(histogram_entropy(arr), 0.0)


# ── quality_grade ─────────────────────────────────────────────────────────────
class TestQualityGrade(unittest.TestCase):
    def test_A_grade(self):
        self.assertEqual(quality_grade(600, 0.02), "A")

    def test_B_grade_by_count(self):
        self.assertEqual(quality_grade(200, 0.02), "B")

    def test_B_grade_by_outlier(self):
        self.assertEqual(quality_grade(600, 0.10), "B")

    def test_C_grade(self):
        self.assertEqual(quality_grade(25, 0.20), "C")

    def test_F_grade(self):
        self.assertEqual(quality_grade(5, 0.50), "F")

    def test_boundary_A(self):
        # Exactly at A threshold
        self.assertEqual(quality_grade(500, 0.05), "B")   # 0.05 is NOT < 0.05
        self.assertEqual(quality_grade(500, 0.04), "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
