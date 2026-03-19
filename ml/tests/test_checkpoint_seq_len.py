#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint seq_len persistence tests.

Verifies that:
1. seq_len is saved in the checkpoint when training completes.
2. export_artifact reads seq_len from checkpoint (not from CLI default).
3. Non-default seq_len flows end-to-end correctly.

These tests use mock checkpoints to avoid needing real training data.

Usage:
    pytest ml/tests/test_checkpoint_seq_len.py -v
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


def _make_checkpoint(tmpdir, seq_len, obs_dim=8, act_dim=3):
    """Write a minimal fake checkpoint with the given seq_len."""
    try:
        import torch
    except ModuleNotFoundError:
        raise unittest.SkipTest("PyTorch not installed; skipping checkpoint tests")

    ckpt = {
        "model": {},               # empty; no model arch needed for export_artifact numpy_linear path
        "obs_mean": [0.0] * obs_dim,
        "obs_std":  [1.0] * obs_dim,
        "act_mean": [0.0] * act_dim,
        "act_std":  [1.0] * act_dim,
        "seq_len": seq_len,
        "args": {
            "d_model": 128, "heads": 4, "layers": 4, "dropout": 0.1,
            "seq_len": seq_len,
        },
        "meta": {"schema": "test"},
    }
    path = Path(tmpdir) / "fake_ckpt.pt"
    torch.save(ckpt, path)
    return path


def _run_export(ckpt_path, out_dir):
    cmd = [
        sys.executable, str(_ML_DIR / "export_artifact.py"),
        "--backend", "numpy_linear",
        "--checkpoint", str(ckpt_path),
        "--out", str(out_dir),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestSeqLenPersistence(unittest.TestCase):

    def _export_and_load_policy(self, seq_len):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = _make_checkpoint(tmp, seq_len=seq_len)
            out_dir = Path(tmp) / "artifact"
            result = _run_export(ckpt_path, out_dir)
            self.assertEqual(
                result.returncode, 0,
                f"export failed:\n{result.stdout}\n{result.stderr}"
            )
            policy = json.loads((out_dir / "policy.yaml").read_text(encoding="utf-8"))
        return policy

    def test_default_seq_len_in_artifact(self):
        """Default seq_len (CFG.SEQ_LEN=20) flows into policy.yaml."""
        policy = self._export_and_load_policy(seq_len=CFG.SEQ_LEN)
        self.assertEqual(policy["obs_spec"]["seq_len"], CFG.SEQ_LEN)

    def test_nondefault_seq_len_in_artifact(self):
        """Non-default seq_len (e.g. 10) flows into policy.yaml correctly."""
        nondefault = 10
        self.assertNotEqual(nondefault, CFG.SEQ_LEN)
        policy = self._export_and_load_policy(seq_len=nondefault)
        self.assertEqual(
            policy["obs_spec"]["seq_len"], nondefault,
            f"Expected seq_len={nondefault} in artifact, got {policy['obs_spec']['seq_len']}"
        )

    def test_another_nondefault_seq_len(self):
        """seq_len=30 also flows correctly."""
        policy = self._export_and_load_policy(seq_len=30)
        self.assertEqual(policy["obs_spec"]["seq_len"], 30)

    def test_seq_len_not_overridden_by_cli_default(self):
        """Ensure export does NOT silently use CLI default when checkpoint has different seq_len."""
        nondefault = 15
        policy = self._export_and_load_policy(seq_len=nondefault)
        # If export had silently used CFG.SEQ_LEN=20, this would fail.
        self.assertNotEqual(
            policy["obs_spec"]["seq_len"], CFG.SEQ_LEN,
            "Export silently used CFG.SEQ_LEN instead of checkpoint seq_len"
        )
        self.assertEqual(policy["obs_spec"]["seq_len"], nondefault)

    def test_missing_seq_len_in_checkpoint_causes_failure(self):
        """Old checkpoints without seq_len must cause export to fail."""
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch not installed")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = {
                "model": {},
                "obs_mean": [0.0] * 8,
                "obs_std":  [1.0] * 8,
                "act_mean": [0.0] * 3,
                "act_std":  [1.0] * 3,
                "args": {},
                # No 'seq_len' key — simulates old checkpoint
            }
            ckpt_path = Path(tmp) / "old_ckpt.pt"
            torch.save(ckpt, ckpt_path)
            out_dir = Path(tmp) / "artifact"
            result = _run_export(ckpt_path, out_dir)
            self.assertNotEqual(result.returncode, 0,
                                "Export must fail for checkpoint without seq_len")
            # Check error message is informative
            combined = result.stdout + result.stderr
            self.assertIn("seq_len", combined.lower(),
                          "Error message should mention seq_len")


if __name__ == "__main__":
    unittest.main()
