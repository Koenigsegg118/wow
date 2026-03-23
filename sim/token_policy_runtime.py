#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token policy inference runtime.

Loads the frozen OneStepTokenBC policy and VQ-VAE decoder, wraps them into
a single ``predict_and_decode`` call that maps:

    (obs_hist[H,8], prev_token)  →  (dense_token, action_chunk[4,3], confidence)

The caller passes **raw** (non-ego-transformed) observation history at
token-boundary timesteps.  The runtime applies ego transform (current-step
anchor), normalizes, and runs inference.

All checkpoint paths, vocab mapping, and normalization stats are resolved
at construction time.  The caller never touches torch directly.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from training.vq.ego_obs_utils import ego_transform_current_anchor
from training.vq.train_onestep_token_bc import OneStepTokenBC
from training.vq.vqvae_model import ActionChunkVQVAE


@dataclass
class TokenRuntimeConfig:
    policy_ckpt: str
    vq_ckpt: str
    vocab_json: str
    device: str = "cpu"


class TokenPolicyRuntime:
    """Stateless inference wrapper around frozen OneStepTokenBC + VQ-VAE."""

    def __init__(self, cfg: TokenRuntimeConfig):
        self.device = torch.device(cfg.device)

        # ── load policy ──────────────────────────────────────
        pol_ckpt = torch.load(cfg.policy_ckpt, map_location=self.device,
                              weights_only=False)
        pa = pol_ckpt["args"]
        self.vocab_size: int = int(pol_ckpt["vocab_size"])
        self.obs_hist_len: int = int(pol_ckpt["obs_hist_len"])
        self.obs_mean = np.asarray(pol_ckpt["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(pol_ckpt["obs_std"], dtype=np.float32)
        self.bos_token: int = self.vocab_size  # sentinel for first step

        self.policy = OneStepTokenBC(
            vocab_size=self.vocab_size,
            obs_dim=self.obs_mean.shape[0],
            obs_hist_len=self.obs_hist_len,
            hidden_dim=pa["hidden_dim"],
            n_layers=pa["num_layers"],
            dropout=0.0,
        )
        self.policy.load_state_dict(pol_ckpt["model_state_dict"])
        self.policy.to(self.device).eval()

        # ── load VQ-VAE (decoder only) ───────────────────────
        vq_ckpt = torch.load(cfg.vq_ckpt, map_location=self.device,
                             weights_only=False)
        va = vq_ckpt["args"]
        self.token_steps: int = int(va["token_steps"])
        self.codebook_size: int = int(va["codebook_size"])

        self.vqvae = ActionChunkVQVAE(
            token_steps=self.token_steps,
            codebook_size=self.codebook_size,
            latent_dim=va.get("latent_dim", 32),
        )
        self.vqvae.load_state_dict(vq_ckpt["model_state_dict"])
        self.vqvae.to(self.device).eval()

        self.normalize_action: bool = bool(va.get("normalize_action", False))
        action_stats = vq_ckpt.get("action_stats", {})
        if self.normalize_action and action_stats:
            fields = action_stats["fields"]
            self.act_mean = np.array(
                [action_stats["mean"][k] for k in fields], dtype=np.float32)
            self.act_std = np.array(
                [action_stats["std"][k] for k in fields], dtype=np.float32)
        else:
            self.act_mean = np.zeros(3, dtype=np.float32)
            self.act_std = np.ones(3, dtype=np.float32)

        # ── load vocab mapping ───────────────────────────────
        with open(cfg.vocab_json, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.dense_to_raw: Dict[int, int] = {
            int(k): int(v)
            for k, v in vocab_data["dense_id_to_raw_code"].items()
        }
        loaded_vocab = int(vocab_data.get("active_vocab_size",
                                          len(self.dense_to_raw)))
        if loaded_vocab != self.vocab_size:
            raise ValueError(
                f"Vocab size mismatch: policy checkpoint has {self.vocab_size}, "
                f"vocab JSON has {loaded_vocab}"
            )

    @torch.no_grad()
    def predict_and_decode(
        self, obs_hist: np.ndarray, prev_token: int,
    ) -> tuple[int, np.ndarray, float]:
        """Run one-step token prediction and decode to action chunk.

        Args:
            obs_hist: observation history, shape ``[H, 8]`` (raw,
                pre-ego-transform).  ``H`` must equal ``self.obs_hist_len``.
                The **last** row is the current decision step (anchor).
            prev_token: dense token id from previous step, or ``bos_token``
                for the first prediction.

        Returns:
            dense_token: predicted dense token id (0 .. vocab_size-1).
            action_chunk: [token_steps, 3] denormalized relative actions
                ``[dpsi_rad, dalt_sp_m, dspd_sp_mps]``.
            confidence: softmax max probability of the predicted token.
        """
        obs_ego = ego_transform_current_anchor(
            obs_hist.astype(np.float32))              # [H, 8]
        obs_norm = (obs_ego - self.obs_mean) / np.maximum(self.obs_std, 1e-6)
        obs_t = torch.from_numpy(
            obs_norm[None, :, :].astype(np.float32)   # [1, H, 8]
        ).to(self.device)
        prev_t = torch.tensor([prev_token], dtype=torch.long,
                              device=self.device)

        logits = self.policy(obs_t, prev_t)           # [1, vocab_size]
        probs = F.softmax(logits, dim=-1)
        confidence = float(probs.max().item())
        dense_token = int(logits.argmax(dim=-1).item())

        raw_token = self.dense_to_raw[dense_token]
        raw_t = torch.tensor([raw_token], dtype=torch.long,
                             device=self.device)
        z_q = self.vqvae.vq.embedding(raw_t)           # [1, latent_dim]
        chunk_flat = self.vqvae.decoder(z_q)            # [1, token_steps*3]
        chunk = chunk_flat.cpu().numpy().reshape(self.token_steps, 3)

        if self.normalize_action:
            chunk = chunk * self.act_std[None, :] + self.act_mean[None, :]

        return dense_token, chunk.astype(np.float32), confidence

    CONTRACT_VERSION = "onestep_h4_current_anchor_ego_v1"

    def info(self) -> dict:
        """Return a summary dict for logging."""
        return {
            "contract_version": self.CONTRACT_VERSION,
            "vocab_size": self.vocab_size,
            "obs_hist_len": self.obs_hist_len,
            "token_steps": self.token_steps,
            "codebook_size": self.codebook_size,
            "bos_token": self.bos_token,
            "normalize_action": self.normalize_action,
            "device": str(self.device),
        }
