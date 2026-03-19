"""Legacy torch checkpoint backend for backward compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from shared.models.transformer_bc import TransformerBC


class TorchCheckpointBackend:
    """Loads old .pt checkpoints emitted by ml/train_bc_transformer_smooth.py."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})

        self.model = TransformerBC(
            obs_dim=8,
            act_dim=3,
            d_model=int(args.get("d_model", 128)),
            nhead=int(args.get("heads", 4)),
            num_layers=int(args.get("layers", 4)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()

        self.obs_mean = torch.tensor(ckpt["obs_mean"], dtype=torch.float32, device=self.device)
        self.obs_std = torch.tensor(ckpt["obs_std"], dtype=torch.float32, device=self.device)
        self.act_mean = torch.tensor(ckpt["act_mean"], dtype=torch.float32, device=self.device)
        self.act_std = torch.tensor(ckpt["act_std"], dtype=torch.float32, device=self.device)

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        if obs_seq.ndim != 2:
            raise ValueError(f"obs_seq must be rank-2 [T,D], got shape={obs_seq.shape}")
        obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_n = (obs_t - self.obs_mean.view(1, 1, -1)) / self.obs_std.view(1, 1, -1)
        with torch.no_grad():
            pred_n = self.model(obs_n)
            pred = pred_n * self.act_std.view(1, 1, -1) + self.act_mean.view(1, 1, -1)
        return pred[0, -1, :].detach().cpu().numpy().astype(np.float32)
