"""Minimal VQ-VAE for action-chunk tokenisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """EMA-free vector quantizer with straight-through gradient estimator."""

    def __init__(self, codebook_size: int, latent_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.beta = beta  # commitment loss weight

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor):
        """
        Parameters
        ----------
        z : [B, latent_dim]

        Returns
        -------
        z_q : [B, latent_dim]  quantised, with straight-through grad
        vq_loss : scalar  (codebook + commitment)
        indices : [B]  codebook indices
        """
        # distances: [B, codebook_size]
        dists = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = dists.argmin(dim=1)  # [B]
        z_q = self.embedding(indices)  # [B, latent_dim]

        # losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + self.beta * commitment_loss

        # straight-through
        z_q = z + (z_q - z).detach()
        return z_q, vq_loss, indices


class ActionChunkVQVAE(nn.Module):
    """VQ-VAE that encodes / decodes action chunks of shape [token_steps, 3].

    Architecture
    ------------
    Encoder : flatten → 128 → latent_dim
    VQ      : codebook_size entries of latent_dim
    Decoder : latent_dim → 128 → flatten
    """

    def __init__(
        self,
        token_steps: int = 2,
        codebook_size: int = 128,
        latent_dim: int = 32,
        beta: float = 0.25,
    ):
        super().__init__()
        self.token_steps = token_steps
        self.input_dim = token_steps * 3
        self.latent_dim = latent_dim

        hidden = 128
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.vq = VectorQuantizer(codebook_size, latent_dim, beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : [B, token_steps, 3]

        Returns
        -------
        x_hat : [B, token_steps, 3]
        vq_loss : scalar
        indices : [B]
        """
        B = x.shape[0]
        flat = x.reshape(B, -1)  # [B, token_steps*3]
        z = self.encoder(flat)
        z_q, vq_loss, indices = self.vq(z)
        x_hat = self.decoder(z_q).reshape(B, self.token_steps, 3)
        return x_hat, vq_loss, indices
