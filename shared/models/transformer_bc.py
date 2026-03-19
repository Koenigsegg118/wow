#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Transformer BC model used by both sim serving and ml training."""

import torch.nn as nn


class TransformerBC(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(obs_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, act_dim))

    def forward(self, obs_seq):
        x = self.in_proj(obs_seq)
        x = self.encoder(x)  # non-causal: uses full window context
        return self.out(x)
