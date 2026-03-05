#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Transformer Behavior Cloning on tacview_dt_dataset (.npz)

Inputs : obs sequence  [B, T, obs_dim]
Targets: action seq    [B, T, act_dim]  (dpsi_rad, alt_sp_m, spd_sp_mps)
Loss   : weighted MSE/Huber + smoothness regularizer on predicted actions

Usage:
  python train_bc_transformer_smooth.py --data tacview_dt_dataset.npz --save bc_transformer.pt
"""

import argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NpzSeqDataset(Dataset):
    def __init__(self, path, split="train", val_frac=0.05, seed=0):
        data = np.load(path, allow_pickle=True)
        self.obs = data["obs"].astype(np.float32)      # [N, T, D]
        self.act = data["action"].astype(np.float32)   # [N, T, A]
        self.obs_mean = data["obs_mean"].astype(np.float32)
        self.obs_std  = data["obs_std"].astype(np.float32)
        self.act_mean = data["act_mean"].astype(np.float32)
        self.act_std  = data["act_std"].astype(np.float32)
        self.meta = json.loads(data["meta"].tobytes().decode("utf-8"))

        n = self.obs.shape[0]
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(n * val_frac))
        self.val_idx = idx[:n_val]
        self.tr_idx  = idx[n_val:]
        self.idx = self.tr_idx if split == "train" else self.val_idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        obs = (self.obs[j] - self.obs_mean) / self.obs_std
        act = (self.act[j] - self.act_mean) / self.act_std
        return torch.from_numpy(obs), torch.from_numpy(act)

class TransformerBC(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(obs_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, act_dim))

    def forward(self, obs_seq):
        x = self.in_proj(obs_seq)
        x = self.encoder(x)  # non-causal: uses full window context
        return self.out(x)

def huber(x, delta=1.0):
    ax = x.abs()
    return torch.where(ax < delta, 0.5*x*x, delta*(ax - 0.5*delta))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--save", default="bc_transformer.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)

    # per-dimension weights (normalized space)
    ap.add_argument("--w_dpsi", type=float, default=1.0)
    ap.add_argument("--w_alt",  type=float, default=0.01)
    ap.add_argument("--w_spd",  type=float, default=0.1)

    ap.add_argument("--lambda_smooth", type=float, default=0.5)
    ap.add_argument("--smooth_order", type=int, default=1, choices=[1,2])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = NpzSeqDataset(args.data, split="train", val_frac=args.val_frac, seed=args.seed)
    ds_va = NpzSeqDataset(args.data, split="val",   val_frac=args.val_frac, seed=args.seed)
    obs_dim = ds_tr.obs.shape[-1]
    act_dim = ds_tr.act.shape[-1]

    tr_loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=args.batch, shuffle=False, drop_last=False)

    model = TransformerBC(obs_dim, act_dim, d_model=args.d_model, nhead=args.heads,
                          num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    w = torch.tensor([args.w_dpsi, args.w_alt, args.w_spd], device=device).view(1,1,act_dim)

    def loss_fn(pred, target):
        err = pred - target
        l_dpsi = err[...,0]**2
        l_alt  = huber(err[...,1], delta=1.0)
        l_spd  = huber(err[...,2], delta=1.0)
        base = (args.w_dpsi*l_dpsi + args.w_alt*l_alt + args.w_spd*l_spd).mean()

        if args.lambda_smooth > 0:
            if args.smooth_order == 1:
                d = pred[:,1:,:] - pred[:,:-1,:]
            else:
                d1 = pred[:,1:,:] - pred[:,:-1,:]
                d = d1[:,1:,:] - d1[:,:-1,:]
            smooth = ((d*d)*w).mean()
            return base + args.lambda_smooth * smooth
        return base

    best_val = float("inf")
    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss, ntr = 0.0, 0
        for obs, act in tr_loader:
            obs, act = obs.to(device), act.to(device)
            pred = model(obs)
            loss = loss_fn(pred, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()) * obs.size(0)
            ntr += obs.size(0)
        tr_loss /= max(1, ntr)

        model.eval()
        va_loss, nva = 0.0, 0
        with torch.no_grad():
            for obs, act in va_loader:
                obs, act = obs.to(device), act.to(device)
                pred = model(obs)
                loss = loss_fn(pred, act)
                va_loss += float(loss.item()) * obs.size(0)
                nva += obs.size(0)
        va_loss /= max(1, nva)

        print(f"epoch {ep:03d} train {tr_loss:.6f} val {va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            ckpt = {
                "model": model.state_dict(),
                "obs_mean": ds_tr.obs_mean,
                "obs_std": ds_tr.obs_std,
                "act_mean": ds_tr.act_mean,
                "act_std": ds_tr.act_std,
                "meta": ds_tr.meta,
                "args": vars(args),
            }
            torch.save(ckpt, args.save)
            print(f"  saved best -> {args.save}")

if __name__ == "__main__":
    main()
