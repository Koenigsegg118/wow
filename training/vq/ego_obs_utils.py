"""Current-step anchored ego-centric observation transform.

This module defines the canonical obs preprocessing contract shared between
training (one-step token policy) and runtime (AFSIM bridge).

Obs layout: [x_e, y_n, z_u, vx_e, vy_n, vz_u, heading_unwrapped, speed]
             0     1    2    3     4     5     6                    7

Transform (anchored at the LAST timestep of the history window):
  1. Position:  dx = x - x_anchor,  dy = y - y_anchor
  2. Rotation:  rotate (dx, dy, vx, vy) by -heading_anchor
                so that the current heading aligns to north (0)
  3. Heading:   heading' = heading - heading_anchor  (starts at 0 for anchor)
  4. Unchanged: z_u (absolute altitude), vz_u, speed

Output layout: [dx', dy', z_u, vx', vy', vz_u, heading', speed]
"""

from __future__ import annotations

import numpy as np


def ego_transform_current_anchor(obs_hist: np.ndarray) -> np.ndarray:
    """Apply current-step anchored ego-centric transform.

    Args:
        obs_hist: [H, 8] or [B, H, 8]  observation history.
                  The LAST timestep (index -1) is the current decision step
                  and serves as the anchor.

    Returns:
        transformed: same shape, with position/velocity/heading rotated
                     relative to the anchor (last) timestep.
    """
    single = obs_hist.ndim == 2
    if single:
        obs_hist = obs_hist[None, :, :]  # [1, H, 8]

    out = obs_hist.copy()

    # anchor = last timestep
    x0 = obs_hist[:, -1, 0]   # [B]
    y0 = obs_hist[:, -1, 1]   # [B]
    h0 = obs_hist[:, -1, 6]   # [B]

    cos_h = np.cos(-h0)  # [B]
    sin_h = np.sin(-h0)  # [B]

    # expand for broadcasting over H dimension
    cos_h = cos_h[:, None]  # [B, 1]
    sin_h = sin_h[:, None]  # [B, 1]

    # 1. position: relative to anchor, then rotate
    dx = obs_hist[:, :, 0] - x0[:, None]   # [B, H]
    dy = obs_hist[:, :, 1] - y0[:, None]   # [B, H]
    out[:, :, 0] = cos_h * dx + sin_h * dy
    out[:, :, 1] = -sin_h * dx + cos_h * dy

    # 2. velocity: rotate
    vx = obs_hist[:, :, 3]  # [B, H]
    vy = obs_hist[:, :, 4]  # [B, H]
    out[:, :, 3] = cos_h * vx + sin_h * vy
    out[:, :, 4] = -sin_h * vx + cos_h * vy

    # 3. heading: relative to anchor
    out[:, :, 6] = obs_hist[:, :, 6] - h0[:, None]

    # z_u (2), vz_u (5), speed (7) unchanged

    if single:
        return out[0]
    return out


def extract_token_boundary_history(
    obs_full: np.ndarray,
    token_pos: int,
    token_steps: int,
    history_len: int,
) -> np.ndarray:
    """Extract obs at token boundary timesteps for a single window.

    Token boundary times within a 20-step window (token_steps=4):
        pos 0 → t=0,  pos 1 → t=4,  pos 2 → t=8,  pos 3 → t=12,  pos 4 → t=16

    For a given token_pos j, the current decision time is t_cur = j * token_steps.
    We look back history_len steps at token boundary times:
        [j - history_len + 1, ..., j-1, j] mapped to obs_full timestep indices.

    If any history index falls before the window start, left-pad with the
    earliest available obs.

    Args:
        obs_full:    [20, 8]  full window observation
        token_pos:   int, 0..4  current token position
        token_steps: int, e.g. 4
        history_len: int, e.g. 4

    Returns:
        obs_hist: [history_len, 8]  observation history at token boundaries
    """
    obs_dim = obs_full.shape[-1]
    t_cur = token_pos * token_steps

    # collect token boundary timestep indices
    t_indices = []
    for h in range(history_len - 1, -1, -1):
        j_hist = token_pos - h
        if j_hist >= 0:
            t_indices.append(j_hist * token_steps)
        else:
            t_indices.append(-1)  # needs padding

    result = np.zeros((history_len, obs_dim), dtype=obs_full.dtype)
    # earliest valid obs for padding
    first_valid_t = t_indices[0]
    for i, t_idx in enumerate(t_indices):
        if t_idx < 0 or t_idx >= obs_full.shape[0]:
            # left-pad with first available obs in the history
            pad_t = next((t for t in t_indices if t >= 0), 0)
            result[i] = obs_full[pad_t]
        else:
            result[i] = obs_full[t_idx]

    return result
