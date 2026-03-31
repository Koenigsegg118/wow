"""V2 geometric features for ACMI mining.

Extends the v1 geometry helpers in ``mine_candidate_events.py`` with
rate-of-change features, neighbour counts, and coordination checks.

Sign / unit conventions follow ``sim/runtime/afsim_obs_geometry.py``:
  heading   : 0 = North, CW positive, degrees
  bearing   : 0 = nose, positive = starboard, [-180, 180] degrees
  aspect    : 0 = head-on, 180 = tail-on, [0, 180] degrees
  closure   : negative = closing, positive = opening, m/s
  range     : metres (always >= 0)
  rates     : deg/s  or  m/s  as labelled

All functions are pure-numpy; no imports from the main VQ/bridge/training
pipeline.
"""

from __future__ import annotations

import numpy as np


# ===================================================================
# Internal helpers
# ===================================================================

def _wrap_pm180(angles_deg: np.ndarray) -> np.ndarray:
    """Wrap an array of angles to [-180, 180] degrees."""
    return (angles_deg + 180.0) % 360.0 - 180.0


def _angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed shortest-arc difference *a - b* in degrees, range [-180, 180]."""
    return _wrap_pm180(a - b)


def _median_smooth_3(x: np.ndarray) -> np.ndarray:
    """Lightweight 3-point running median (edges are kept as-is).

    Parameters
    ----------
    x : 1-D array, length >= 1.

    Returns
    -------
    Smoothed array of same length.
    """
    if x.size < 3:
        return x.copy()
    out = x.copy()
    # vectorised interior: stack shifted views and take median along axis 0
    out[1:-1] = np.median(
        np.stack([x[:-2], x[1:-1], x[2:]], axis=0), axis=0
    )
    return out


# ===================================================================
# 1. heading_rate_degps
# ===================================================================

def heading_rate_degps(headings_deg: np.ndarray, dt: float) -> np.ndarray:
    """Rate of heading change in deg/s.

    Parameters
    ----------
    headings_deg : 1-D array of headings in degrees (0 = North, CW+).
    dt : timestep in seconds (must be > 0).

    Returns
    -------
    1-D array of same length; heading rate in **deg/s**.
    Positive = turning clockwise, negative = counter-clockwise.

    Notes
    -----
    * A 3-point median filter is applied before differencing to reject
      single-frame heading spikes (common in ACMI telemetry).
    * Wraparound at +/-180 is handled via angular differencing so that
      a transition from 179 to -179 yields +2 deg, not -358 deg.
    * Edge frames use forward / backward difference (numpy.gradient).
    """
    headings_deg = np.asarray(headings_deg, dtype=np.float64)
    if headings_deg.size == 0:
        return np.array([], dtype=np.float64)
    if headings_deg.size == 1:
        return np.array([0.0])

    smoothed = _median_smooth_3(headings_deg)

    # angular differences (handles wraparound)
    diffs = _angular_diff_deg(smoothed[1:], smoothed[:-1])  # length N-1

    # use numpy.gradient-style edge padding: forward / backward at ends
    rate = np.empty_like(smoothed)
    rate[1:-1] = (diffs[1:] + diffs[:-1]) / 2.0 / dt  # central
    rate[0] = diffs[0] / dt                             # forward
    rate[-1] = diffs[-1] / dt                           # backward
    return rate


# ===================================================================
# 2. range_rate_smooth_mps
# ===================================================================

def range_rate_smooth_mps(
    ranges_m: np.ndarray,
    dt: float,
    window: int = 3,
) -> np.ndarray:
    """Smoothed range-rate (closure) in m/s.

    Parameters
    ----------
    ranges_m : 1-D array of slant ranges in metres.
    dt : timestep in seconds.
    window : smoothing window for the running mean applied *before*
             central differencing.  Must be odd and >= 1.

    Returns
    -------
    1-D array of same length; rate in **m/s**.
    Negative = approaching, positive = separating (same sign convention
    as ``compute_closure`` in ``afsim_obs_geometry.py``).

    Notes
    -----
    Central difference: ``(r[i+1] - r[i-1]) / (2 * dt)`` with
    forward/backward at edges.  A running-mean pre-smooth reduces noise
    from single-frame position jitter.
    """
    ranges_m = np.asarray(ranges_m, dtype=np.float64)
    if ranges_m.size == 0:
        return np.array([], dtype=np.float64)
    if ranges_m.size == 1:
        return np.array([0.0])

    # ensure odd window
    window = max(1, window)
    if window % 2 == 0:
        window += 1

    # running-mean smoothing via convolution
    if window > 1 and ranges_m.size >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(ranges_m, kernel, mode="same")
    else:
        smoothed = ranges_m.copy()

    # central difference (numpy.gradient handles edges automatically)
    return np.gradient(smoothed, dt)


# ===================================================================
# 3. local_friend_count_15km
# ===================================================================

def local_friend_count_15km(
    ego_pos: np.ndarray,
    all_friend_positions: np.ndarray,
    radius_m: float = 15_000.0,
) -> int:
    """Count friendly entities within *radius_m* of ego.

    Parameters
    ----------
    ego_pos : shape (2,) or (3,) -- [x, y] or [x, y, z] in metres.
    all_friend_positions : shape (N, 2) or (N, 3), same dim as ego_pos.
        May include ego itself; ego-to-ego distance is 0 and is **not**
        counted (we use strict ``<``).
    radius_m : search radius in metres (default 15 000 m = 15 km).

    Returns
    -------
    int -- number of friends strictly inside *radius_m* (excluding ego).
    """
    ego_pos = np.asarray(ego_pos, dtype=np.float64)
    all_friend_positions = np.asarray(all_friend_positions, dtype=np.float64)

    if all_friend_positions.ndim != 2 or all_friend_positions.shape[0] == 0:
        return 0

    # truncate to common dimensionality
    ndim = min(ego_pos.shape[0], all_friend_positions.shape[1])
    diffs = all_friend_positions[:, :ndim] - ego_pos[:ndim]
    dists = np.linalg.norm(diffs, axis=1)

    # strict < excludes ego (dist == 0) and entities exactly at boundary
    return int(np.sum(dists < radius_m)) - int(np.any(dists == 0.0))


# ===================================================================
# 4. same_target_as_nearest_ally
# ===================================================================

def same_target_as_nearest_ally(
    ego_nearest_enemy_id: object | None,
    ally_nearest_enemy_id: object | None,
) -> bool | None:
    """Check whether ego and its nearest ally are targeting the same enemy.

    Parameters
    ----------
    ego_nearest_enemy_id : hashable identifier (str / int) or None.
    ally_nearest_enemy_id : hashable identifier (str / int) or None.

    Returns
    -------
    True  -- both IDs are known and equal (same target).
    False -- both IDs are known but different.
    None  -- at least one ID is unknown / None.
    """
    if ego_nearest_enemy_id is None or ally_nearest_enemy_id is None:
        return None
    return ego_nearest_enemy_id == ally_nearest_enemy_id


# ===================================================================
# 5. bearing_rate_degps
# ===================================================================

def bearing_rate_degps(bearings_deg: np.ndarray, dt: float) -> np.ndarray:
    """Rate of bearing change in deg/s.

    Parameters
    ----------
    bearings_deg : 1-D array of bearings in degrees ([-180, 180],
                   0 = nose, positive = starboard).
    dt : timestep in seconds.

    Returns
    -------
    1-D array of same length; bearing rate in **deg/s**.

    Notes
    -----
    Wraparound at +/-180 is handled via angular differencing so that a
    transition from 170 to -170 correctly yields -20/dt rather than
    +340/dt.
    """
    bearings_deg = np.asarray(bearings_deg, dtype=np.float64)
    if bearings_deg.size == 0:
        return np.array([], dtype=np.float64)
    if bearings_deg.size == 1:
        return np.array([0.0])

    diffs = _angular_diff_deg(bearings_deg[1:], bearings_deg[:-1])

    rate = np.empty_like(bearings_deg)
    rate[1:-1] = (diffs[1:] + diffs[:-1]) / 2.0 / dt
    rate[0] = diffs[0] / dt
    rate[-1] = diffs[-1] / dt
    return rate


# ===================================================================
# 6. aspect_rate_degps
# ===================================================================

def aspect_rate_degps(aspects_deg: np.ndarray, dt: float) -> np.ndarray:
    """Rate of aspect-angle change in deg/s.

    Parameters
    ----------
    aspects_deg : 1-D array of aspect angles in degrees ([0, 180]).
    dt : timestep in seconds.

    Returns
    -------
    1-D array of same length; aspect rate in **deg/s**.
    Positive = aspect increasing (enemy turning away from head-on
    towards tail), negative = decreasing.

    Notes
    -----
    Aspect is defined in [0, 180] and is **not** a full-circle angle, so
    ordinary (non-angular) differencing is appropriate -- there is no
    wraparound discontinuity.  ``numpy.gradient`` is used for edge
    handling.
    """
    aspects_deg = np.asarray(aspects_deg, dtype=np.float64)
    if aspects_deg.size == 0:
        return np.array([], dtype=np.float64)
    if aspects_deg.size == 1:
        return np.array([0.0])

    return np.gradient(aspects_deg, dt)


# ===================================================================
# 7. local_enemy_count_15km
# ===================================================================

def local_enemy_count_15km(
    ego_pos: np.ndarray,
    all_enemy_positions: np.ndarray,
    radius_m: float = 15_000.0,
) -> int:
    """Count enemy entities within *radius_m* of ego.

    Parameters
    ----------
    ego_pos : shape (2,) or (3,) -- [x, y] or [x, y, z] in metres.
    all_enemy_positions : shape (N, 2) or (N, 3).
    radius_m : search radius in metres (default 15 000 m = 15 km).

    Returns
    -------
    int -- number of enemies strictly inside *radius_m*.
    """
    ego_pos = np.asarray(ego_pos, dtype=np.float64)
    all_enemy_positions = np.asarray(all_enemy_positions, dtype=np.float64)

    if all_enemy_positions.ndim != 2 or all_enemy_positions.shape[0] == 0:
        return 0

    ndim = min(ego_pos.shape[0], all_enemy_positions.shape[1])
    diffs = all_enemy_positions[:, :ndim] - ego_pos[:ndim]
    dists = np.linalg.norm(diffs, axis=1)

    return int(np.sum(dists < radius_m))


# ===================================================================
# 8. nearest_ally_to_target_range_m
# ===================================================================

def nearest_ally_to_target_range_m(
    ally_positions: np.ndarray,
    target_pos: np.ndarray,
) -> float | None:
    """Distance from the nearest friendly to a given target.

    Parameters
    ----------
    ally_positions : shape (N, 2) or (N, 3) -- positions of all allies
                     in metres.
    target_pos : shape (2,) or (3,) -- target position in metres.

    Returns
    -------
    float -- minimum Euclidean distance in **metres**, or
    None  -- if *ally_positions* is empty.
    """
    ally_positions = np.asarray(ally_positions, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    if ally_positions.ndim != 2 or ally_positions.shape[0] == 0:
        return None

    ndim = min(target_pos.shape[0], ally_positions.shape[1])
    diffs = ally_positions[:, :ndim] - target_pos[:ndim]
    dists = np.linalg.norm(diffs, axis=1)

    return float(np.min(dists))
