# ACMI Mining V2 -- Feature Contract

Module: `tools/acmi_mining/event_features_v2.py`

## Convention References

All sign/unit conventions follow the canonical geometry module
`sim/runtime/afsim_obs_geometry.py` (frozen at v1.1):

| Convention | Definition |
|---|---|
| heading | 0 = North, CW positive, degrees |
| bearing | 0 = nose, positive = starboard, [-180, 180] degrees |
| aspect | 0 = head-on, 180 = tail-on, [0, 180] degrees |
| closure / range-rate | negative = closing, positive = opening, m/s |
| range | metres, always >= 0 |
| positions | Cartesian [east_m, north_m, alt_m] |

---

## Feature Catalogue

### 1. `heading_rate_degps`

| Field | Value |
|---|---|
| **Name** | heading_rate_degps |
| **Type** | float (per-frame array) |
| **Unit** | deg/s |
| **Formula** | Central difference of 3-point-median-smoothed headings with angular wraparound handling. Forward/backward difference at edges. |
| **Source convention** | heading: 0=N, CW+ (afsim_obs_geometry) |
| **Buckets served** | D (turn_defense) -- detects sustained heading changes; A/B for stability filtering |

### 2. `range_rate_smooth_mps`

| Field | Value |
|---|---|
| **Name** | range_rate_smooth_mps |
| **Type** | float (per-frame array) |
| **Unit** | m/s |
| **Formula** | `numpy.gradient` of running-mean-smoothed range series. Smoothing window defaults to 3 (odd, >= 1). |
| **Source convention** | closure: negative = closing, positive = opening (afsim_obs_geometry) |
| **Buckets served** | A (far_intercept) -- stable closure estimate; C (extend_disengage) -- confirms opening trend |

### 3. `local_friend_count_15km`

| Field | Value |
|---|---|
| **Name** | local_friend_count_15km |
| **Type** | int (scalar per frame) |
| **Unit** | count |
| **Formula** | Euclidean distance from ego to each friendly position; count where dist < radius_m (default 15 km). Ego itself excluded via strict `<` on zero distance. |
| **Source convention** | positions in metres, Cartesian (2-D or 3-D) |
| **Buckets served** | E (handoff_coordination) -- requires ally presence; B (standoff_hold) -- formation context |

### 4. `same_target_as_nearest_ally`

| Field | Value |
|---|---|
| **Name** | same_target_as_nearest_ally |
| **Type** | bool or None (scalar per frame) |
| **Unit** | -- (boolean flag) |
| **Formula** | `True` if both ego and ally nearest-enemy IDs are known and equal; `False` if known and different; `None` if either is unknown. |
| **Source convention** | entity IDs (hashable: str or int) |
| **Buckets served** | E (handoff_coordination) -- detects target sharing / target switch between allies |

### 5. `bearing_rate_degps`

| Field | Value |
|---|---|
| **Name** | bearing_rate_degps |
| **Type** | float (per-frame array) |
| **Unit** | deg/s |
| **Formula** | Central difference of bearing series with angular wraparound at +/-180. Forward/backward difference at edges. |
| **Source convention** | bearing: 0=nose, +right, [-180,180] (afsim_obs_geometry) |
| **Buckets served** | D (turn_defense) -- rate of nose-to-threat change; A (far_intercept) -- intercept geometry stability |

### 6. `aspect_rate_degps`

| Field | Value |
|---|---|
| **Name** | aspect_rate_degps |
| **Type** | float (per-frame array) |
| **Unit** | deg/s |
| **Formula** | `numpy.gradient` of aspect series (no wraparound -- aspect is [0, 180]). |
| **Source convention** | aspect: 0=head-on, 180=tail-on (afsim_obs_geometry) |
| **Buckets served** | D (turn_defense) -- detects offensive/defensive transition; B (standoff_hold) -- aspect drift |

### 7. `local_enemy_count_15km`

| Field | Value |
|---|---|
| **Name** | local_enemy_count_15km |
| **Type** | int (scalar per frame) |
| **Unit** | count |
| **Formula** | Euclidean distance from ego to each enemy position; count where dist < radius_m (default 15 km). |
| **Source convention** | positions in metres, Cartesian (2-D or 3-D) |
| **Buckets served** | E (handoff_coordination) -- multi-threat awareness; C (extend_disengage) -- threat density for disengage decision |

### 8. `nearest_ally_to_target_range_m`

| Field | Value |
|---|---|
| **Name** | nearest_ally_to_target_range_m |
| **Type** | float or None (scalar per frame) |
| **Unit** | metres |
| **Formula** | Minimum Euclidean distance from any ally position to a given target position. Returns None if no allies provided. |
| **Source convention** | positions in metres, Cartesian (2-D or 3-D) |
| **Buckets served** | E (handoff_coordination) -- nearest-ally range drives handoff decision; A (far_intercept) -- formation range context |

---

## Edge-Case Guarantees

All functions handle the following edge cases without raising exceptions:

| Condition | Behaviour |
|---|---|
| Empty input array (length 0) | Returns empty array or 0 / None as appropriate |
| Single-frame input (length 1) | Returns `[0.0]` for rate functions, valid scalar for counts |
| NaN in input | Propagated through numpy operations (NaN in, NaN out) |
| Heading wraparound (e.g. 179 -> -179) | Correct shortest-arc difference via `(a - b + 180) % 360 - 180` |
| Ego included in friend list | Excluded by strict `<` on zero distance |

## Dependency Policy

This module imports **only** `numpy`. It has no dependency on:
- VQ-VAE / token bridge / training pipeline
- LangGraph / LLM clients
- ACMI parsers or dataset builders
