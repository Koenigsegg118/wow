#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entity filter: regex-based name/type filtering + orbiter behaviour detection.

Used by both acmi_to_dt_dataset_smooth.py and analyze_acmi_coverage.py.

Orbiter heuristic
-----------------
An entity is classified as a non-combat orbiter (AWACS, tanker, long-orbit
patrol, etc.) if ALL three criteria hold:
  - n_valid_steps >= orb_min_steps  (long enough to be reliable)
  - std(altitude)  < orb_alt_std_m  (flies at steady altitude)
  - std(speed)     < orb_spd_std_mps (flies at steady speed)
  - p95(|Δψ|)      < orb_dpsi_p95_rad (rarely manoeuvres)
"""

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Default exclude pattern – common non-fighter types present in ACMI data
# ---------------------------------------------------------------------------
DEFAULT_EXCLUDE_REGEX: str = (
    r"(?i)(A-50|E-3|E-2|E-767|AWACS|Sentry|Hawkeye|"
    r"KC-135|KC-10|KC-46|IL-78|Il-78|Tanker|"
    r"C-130|C-17|C-5|An-26|An-12|Il-76|"
    r"Tu-95|B-52|B-1|B-2|Bomber|Transport|JSTAR)"
)


class EntityFilter:
    """Combined regex + orbiter filter for ACMI entities.

    Parameters
    ----------
    include_regex : str or None
        If given, only entities whose ``Name|Type`` string matches this
        pattern are kept.
    exclude_regex : str or None
        Entities whose ``Name|Type`` string matches this pattern are dropped.
        Defaults to ``DEFAULT_EXCLUDE_REGEX`` when *exclude_orbiters* is True
        and no value is supplied; pass ``""`` to disable.
    exclude_orbiters : bool
        Whether to apply the orbiter behaviour filter.
    orb_alt_std_m : float
        Maximum altitude std-dev (m) for an entity to be called an orbiter.
    orb_spd_std_mps : float
        Maximum speed std-dev (m/s) for an entity to be called an orbiter.
    orb_dpsi_p95_rad : float
        Maximum p95(|Δψ|) (rad) for an entity to be called an orbiter.
    orb_min_steps : int
        Minimum number of valid steps required before the orbiter heuristic
        is applied (too-short tracks are not flagged).
    """

    def __init__(
        self,
        include_regex: Optional[str] = None,
        exclude_regex: Optional[str] = None,
        exclude_orbiters: bool = True,
        orb_alt_std_m: float = 80.0,
        orb_spd_std_mps: float = 7.0,
        orb_dpsi_p95_rad: float = 0.05,
        orb_min_steps: int = 600,
    ):
        self.include_regex = include_regex
        # When exclude_regex is None (not supplied) and orbiters are being
        # excluded, apply the default name/type pattern as well.
        if exclude_regex is None and exclude_orbiters:
            self.exclude_regex: Optional[str] = DEFAULT_EXCLUDE_REGEX
        else:
            self.exclude_regex = exclude_regex if exclude_regex else None

        self.exclude_orbiters = exclude_orbiters
        self.orb_alt_std_m = orb_alt_std_m
        self.orb_spd_std_mps = orb_spd_std_mps
        self.orb_dpsi_p95_rad = orb_dpsi_p95_rad
        self.orb_min_steps = orb_min_steps

        self._include_rx = re.compile(include_regex) if include_regex else None
        self._exclude_rx = re.compile(self.exclude_regex) if self.exclude_regex else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True if any filtering is enabled."""
        return bool(self._include_rx or self._exclude_rx or self.exclude_orbiters)

    def check_name_type(self, name: str, typ: str) -> Tuple[bool, str]:
        """Check entity name/type string against include/exclude regexes.

        Returns
        -------
        (keep, reason) : (bool, str)
            *keep* is True if the entity passes the name/type filter.
            *reason* is ``"ok"`` on pass, or the filter that rejected it.
        """
        target = f"{name}|{typ}"
        if self._exclude_rx and self._exclude_rx.search(target):
            return False, "regex_exclude"
        if self._include_rx and not self._include_rx.search(target):
            return False, "regex_include_miss"
        return True, "ok"

    def is_orbiter_from_stats(self, entity_stats: dict) -> Tuple[bool, str]:
        """Check whether an entity behaves like an orbiter based on its stats.

        *entity_stats* is the dict returned by ``analyse_entity()`` in
        ``analyze_acmi_coverage.py``, or the equivalent dict constructed in
        ``acmi_to_dt_dataset_smooth.py``.  Required keys:

          - ``"n_after_filter"`` (int) – number of valid samples used
          - ``"alt"``      → dict with ``"std"`` key
          - ``"speed"``    → dict with ``"std"`` key
          - ``"abs_dpsi"`` → dict with ``"p95"`` key

        Returns
        -------
        (is_orbiter, reason) : (bool, str)
        """
        if not self.exclude_orbiters:
            return False, "ok"

        n_valid = entity_stats.get("n_after_filter", 0)
        if n_valid < self.orb_min_steps:
            return False, "ok"  # too short to judge

        alt_std  = entity_stats.get("alt",      {}).get("std",  9999.0)
        spd_std  = entity_stats.get("speed",    {}).get("std",  9999.0)
        dpsi_p95 = entity_stats.get("abs_dpsi", {}).get("p95",  9999.0)

        is_orb = (
            alt_std  < self.orb_alt_std_m
            and spd_std  < self.orb_spd_std_mps
            and dpsi_p95 < self.orb_dpsi_p95_rad
        )
        return is_orb, ("orbiter" if is_orb else "ok")

    def to_dict(self) -> dict:
        """Serialisable representation for dataset metadata."""
        return dict(
            include_regex=self.include_regex,
            exclude_regex=self.exclude_regex,
            exclude_orbiters=self.exclude_orbiters,
            orb_alt_std_m=self.orb_alt_std_m,
            orb_spd_std_mps=self.orb_spd_std_mps,
            orb_dpsi_p95_rad=self.orb_dpsi_p95_rad,
            orb_min_steps=self.orb_min_steps,
        )
