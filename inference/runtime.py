"""Policy runtime: artifact loading, strict schema checks, preprocessing,
backend dispatch.

Default behaviour is **strict**: the runtime refuses to load any artifact
that is missing required metadata fields or has a schema_version other than
the current one.  Pass ``allow_legacy=True`` to ``from_artifact_dir()`` (or
``--allow_legacy_artifact`` on the CLI) to enter a compatibility path that
accepts older artifacts with warnings.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from inference.backends import NumpyLinearBackend, OnnxRuntimeBackend
from shared.schema import SCHEMA_VERSION


HEADING_FIELD_ALIASES: Dict[str, str] = {
    "heading_rad": "track_angle_rad_unwrapped",
    "heading_rad_unwrapped": "track_angle_rad_unwrapped",
}

_LEGACY_FIELD_ORDER: List[str] = [
    "x_e_m", "y_n_m", "z_u_m",
    "vx_e_mps", "vy_n_mps", "vz_u_mps",
    "heading_rad", "ground_speed_mps",
]

REQUIRED_ACTION_FIELD_ORDER: List[str] = [
    "dpsi_rad", "alt_sp_m", "spd_sp_mps",
]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
    except ModuleNotFoundError:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid policy config payload in {path}")
    return cfg


def _legacy_or_raise(
    field_name: str,
    artifact_path: str,
    *,
    allow_legacy: bool,
    default: Any = None,
) -> Any:
    """In strict mode raise; in legacy mode warn and return *default*."""
    if not allow_legacy:
        raise ValueError(
            f"Artifact at {artifact_path} missing required field: "
            f"{field_name}.  Re-export artifact or pass allow_legacy=True."
        )
    warnings.warn(
        f"[inference] LEGACY COMPAT: artifact at {artifact_path} missing "
        f"{field_name}.  Using default={default!r}.  "
        f"This compat path is NOT suitable for production / formal evaluation.",
        stacklevel=3,
    )
    return default


def _check_schema_compat(
    artifact_version: str,
    current_version: str,
    artifact_path: str,
    *,
    allow_legacy: bool,
) -> None:
    if artifact_version == current_version:
        return
    if not allow_legacy:
        raise ValueError(
            f"Artifact schema_version={artifact_version!r} does not match "
            f"runtime version {current_version!r}.  Re-export artifact or "
            f"pass allow_legacy=True."
        )
    try:
        a = [int(x) for x in artifact_version.split(".")]
        c = [int(x) for x in current_version.split(".")]
    except (ValueError, AttributeError):
        raise ValueError(
            f"Unparseable schema versions: artifact={artifact_version!r}, "
            f"current={current_version!r}"
        )
    if a[0] != c[0]:
        raise ValueError(
            f"Artifact schema major version mismatch: "
            f"artifact={artifact_version!r}, runtime={current_version!r}.  "
            f"Even legacy mode cannot bridge major version differences."
        )
    if tuple(a) > tuple(c):
        raise ValueError(
            f"Artifact schema version is newer than runtime: "
            f"artifact={artifact_version!r}, runtime={current_version!r}.  "
            f"Upgrade the runtime."
        )
    warnings.warn(
        f"[inference] LEGACY COMPAT: artifact at {artifact_path} has "
        f"schema_version={artifact_version!r}, runtime expects "
        f"{current_version!r}.  This compat path is NOT suitable for "
        f"production or formal evaluation.  Re-export artifact to update.",
        stacklevel=3,
    )


def _normalize_field_order(
    field_order: List[str], source: str, *, allow_legacy: bool,
) -> List[str]:
    result: List[str] = []
    for f in field_order:
        canonical = HEADING_FIELD_ALIASES.get(f)
        if canonical is not None:
            if not allow_legacy:
                raise ValueError(
                    f"Artifact ({source}) uses deprecated field name '{f}' "
                    f"in obs_spec.field_order.  Use '{canonical}' instead, "
                    f"or pass allow_legacy=True."
                )
            warnings.warn(
                f"[inference] LEGACY COMPAT: artifact ({source}) uses '{f}' "
                f"in obs_spec.field_order.  Mapping to '{canonical}'.  "
                f"This compat path is NOT suitable for production.",
                stacklevel=3,
            )
            result.append(canonical)
        else:
            result.append(f)
    return result


# ------------------------------------------------------------------
# PolicyRuntime
# ------------------------------------------------------------------

class PolicyRuntime:
    """Inference runtime used by sim-side adapters.

    Exposes structured metadata so that the sim layer can validate its own
    observation / action construction against the artifact contract.

    ``dpsi_rad`` in ``action_spec.field_order`` is defined by the producer as a
    lookahead delta over ``horizon_steps / control_frequency_hz`` seconds.  The
    sim layer must use both metadata fields when mapping that output to
    per-tick actuator commands.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        obs_dim: int = 8,
        backend_name: str,
        predict_fn,
        policy_meta: Dict[str, Any] | None = None,
        field_order: List[str] | None = None,
        action_field_order: List[str] | None = None,
        control_frequency_hz: float | None = None,
        horizon_steps: int | None = None,
    ) -> None:
        self.seq_len = int(seq_len)
        self.obs_dim = int(obs_dim)
        self.backend_name = backend_name
        self._predict_fn = predict_fn
        self.policy_meta = policy_meta or {}
        self._field_order = field_order
        self._action_field_order = action_field_order
        self._control_frequency_hz = control_frequency_hz
        self._horizon_steps = horizon_steps

    @property
    def field_order(self) -> List[str] | None:
        return list(self._field_order) if self._field_order else None

    @property
    def action_field_order(self) -> List[str] | None:
        return list(self._action_field_order) if self._action_field_order else None

    @property
    def control_frequency_hz(self) -> float | None:
        return self._control_frequency_hz

    @property
    def horizon_steps(self) -> int | None:
        return self._horizon_steps

    @property
    def lookahead_horizon_s(self) -> float | None:
        if self._horizon_steps is None or self._control_frequency_hz is None:
            return None
        return float(self._horizon_steps) / float(self._control_frequency_hz)

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        """Run inference.  ``obs_seq`` must have **exactly** ``seq_len`` rows.

        No silent truncation or zero-padding is performed.
        """
        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        if obs_seq.ndim != 2:
            raise ValueError(
                f"obs_seq must be rank-2 [T,D], got shape={obs_seq.shape}"
            )
        if obs_seq.shape[0] != self.seq_len:
            raise ValueError(
                f"obs_seq window length={obs_seq.shape[0]} does not match "
                f"artifact seq_len={self.seq_len}.  "
                f"No silent truncation or padding is allowed."
            )
        pred = np.asarray(
            self._predict_fn(obs_seq), dtype=np.float32,
        ).reshape(-1)
        if pred.shape[0] != 3:
            raise ValueError(
                f"policy backend must output 3 dims, got {pred.shape}"
            )
        return pred

    # ------------------------------------------------------------------
    # Factory: artifact directory (preferred path)
    # ------------------------------------------------------------------
    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        allow_legacy: bool = False,
    ) -> "PolicyRuntime":
        artifact = Path(artifact_dir)
        policy_path = artifact / "policy.yaml"
        if not policy_path.exists():
            raise FileNotFoundError(
                f"policy.yaml not found in artifact dir: {artifact}"
            )

        policy = _load_yaml_or_json(policy_path)

        # --- schema_version ---
        policy_schema_version = str(policy.get("schema_version", "")).strip()
        if not policy_schema_version:
            raise ValueError(f"Missing schema_version in {policy_path}")
        _check_schema_compat(
            policy_schema_version, SCHEMA_VERSION, str(artifact),
            allow_legacy=allow_legacy,
        )

        backend_name = str(policy.get("backend", "")).strip()
        if not backend_name:
            raise ValueError(f"Missing 'backend' in {policy_path}")

        # --- obs_spec ---
        obs_spec = policy.get("obs_spec", {}) or {}

        raw_seq_len = obs_spec.get("seq_len")
        if raw_seq_len is not None:
            seq_len = int(raw_seq_len)
        else:
            seq_len = int(_legacy_or_raise(
                "obs_spec.seq_len", str(artifact),
                allow_legacy=allow_legacy, default=20,
            ))

        obs_dim = int(obs_spec.get("dim", 8))

        raw_field_order = obs_spec.get("field_order")
        if raw_field_order is not None and isinstance(raw_field_order, list):
            field_order = _normalize_field_order(
                raw_field_order, str(artifact), allow_legacy=allow_legacy,
            )
        else:
            legacy_default = _legacy_or_raise(
                "obs_spec.field_order", str(artifact),
                allow_legacy=allow_legacy,
                default=list(_LEGACY_FIELD_ORDER),
            )
            field_order = _normalize_field_order(
                list(legacy_default), str(artifact), allow_legacy=True,
            )

        if len(field_order) != obs_dim:
            raise ValueError(
                f"obs_spec.field_order has {len(field_order)} fields "
                f"but dim={obs_dim}"
            )

        # --- action_spec ---
        act_spec = policy.get("action_spec", {}) or {}
        act_dim = int(act_spec.get("dim", 3))
        if act_dim != 3:
            raise ValueError(f"action_spec.dim must be 3, got {act_dim}")

        raw_act_fo = act_spec.get("field_order")
        if raw_act_fo is not None and isinstance(raw_act_fo, list):
            if raw_act_fo != REQUIRED_ACTION_FIELD_ORDER:
                raise ValueError(
                    f"action_spec.field_order must be exactly "
                    f"{REQUIRED_ACTION_FIELD_ORDER}, got {raw_act_fo}"
                )
            action_field_order: list[str] = list(raw_act_fo)
        else:
            _legacy_or_raise(
                "action_spec.field_order", str(artifact),
                allow_legacy=allow_legacy,
                default=REQUIRED_ACTION_FIELD_ORDER,
            )
            action_field_order = list(REQUIRED_ACTION_FIELD_ORDER)

        raw_freq = act_spec.get("control_frequency_hz")
        if raw_freq is not None:
            control_frequency_hz: float | None = float(raw_freq)
            if control_frequency_hz <= 0:
                raise ValueError(
                    f"control_frequency_hz must be positive, "
                    f"got {control_frequency_hz}"
                )
        else:
            _legacy_or_raise(
                "action_spec.control_frequency_hz", str(artifact),
                allow_legacy=allow_legacy, default=None,
            )
            control_frequency_hz = None

        raw_horizon = act_spec.get("horizon_steps")
        if raw_horizon is not None:
            horizon_steps: int | None = int(raw_horizon)
            if horizon_steps < 1:
                raise ValueError(
                    f"horizon_steps must be >= 1, got {horizon_steps}"
                )
        else:
            _legacy_or_raise(
                "action_spec.horizon_steps", str(artifact),
                allow_legacy=allow_legacy, default=None,
            )
            horizon_steps = None

        # --- preprocess ---
        preprocess = policy.get("preprocess", {}) or {}
        normalize_obs = bool(preprocess.get("normalize_obs", False))
        denormalize_action = bool(preprocess.get("denormalize_action", False))
        norm_stats_file = str(
            preprocess.get("norm_stats_file", "norm_stats.json")
        )
        stats_path = artifact / norm_stats_file

        obs_mean = obs_std = act_mean = act_std = None
        if normalize_obs or denormalize_action:
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"norm stats file not found: {stats_path}"
                )
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)
            obs_std = np.asarray(stats["obs_std"], dtype=np.float32)
            act_mean = np.asarray(stats["act_mean"], dtype=np.float32)
            act_std = np.asarray(stats["act_std"], dtype=np.float32)

        # --- backend ---
        model_file = str(policy.get("model_file", "")).strip()
        model_path = artifact / model_file if model_file else None

        if backend_name == "onnxruntime":
            if model_path is None:
                raise ValueError(
                    "policy.backend=onnxruntime requires model_file"
                )
            backend = OnnxRuntimeBackend(
                model_path=model_path,
                input_tensor_names=policy.get("input_tensor_names", []),
                output_tensor_names=policy.get("output_tensor_names", []),
            )
        elif backend_name == "numpy_linear":
            backend = NumpyLinearBackend(model_path=model_path)
        else:
            raise ValueError(f"Unsupported policy backend: {backend_name}")

        def _predict(obs_seq: np.ndarray) -> np.ndarray:
            if obs_seq.shape[1] != obs_dim:
                raise ValueError(
                    f"obs dim mismatch: expected {obs_dim}, "
                    f"got {obs_seq.shape[1]}"
                )
            arr = obs_seq
            if normalize_obs:
                assert obs_mean is not None and obs_std is not None
                arr = (arr - obs_mean.reshape(1, -1)) / np.maximum(
                    obs_std.reshape(1, -1), 1e-6,
                )
            pred = backend.predict(arr)
            if denormalize_action:
                assert act_mean is not None and act_std is not None
                pred = pred * act_std + act_mean
            return pred

        return cls(
            seq_len=seq_len,
            obs_dim=obs_dim,
            backend_name=backend_name,
            predict_fn=_predict,
            policy_meta=policy,
            field_order=field_order,
            action_field_order=action_field_order,
            control_frequency_hz=control_frequency_hz,
            horizon_steps=horizon_steps,
        )

    # ------------------------------------------------------------------
    # Factory: legacy .pt checkpoint (deprecated, requires opt-in)
    # ------------------------------------------------------------------
    @classmethod
    def from_legacy_checkpoint(
        cls,
        model_path: str | Path,
        seq_len: int = 20,
        *,
        allow_legacy: bool = False,
    ) -> "PolicyRuntime":
        if not allow_legacy:
            raise ValueError(
                "Legacy .pt checkpoint loading requires explicit opt-in: "
                "pass allow_legacy=True or use --allow_legacy_artifact.  "
                "Migrate to artifact-based deployment for full contract "
                "validation."
            )

        from inference.backends.torch_ckpt_backend import TorchCheckpointBackend

        warnings.warn(
            "[inference] LEGACY COMPAT: loading legacy .pt checkpoint "
            "without policy.yaml metadata.  Field order, "
            "control_frequency_hz, and horizon_steps are unavailable.  "
            "This compat path is NOT suitable for production or formal "
            "evaluation.",
            stacklevel=2,
        )

        backend = TorchCheckpointBackend(model_path)
        legacy_field_order = _normalize_field_order(
            list(_LEGACY_FIELD_ORDER), "legacy_checkpoint",
            allow_legacy=True,
        )

        def _predict(obs_seq: np.ndarray) -> np.ndarray:
            return backend.predict(obs_seq)

        return cls(
            seq_len=int(seq_len),
            obs_dim=8,
            backend_name="legacy_torch_checkpoint",
            predict_fn=_predict,
            policy_meta={"legacy_model_path": str(model_path)},
            field_order=legacy_field_order,
            action_field_order=list(REQUIRED_ACTION_FIELD_ORDER),
            control_frequency_hz=None,
            horizon_steps=None,
        )
