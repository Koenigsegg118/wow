"""ONNX Runtime inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class OnnxRuntimeBackend:
    def __init__(
        self,
        model_path: str | Path,
        input_tensor_names: Sequence[str] | None = None,
        output_tensor_names: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "onnxruntime is required for backend=onnxruntime. "
                "Install it or switch policy backend to numpy_linear."
            ) from exc

        self._ort = ort
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = (
            input_tensor_names[0] if input_tensor_names else self.session.get_inputs()[0].name
        )
        self.output_name = (
            output_tensor_names[0] if output_tensor_names else self.session.get_outputs()[0].name
        )

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        if obs_seq.ndim != 2:
            raise ValueError(f"obs_seq must be rank-2 [T,D], got shape={obs_seq.shape}")
        inp = obs_seq[np.newaxis, :, :]
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        out_arr = np.asarray(out, dtype=np.float32)
        if out_arr.ndim == 3:
            return out_arr[0, -1, :]
        if out_arr.ndim == 2:
            return out_arr[-1, :]
        if out_arr.ndim == 1:
            return out_arr
        raise ValueError(f"Unexpected ONNX output shape: {out_arr.shape}")
