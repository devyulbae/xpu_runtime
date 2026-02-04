"""OpenVINO backend: simple session API. Users do not configure OpenVINO directly.

Device mapping: "auto" -> Intel GPU if available else CPU; "intel_gpu" -> GPU;
"cpu" -> CPU; "npu" -> NPU (OpenVINO NPU support may be limited; we fall back to CPU if unavailable).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _resolve_device(device: str) -> str:
    """Map user-friendly device to OpenVINO device_name. 'auto' -> GPU if Intel GPU else CPU."""
    if device in ("cpu", "CPU"):
        return "CPU"
    if device in ("intel_gpu", "gpu", "GPU"):
        return "GPU"
    if device in ("npu", "NPU"):
        return "NPU"
    if device == "auto":
        try:
            from xpuruntime import get_device_counts
            counts = get_device_counts()
            if counts.get("gpu", {}).get("intel", 0) > 0:
                return "GPU"
        except Exception:
            pass
        return "CPU"
    return device


def create_openvino_session(
    model_path: str,
    device: str = "auto",
    **kwargs: Any,
) -> "OpenVINOSession":
    """Create an inference session on OpenVINO. No manual Core/compile_model needed.

    Args:
        model_path: Path to ONNX or OpenVINO IR (.xml) model.
        device: One of "auto", "intel_gpu", "cpu", "npu". Default "auto" uses Intel GPU if available.
        **kwargs: Passed to openvino compile_model config (optional).

    Returns:
        OpenVINOSession with .run(inputs) and .input_names / .output_names.

    Raises:
        ImportError: If openvino is not installed (pip install xpuruntime[openvino]).
    """
    try:
        import openvino as ov
    except ImportError as e:
        raise ImportError(
            "OpenVINO is required. Install with: pip install xpuruntime[openvino]"
        ) from e

    ov_device = _resolve_device(device)
    core = ov.Core()
    available = core.available_devices
    if ov_device not in available and ov_device == "NPU":
        ov_device = "CPU"
    elif ov_device not in available:
        ov_device = "CPU"

    model = core.read_model(model_path)
    compiled = core.compile_model(model, ov_device, config=kwargs or {})

    return OpenVINOSession(compiled, device=ov_device)


class OpenVINOSession:
    """Thin wrapper over OpenVINO compiled model. Use create_openvino_session()."""

    def __init__(self, compiled_model: Any, device: str = "CPU") -> None:
        self._compiled = compiled_model
        self.device = device

    @property
    def input_names(self) -> list[str]:
        return [inp.get_any_name() for inp in self._compiled.inputs]

    @property
    def output_names(self) -> list[str]:
        return [out.get_any_name() for out in self._compiled.outputs]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference. inputs: name -> numpy array. Returns name -> numpy array."""
        infer_request = self._compiled.create_infer_request()
        infer_request.infer(inputs)
        out: dict[str, np.ndarray] = {}
        for output in self._compiled.outputs:
            name = output.get_any_name()
            tensor = infer_request.get_tensor(output)
            data = getattr(tensor, "data", tensor)
            out[name] = np.array(data) if not isinstance(data, np.ndarray) else data
        return out
