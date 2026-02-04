"""xpuruntime.inference - Inference session and model execution."""

from xpuruntime.inference.openvino_backend import (
    OpenVINOSession,
    create_openvino_session,
)

__all__ = [
    "OpenVINOSession",
    "create_openvino_session",
]
