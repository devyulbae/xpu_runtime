"""xpuruntime - Unified GPU/NPU execution runtime."""

__version__ = "0.1.0"


class XpuRuntimeError(RuntimeError):
    """Base runtime error (fallback when _core is not built)."""


class CudaError(XpuRuntimeError):
    """CUDA error (fallback when _core is not built)."""


class OutOfMemoryError(XpuRuntimeError):
    """Out of memory error (fallback when _core is not built)."""


class UnsupportedOperationError(XpuRuntimeError):
    """Unsupported operation (fallback when _core is not built)."""


try:
    from xpuruntime._core import (
        get_version as _get_native_version,
        XpuRuntimeError as _XpuRuntimeError,
        CudaError as _CudaError,
        OutOfMemoryError as _OutOfMemoryError,
        UnsupportedOperationError as _UnsupportedOperationError,
        DeviceInfo,
        DeviceManager,
        MemoryType,
        MemoryStats,
        MemoryManager,
    )
    XpuRuntimeError = _XpuRuntimeError
    CudaError = _CudaError
    OutOfMemoryError = _OutOfMemoryError
    UnsupportedOperationError = _UnsupportedOperationError
except ImportError:
    _get_native_version = None
    DeviceInfo = None  # type: ignore[misc, assignment]
    DeviceManager = None  # type: ignore[misc, assignment]
    MemoryType = None  # type: ignore[misc, assignment]
    MemoryStats = None  # type: ignore[misc, assignment]
    MemoryManager = None  # type: ignore[misc, assignment]


def get_version() -> str:
    """Return package version (from native if available)."""
    if _get_native_version is not None:
        return _get_native_version()
    return __version__


# Convenience re-exports for device API
from xpuruntime.runtime.device import (
    Device,
    DeviceCountsType,
    format_device_counts,
    get_current_device,
    get_device_count,
    get_device_counts,
    set_current_device,
)

__all__ = [
    "__version__",
    "get_version",
    "XpuRuntimeError",
    "CudaError",
    "OutOfMemoryError",
    "UnsupportedOperationError",
    "DeviceInfo",
    "DeviceManager",
    "MemoryType",
    "MemoryStats",
    "MemoryManager",
    "Device",
    "DeviceCountsType",
    "format_device_counts",
    "get_device_count",
    "get_device_counts",
    "get_current_device",
    "set_current_device",
]
