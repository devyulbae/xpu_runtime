"""xpuruntime - Unified GPU/NPU execution runtime."""

__version__ = "0.1.0"

try:
    from xpuruntime._core import get_version as _get_native_version
except ImportError:
    _get_native_version = None


def get_version() -> str:
    """Return package version (from native if available)."""
    if _get_native_version is not None:
        return _get_native_version()
    return __version__
