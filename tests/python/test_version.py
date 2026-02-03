"""Minimal test for scaffold: version and import."""
import pytest


def test_import_xpuruntime():
    import xpuruntime
    assert xpuruntime.__version__ == "0.1.0"


def test_get_version():
    import xpuruntime
    v = xpuruntime.get_version()
    assert isinstance(v, str)
    assert "0.1.0" in v


def test_exception_classes_available():
    """XpuRuntimeError and subclasses are importable and usable in except."""
    import xpuruntime as xrt
    assert issubclass(xrt.XpuRuntimeError, BaseException)
    assert issubclass(xrt.CudaError, xrt.XpuRuntimeError)
    assert issubclass(xrt.OutOfMemoryError, xrt.XpuRuntimeError)
    assert issubclass(xrt.UnsupportedOperationError, xrt.XpuRuntimeError)


def test_exception_can_be_raised_and_caught():
    """Python fallback exceptions can be raised and caught."""
    import xpuruntime as xrt
    try:
        raise xrt.XpuRuntimeError("test message")
    except xrt.XpuRuntimeError as e:
        assert str(e) == "test message"


def _has_native_core():
    try:
        import xpuruntime._core  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_native_core(), reason="Native _core not built")
def test_device_manager_instance():
    """When _core is built, DeviceManager.instance() and get_device_count() work."""
    import xpuruntime as xrt
    dm = xrt.DeviceManager.instance()
    count = dm.get_device_count()
    assert isinstance(count, int)
    assert count >= 0


@pytest.mark.skipif(not _has_native_core(), reason="Native _core not built")
def test_memory_manager_instance():
    """When _core is built, MemoryManager.instance() and get_allocated_size() work."""
    import xpuruntime as xrt
    mm = xrt.MemoryManager.instance()
    size = mm.get_allocated_size()
    assert isinstance(size, int)
    assert size >= 0
