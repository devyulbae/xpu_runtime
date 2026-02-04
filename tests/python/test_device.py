"""Tests for device API: get_device_count, get_device_counts, Device, set_current_device."""

import pytest


def test_get_device_counts_structure():
    import xpuruntime as xrt
    counts = xrt.get_device_counts()
    assert "cpu" in counts and "gpu" in counts and "npu" in counts
    assert counts["cpu"].keys() >= {"x86"}
    assert counts["gpu"].keys() >= {"nvidia", "intel", "amd"}
    assert counts["npu"].keys() >= {"intel", "qualcomm"}
    for cat in ("cpu", "gpu", "npu"):
        for v in counts[cat].values():
            assert isinstance(v, int) and v >= 0


def test_format_device_counts():
    import xpuruntime as xrt
    counts = xrt.get_device_counts()
    text = xrt.format_device_counts(counts)
    lines = text.strip().split("\n")
    assert len(lines) == 3
    assert "cpu :" in lines[0] and "gpu :" in lines[1] and "npu :" in lines[2]


def test_get_device_count():
    import xpuruntime as xrt
    count = xrt.get_device_count()
    assert isinstance(count, int)
    assert count >= 0


def test_device_creation_and_properties():
    import xpuruntime as xrt
    dev = xrt.Device(0)
    assert dev.device_id == 0
    assert isinstance(dev.name, str)
    assert isinstance(dev.total_memory, int)
    assert isinstance(dev.free_memory, int)
    maj, min_ = dev.compute_capability
    assert isinstance(maj, int) and isinstance(min_, int)
    assert "Device" in repr(dev)


def _has_native():
    try:
        import xpuruntime._core  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_native(), reason="Native _core not built")
def test_set_current_device():
    import xpuruntime as xrt
    count = xrt.get_device_count()
    if count == 0:
        pytest.skip("No GPU")
    xrt.set_current_device(0)
    current = xrt.get_current_device()
    assert current == 0


@pytest.mark.skipif(not _has_native(), reason="Native _core not built")
def test_device_info_from_native():
    import xpuruntime as xrt
    if xrt.get_device_count() == 0:
        pytest.skip("No GPU")
    dev = xrt.Device(0)
    assert dev.total_memory >= 0
    assert dev.name  # non-empty when native returns real props
