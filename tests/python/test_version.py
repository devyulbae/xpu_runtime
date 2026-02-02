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
