"""Device management: get_device_count, set_current_device, Device wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Type for get_device_counts(): category -> (vendor/backend -> count)
DeviceCountsType = dict[str, dict[str, int]]


def _get_dm():
    try:
        from xpuruntime._core import DeviceManager
        return DeviceManager.instance()
    except ImportError:
        return None


def _count_intel_gpus_opencl() -> int:
    """Count Intel GPU devices via OpenCL. Returns 0 if pyopencl not available or on error."""
    try:
        import pyopencl as cl
    except ImportError:
        return 0
    count = 0
    try:
        for platform in cl.get_platforms():
            name = (platform.get_info(cl.platform_info.NAME) or "").lower()
            vendor = (platform.get_info(cl.platform_info.VENDOR) or "").lower()
            if "intel" not in name and "intel" not in vendor:
                continue
            for _ in platform.get_devices(device_type=cl.device_type.GPU):
                count += 1
    except Exception:
        pass
    return count


def get_device_counts() -> DeviceCountsType:
    """Return device counts by category and vendor. Format: { cpu: {x86: n}, gpu: {nvidia, intel, amd}, npu: {intel, qualcomm} }."""
    cpu: dict[str, int] = {"x86": 1}
    gpu: dict[str, int] = {"nvidia": 0, "intel": 0, "amd": 0}
    npu: dict[str, int] = {"intel": 0, "qualcomm": 0}

    dm = _get_dm()
    if dm is not None:
        gpu["nvidia"] = dm.get_device_count()

    gpu["intel"] = _count_intel_gpus_opencl()

    return {"cpu": cpu, "gpu": gpu, "npu": npu}


def format_device_counts(counts: DeviceCountsType | None = None) -> str:
    """Format device counts as three lines: cpu / gpu / npu with vendor breakdown."""
    if counts is None:
        counts = get_device_counts()
    lines = []
    for category in ("cpu", "gpu", "npu"):
        parts = counts.get(category, {})
        line = " ".join(f"{k} {v}" for k, v in sorted(parts.items()))
        lines.append(f"{category} : {line}")
    return "\n".join(lines)


def get_device_count() -> int:
    """Return the number of available CUDA GPU devices. 0 if native runtime is not built."""
    dm = _get_dm()
    if dm is None:
        return 0
    return dm.get_device_count()


def set_current_device(device_id: int) -> None:
    """Set the current CUDA device. Raises if native runtime is not built or device_id is invalid."""
    dm = _get_dm()
    if dm is None:
        raise RuntimeError("Native runtime not available; cannot set device.")
    dm.set_current_device(device_id)


def get_current_device() -> int:
    """Return the current device id. 0 if native runtime is not built."""
    dm = _get_dm()
    if dm is None:
        return 0
    return dm.get_current_device()


class Device:
    """Wrapper for a GPU device by id. Properties come from DeviceInfo when native runtime is available."""

    __slots__ = ("_device_id", "_info")

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id
        self._info: Any = None
        dm = _get_dm()
        if dm is not None and 0 <= device_id < dm.get_device_count():
            self._info = dm.get_device_info(device_id)

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def name(self) -> str:
        if self._info is not None:
            return self._info.name
        return f"GPU {self._device_id}"

    @property
    def total_memory(self) -> int:
        if self._info is not None:
            return self._info.total_memory
        return 0

    @property
    def free_memory(self) -> int:
        if self._info is not None:
            return self._info.free_memory
        return 0

    @property
    def compute_capability(self) -> tuple[int, int]:
        if self._info is not None:
            return (self._info.compute_capability_major, self._info.compute_capability_minor)
        return (0, 0)

    def __repr__(self) -> str:
        return f"Device(device_id={self._device_id}, name={self.name!r})"
