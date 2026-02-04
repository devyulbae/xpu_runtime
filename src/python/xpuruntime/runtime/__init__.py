"""xpuruntime.runtime - Device, stream, memory, profiling API."""

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
    "Device",
    "DeviceCountsType",
    "format_device_counts",
    "get_device_count",
    "get_device_counts",
    "get_current_device",
    "set_current_device",
]
