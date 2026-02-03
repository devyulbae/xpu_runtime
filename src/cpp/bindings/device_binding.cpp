#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xpuruntime/device_manager.h"

namespace py = pybind11;
using namespace xpuruntime;

void bind_device(py::module_& m) {
  py::class_<DeviceInfo>(m, "DeviceInfo")
      .def_readonly("device_id", &DeviceInfo::device_id)
      .def_readonly("name", &DeviceInfo::name)
      .def_readonly("total_memory", &DeviceInfo::total_memory)
      .def_readonly("free_memory", &DeviceInfo::free_memory)
      .def_readonly("compute_capability_major", &DeviceInfo::compute_capability_major)
      .def_readonly("compute_capability_minor", &DeviceInfo::compute_capability_minor)
      .def_readonly("sm_count", &DeviceInfo::sm_count)
      .def_readonly("supports_fp16", &DeviceInfo::supports_fp16)
      .def_readonly("supports_bf16", &DeviceInfo::supports_bf16)
      .def_readonly("supports_int8", &DeviceInfo::supports_int8)
      .def_readonly("supports_fp8", &DeviceInfo::supports_fp8)
      .def_readonly("supports_int4", &DeviceInfo::supports_int4)
      .def("__repr__", [](const DeviceInfo& info) {
        return "<DeviceInfo " + std::to_string(info.device_id) + ": " + info.name + " (" +
               std::to_string(info.total_memory / (1024 * 1024 * 1024)) + " GB)>";
      });

  py::class_<DeviceManager, std::unique_ptr<DeviceManager, py::nodelete>>(m, "DeviceManager")
      .def_static("instance", &DeviceManager::instance, py::return_value_policy::reference)
      .def("get_device_count", &DeviceManager::get_device_count)
      .def("get_all_devices", &DeviceManager::get_all_devices)
      .def("get_device_info", &DeviceManager::get_device_info)
      .def("get_current_device", &DeviceManager::get_current_device)
      .def("set_current_device", &DeviceManager::set_current_device)
      .def("synchronize", &DeviceManager::synchronize, py::arg("device_id") = -1);
}
