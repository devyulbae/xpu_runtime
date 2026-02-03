#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xpuruntime/memory_manager.h"

namespace py = pybind11;
using namespace xpuruntime;

void bind_memory(py::module_& m) {
  py::enum_<MemoryType>(m, "MemoryType")
      .value("Device", MemoryType::Device)
      .value("Pinned", MemoryType::Pinned)
      .value("Managed", MemoryType::Managed)
      .export_values();

  py::class_<MemoryManager::Stats>(m, "MemoryStats")
      .def_readonly("total_allocated", &MemoryManager::Stats::total_allocated)
      .def_readonly("total_cached", &MemoryManager::Stats::total_cached)
      .def_readonly("peak_allocated", &MemoryManager::Stats::peak_allocated)
      .def_readonly("allocation_count", &MemoryManager::Stats::allocation_count)
      .def_readonly("deallocation_count", &MemoryManager::Stats::deallocation_count);

  py::class_<MemoryManager, std::unique_ptr<MemoryManager, py::nodelete>>(m, "MemoryManager")
      .def_static("instance", &MemoryManager::instance, py::return_value_policy::reference)
      .def("get_allocated_size", &MemoryManager::get_allocated_size, py::arg("device_id") = -1)
      .def("get_cached_size", &MemoryManager::get_cached_size, py::arg("device_id") = -1)
      .def("get_stats", &MemoryManager::get_stats, py::arg("device_id") = -1)
      .def("empty_cache", &MemoryManager::empty_cache, py::arg("device_id") = -1);
}
