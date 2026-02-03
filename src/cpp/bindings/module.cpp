#include <pybind11/pybind11.h>

#include "xpuruntime/config.hpp"

namespace py = pybind11;

void bind_exceptions(py::module_& m);
void bind_device(py::module_& m);
void bind_memory(py::module_& m);

PYBIND11_MODULE(_core, m) {
  m.doc() = "xpuruntime C++ core bindings";

  bind_exceptions(m);
  bind_device(m);
  bind_memory(m);

  m.def("get_version", &xpuruntime::get_version, "Return runtime version string.");
}
