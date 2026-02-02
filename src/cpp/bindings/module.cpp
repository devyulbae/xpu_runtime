#include <pybind11/pybind11.h>
#include "xpuruntime/config.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "xpuruntime C++ core bindings (scaffold)";
  m.def("get_version", &xpuruntime::get_version, "Return runtime version string.");
}
