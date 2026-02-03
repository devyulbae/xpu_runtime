#include <pybind11/pybind11.h>

#include "xpuruntime/exceptions.h"

namespace py = pybind11;
using namespace xpuruntime;

void bind_exceptions(py::module_& m) {
  static py::exception<XpuRuntimeError> exc_runtime(m, "XpuRuntimeError");
  static py::exception<CudaError> exc_cuda(m, "CudaError", exc_runtime.ptr());
  static py::exception<OutOfMemoryError> exc_oom(m, "OutOfMemoryError", exc_runtime.ptr());
  static py::exception<UnsupportedOperationError> exc_unsupported(
      m, "UnsupportedOperationError", exc_runtime.ptr());

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const OutOfMemoryError& e) {
      exc_oom(e.what());
    } catch (const CudaError& e) {
      exc_cuda(e.what());
    } catch (const UnsupportedOperationError& e) {
      exc_unsupported(e.what());
    } catch (const XpuRuntimeError& e) {
      exc_runtime(e.what());
    }
  });
}
