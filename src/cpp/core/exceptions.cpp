#include "xpuruntime/exceptions.h"

namespace xpuruntime {

CudaError::CudaError(cudaError_t error, const std::string& context)
    : XpuRuntimeError(std::string("CUDA error: ") + (context.empty() ? cudaGetErrorString(error) : context + ": " + cudaGetErrorString(error))),
      error_(error) {}

OutOfMemoryError::OutOfMemoryError(size_t requested, size_t available)
    : XpuRuntimeError("Out of memory: requested " + std::to_string(requested) + ", available " + std::to_string(available)) {}

}  // namespace xpuruntime
