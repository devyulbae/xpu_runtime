#ifndef XPURUNTIME_EXCEPTIONS_H
#define XPURUNTIME_EXCEPTIONS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace xpuruntime {

class XpuRuntimeError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class CudaError : public XpuRuntimeError {
 public:
  CudaError(cudaError_t error, const std::string& context = "");
  cudaError_t cuda_error() const { return error_; }

 private:
  cudaError_t error_;
};

class OutOfMemoryError : public XpuRuntimeError {
 public:
  OutOfMemoryError(size_t requested, size_t available);
};

class UnsupportedOperationError : public XpuRuntimeError {
 public:
  using XpuRuntimeError::XpuRuntimeError;
};

#define CUDA_CHECK(expr)                                    \
  do {                                                      \
    cudaError_t err = (expr);                                \
    if (err != cudaSuccess) {                               \
      throw xpuruntime::CudaError(err, #expr);               \
    }                                                       \
  } while (0)

}  // namespace xpuruntime

#endif  // XPURUNTIME_EXCEPTIONS_H
