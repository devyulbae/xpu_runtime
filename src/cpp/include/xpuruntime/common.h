#ifndef XPURUNTIME_COMMON_H
#define XPURUNTIME_COMMON_H

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace xpuruntime {

using DeviceId = int;
using StreamHandle = cudaStream_t;
using EventHandle = cudaEvent_t;

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;
};

}  // namespace xpuruntime

#endif  // XPURUNTIME_COMMON_H
