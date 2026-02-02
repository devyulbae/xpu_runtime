#include "xpuruntime/device_manager.h"

#include "xpuruntime/exceptions.h"

#include <cuda_runtime.h>

namespace xpuruntime {

DeviceManager& DeviceManager::instance() {
  static DeviceManager instance;
  return instance;
}

DeviceManager::DeviceManager() : current_device_(0) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    devices_.clear();
    return;
  }
  devices_.resize(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    DeviceInfo& info = devices_[static_cast<size_t>(i)];
    info.device_id = i;
    info.name = "GPU " + std::to_string(i);
    info.total_memory = 0;
    info.free_memory = 0;
    info.compute_capability_major = 0;
    info.compute_capability_minor = 0;
    info.sm_count = 0;
    info.supports_fp16 = false;
    info.supports_bf16 = false;
    info.supports_int8 = false;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      info.name = prop.name;
      info.total_memory = prop.totalGlobalMem;
      size_t free_mem = 0;
      size_t total_mem = 0;
      if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        info.free_memory = free_mem;
      }
      info.compute_capability_major = prop.major;
      info.compute_capability_minor = prop.minor;
      info.sm_count = prop.multiProcessorCount;
      info.supports_fp16 = (prop.major >= 5);
      info.supports_bf16 = (prop.major >= 8);
      info.supports_int8 = true;
    }
  }
}

DeviceManager::~DeviceManager() = default;

int DeviceManager::get_device_count() const {
  return static_cast<int>(devices_.size());
}

std::vector<DeviceInfo> DeviceManager::get_all_devices() const {
  return devices_;
}

DeviceInfo DeviceManager::get_device_info(int device_id) const {
  int id = (device_id >= 0) ? device_id : current_device_.load();
  if (id < 0 || static_cast<size_t>(id) >= devices_.size()) {
    throw XpuRuntimeError("Invalid device_id: " + std::to_string(device_id));
  }
  return devices_[static_cast<size_t>(id)];
}

int DeviceManager::get_current_device() const {
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    return 0;
  }
  return dev;
}

void DeviceManager::set_current_device(int device_id) {
  CUDA_CHECK(cudaSetDevice(device_id));
  current_device_.store(device_id);
}

void DeviceManager::synchronize(int device_id) {
  if (device_id >= 0) {
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device_id);
    cudaDeviceSynchronize();
    cudaSetDevice(prev);
  } else {
    cudaDeviceSynchronize();
  }
}

}  // namespace xpuruntime
