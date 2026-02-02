#include "xpuruntime/memory_manager.h"

#include "xpuruntime/device_manager.h"
#include "xpuruntime/exceptions.h"

#include <cuda_runtime.h>
#include <cstring>

namespace xpuruntime {

MemoryManager& MemoryManager::instance() {
  static MemoryManager instance;
  return instance;
}

MemoryManager::MemoryManager() {
  int count = 0;
  if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
    device_pools_.resize(static_cast<size_t>(count));
  }
}

MemoryManager::~MemoryManager() {
  for (auto& pool : device_pools_) {
    std::lock_guard<std::mutex> lock(pool.mutex);
    for (auto& pair : pool.allocated_blocks) {
      cudaFree(pair.first);
    }
    for (auto& pair : pool.free_blocks) {
      for (void* ptr : pair.second) {
        cudaFree(ptr);
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(pinned_pool_.mutex);
    for (auto& pair : pinned_pool_.allocated_blocks) {
      cudaFreeHost(pair.first);
    }
    for (auto& pair : pinned_pool_.free_blocks) {
      for (void* ptr : pair.second) {
        cudaFreeHost(ptr);
      }
    }
  }
}

void* MemoryManager::allocate(size_t size, MemoryType type, int device_id) {
  if (type != MemoryType::Device) {
    if (type == MemoryType::Pinned) {
      return allocate_pinned(size);
    }
    throw UnsupportedOperationError("MemoryType not implemented");
  }
  int dev = (device_id >= 0) ? device_id : 0;
  if (static_cast<size_t>(dev) >= device_pools_.size()) {
    throw XpuRuntimeError("Invalid device_id for allocation");
  }
  void* ptr = nullptr;
  cudaError_t err = cudaSetDevice(dev);
  if (err != cudaSuccess) {
    throw CudaError(err, "cudaSetDevice");
  }
  err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    if (err == cudaErrorMemoryAllocation) {
      throw OutOfMemoryError(size, 0);
    }
    throw CudaError(err, "cudaMalloc");
  }
  return ptr;
}

void MemoryManager::deallocate(void* ptr) {
  if (ptr == nullptr) return;
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    throw CudaError(err, "cudaFree");
  }
}

void* MemoryManager::allocate_pinned(size_t size) {
  void* ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, size);
  if (err != cudaSuccess) {
    if (err == cudaErrorMemoryAllocation) {
      throw OutOfMemoryError(size, 0);
    }
    throw CudaError(err, "cudaMallocHost");
  }
  return ptr;
}

void MemoryManager::deallocate_pinned(void* ptr) {
  if (ptr == nullptr) return;
  cudaFreeHost(ptr);
}

size_t MemoryManager::get_allocated_size(int device_id) const {
  (void)device_id;
  return 0;
}

size_t MemoryManager::get_cached_size(int device_id) const {
  (void)device_id;
  return 0;
}

void MemoryManager::empty_cache(int device_id) {
  (void)device_id;
}

void MemoryManager::trim_cache(size_t target_size, int device_id) {
  (void)target_size;
  (void)device_id;
}

MemoryManager::Stats MemoryManager::get_stats(int device_id) const {
  (void)device_id;
  return {0, 0, 0, 0, 0};
}

}  // namespace xpuruntime
