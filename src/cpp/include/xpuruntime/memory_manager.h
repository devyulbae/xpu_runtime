#ifndef XPURUNTIME_MEMORY_MANAGER_H
#define XPURUNTIME_MEMORY_MANAGER_H

#include "xpuruntime/common.h"

#include <cstddef>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace xpuruntime {

enum class MemoryType {
  Device,
  Pinned,
  Managed,
};

struct MemoryBlock {
  void* ptr;
  size_t size;
  MemoryType type;
  int device_id;
};

class MemoryManager {
 public:
  static MemoryManager& instance();

  void* allocate(size_t size, MemoryType type = MemoryType::Device, int device_id = -1);
  void deallocate(void* ptr);

  void* allocate_pinned(size_t size);
  void deallocate_pinned(void* ptr);

  size_t get_allocated_size(int device_id = -1) const;
  size_t get_cached_size(int device_id = -1) const;

  void empty_cache(int device_id = -1);
  void trim_cache(size_t target_size, int device_id = -1);

  struct Stats {
    size_t total_allocated;
    size_t total_cached;
    size_t peak_allocated;
    int64_t allocation_count;
    int64_t deallocation_count;
  };
  Stats get_stats(int device_id = -1) const;

 private:
  MemoryManager();
  ~MemoryManager();
  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;

  struct Pool {
    std::map<size_t, std::vector<void*>> free_blocks;
    std::unordered_map<void*, size_t> allocated_blocks;
    std::mutex mutex;
  };

  std::vector<Pool> device_pools_;
  Pool pinned_pool_;
};

}  // namespace xpuruntime

#endif  // XPURUNTIME_MEMORY_MANAGER_H
