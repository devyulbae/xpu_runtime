#ifndef XPURUNTIME_DEVICE_MANAGER_H
#define XPURUNTIME_DEVICE_MANAGER_H

#include "xpuruntime/common.h"

#include <atomic>
#include <string>
#include <vector>

namespace xpuruntime {

struct DeviceInfo {
  int device_id;
  std::string name;
  size_t total_memory;
  size_t free_memory;
  int compute_capability_major;
  int compute_capability_minor;
  int sm_count;
  bool supports_fp16;
  bool supports_bf16;
  bool supports_int8;
};

class DeviceManager {
 public:
  static DeviceManager& instance();

  int get_device_count() const;
  std::vector<DeviceInfo> get_all_devices() const;
  DeviceInfo get_device_info(int device_id) const;

  int get_current_device() const;
  void set_current_device(int device_id);

  void synchronize(int device_id = -1);

 private:
  DeviceManager();
  ~DeviceManager();
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;

  std::vector<DeviceInfo> devices_;
  std::atomic<int> current_device_;
};

}  // namespace xpuruntime

#endif  // XPURUNTIME_DEVICE_MANAGER_H
