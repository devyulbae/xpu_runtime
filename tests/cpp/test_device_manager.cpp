#include "xpuruntime/device_manager.h"

#include <cassert>
#include <iostream>

int main() {
  auto& dm = xpuruntime::DeviceManager::instance();
  int count = dm.get_device_count();
  std::cout << "DeviceManager::get_device_count() = " << count << "\n";
  assert(count >= 0);

  if (count > 0) {
    auto info = dm.get_device_info(0);
    assert(info.device_id == 0);
    assert(!info.name.empty());
    std::cout << "  Device 0: " << info.name << ", " << (info.total_memory / (1024 * 1024)) << " MB\n";

    int current = dm.get_current_device();
    dm.set_current_device(0);
    assert(dm.get_current_device() == 0);
    dm.set_current_device(current >= 0 ? current : 0);
  }

  std::cout << "DeviceManager tests passed.\n";
  return 0;
}
