#include "xpuruntime/config.hpp"
#include "xpuruntime/device_manager.h"
#include "xpuruntime/dispatcher.h"
#include "xpuruntime/kernel_registry.h"
#include "xpuruntime/memory_manager.h"
#include "xpuruntime/profiler.h"
#include "xpuruntime/stream_manager.h"
#include "xpuruntime/backends/ibackend.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
  std::cout << "xpuruntime C++ core skeleton test\n";

  assert(xpuruntime::get_version() != nullptr);
  std::cout << "  get_version(): " << xpuruntime::get_version() << "\n";

  auto& dm = xpuruntime::DeviceManager::instance();
  int count = dm.get_device_count();
  std::cout << "  DeviceManager::get_device_count(): " << count << "\n";

  auto& mm = xpuruntime::MemoryManager::instance();
  (void)mm.get_allocated_size();
  std::cout << "  MemoryManager::instance() ok\n";

  auto& sm = xpuruntime::StreamManager::instance();
  if (count > 0) {
    (void)sm.get_default_stream(0);
  }
  std::cout << "  StreamManager::instance() ok\n";

  auto& kr = xpuruntime::KernelRegistry::instance();
  auto ops = kr.get_registered_ops();
  std::cout << "  KernelRegistry::get_registered_ops(): " << ops.size() << " ops\n";

  auto& disp = xpuruntime::Dispatcher::instance();
  auto policy = disp.get_policy();
  (void)policy.to_json();
  std::cout << "  Dispatcher::get_policy() ok\n";

  auto& prof = xpuruntime::Profiler::instance();
  assert(!prof.is_enabled());
  std::cout << "  Profiler::instance() ok\n";

  auto& back = xpuruntime::BackendRegistry::instance();
  auto backends = back.get_available_backends();
  std::cout << "  BackendRegistry::get_available_backends(): " << backends.size() << " backends\n";

  std::cout << "All skeleton tests passed.\n";
  return 0;
}
