#include "xpuruntime/backends/ibackend.h"

#include <unordered_map>

namespace xpuruntime {

BackendRegistry& BackendRegistry::instance() {
  static BackendRegistry instance;
  return instance;
}

BackendRegistry::BackendRegistry() = default;

void BackendRegistry::register_backend(std::unique_ptr<IBackend> backend) {
  if (backend) {
    backends_[backend->name()] = std::move(backend);
  }
}

IBackend* BackendRegistry::get_backend(const std::string& name) {
  auto it = backends_.find(name);
  if (it == backends_.end()) return nullptr;
  return it->second.get();
}

std::vector<std::string> BackendRegistry::get_available_backends() const {
  std::vector<std::string> result;
  for (const auto& pair : backends_) {
    result.push_back(pair.first);
  }
  return result;
}

}  // namespace xpuruntime
