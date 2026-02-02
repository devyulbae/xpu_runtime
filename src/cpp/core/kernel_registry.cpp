#include "xpuruntime/kernel_registry.h"

#include <shared_mutex>

namespace xpuruntime {

KernelRegistry& KernelRegistry::instance() {
  static KernelRegistry instance;
  return instance;
}

KernelRegistry::KernelRegistry() = default;

void KernelRegistry::register_kernel(const std::string& op_name, std::unique_ptr<IKernel> kernel) {
  std::unique_lock lock(mutex_);
  registry_[op_name].push_back(std::move(kernel));
}

std::vector<IKernel*> KernelRegistry::get_kernels(const std::string& op_name) const {
  std::shared_lock lock(mutex_);
  std::vector<IKernel*> result;
  auto it = registry_.find(op_name);
  if (it != registry_.end()) {
    for (auto& k : it->second) {
      result.push_back(k.get());
    }
  }
  return result;
}

IKernel* KernelRegistry::get_kernel(const std::string& op_name, const std::string& kernel_name) const {
  std::shared_lock lock(mutex_);
  auto it = registry_.find(op_name);
  if (it == registry_.end()) return nullptr;
  for (auto& k : it->second) {
    if (k->name() == kernel_name) return k.get();
  }
  return nullptr;
}

std::vector<std::string> KernelRegistry::get_registered_ops() const {
  std::shared_lock lock(mutex_);
  std::vector<std::string> result;
  for (const auto& pair : registry_) {
    result.push_back(pair.first);
  }
  return result;
}

std::vector<std::string> KernelRegistry::get_kernel_names(const std::string& op_name) const {
  std::shared_lock lock(mutex_);
  std::vector<std::string> result;
  auto it = registry_.find(op_name);
  if (it != registry_.end()) {
    for (auto& k : it->second) {
      result.push_back(k->name());
    }
  }
  return result;
}

}  // namespace xpuruntime
