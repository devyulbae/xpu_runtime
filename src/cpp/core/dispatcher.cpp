#include "xpuruntime/dispatcher.h"

#include "xpuruntime/kernel_registry.h"

#include <sstream>

namespace xpuruntime {

Dispatcher& Dispatcher::instance() {
  static Dispatcher instance;
  return instance;
}

Dispatcher::Dispatcher() = default;

void Dispatcher::set_policy(const KernelPolicy& policy) {
  std::lock_guard lock(mutex_);
  policy_ = policy;
}

KernelPolicy Dispatcher::get_policy() const {
  std::lock_guard lock(mutex_);
  return policy_;
}

void Dispatcher::dispatch(const std::string& op_name, KernelContext& ctx) {
  IKernel* kernel = select_kernel(op_name, ctx);
  if (kernel) {
    kernel->execute(ctx);
  }
}

IKernel* Dispatcher::select_kernel(const std::string& op_name, const KernelContext& ctx) {
  auto& registry = KernelRegistry::instance();
  std::vector<IKernel*> kernels = registry.get_kernels(op_name);

  std::lock_guard lock(mutex_);

  auto it = policy_.preferences.find(op_name);
  if (it != policy_.preferences.end()) {
    for (IKernel* k : kernels) {
      if (k->name() == it->second && k->supports(ctx)) {
        return k;
      }
    }
  }

  for (IKernel* k : kernels) {
    if (k->supports(ctx)) {
      return k;
    }
  }
  return nullptr;
}

std::vector<Dispatcher::DispatchLog> Dispatcher::get_dispatch_logs() const {
  std::lock_guard lock(mutex_);
  return logs_;
}

void Dispatcher::clear_dispatch_logs() {
  std::lock_guard lock(mutex_);
  logs_.clear();
}

void Dispatcher::enable_logging(bool enable) {
  std::lock_guard lock(mutex_);
  logging_enabled_ = enable;
}

std::string KernelPolicy::to_json() const {
  std::ostringstream oss;
  oss << "{\"preferences\":{";
  bool first = true;
  for (const auto& p : preferences) {
    if (!first) oss << ",";
    oss << "\"" << p.first << "\":\"" << p.second << "\"";
    first = false;
  }
  oss << "},\"auto_strategy\":"
      << (auto_strategy == AutoSelectStrategy::FirstSupported ? "0" : "1") << "}";
  return oss.str();
}

KernelPolicy KernelPolicy::from_json(const std::string& json) {
  (void)json;
  return KernelPolicy{};
}

}  // namespace xpuruntime
