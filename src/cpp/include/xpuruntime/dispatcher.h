#ifndef XPURUNTIME_DISPATCHER_H
#define XPURUNTIME_DISPATCHER_H

#include "xpuruntime/kernel_registry.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xpuruntime {

struct KernelPolicy {
  std::unordered_map<std::string, std::string> preferences;
  enum class AutoSelectStrategy {
    FirstSupported,
    BestPerformance,
  };
  AutoSelectStrategy auto_strategy = AutoSelectStrategy::FirstSupported;

  std::string to_json() const;
  static KernelPolicy from_json(const std::string& json);
};

class Dispatcher {
 public:
  static Dispatcher& instance();

  void set_policy(const KernelPolicy& policy);
  KernelPolicy get_policy() const;

  void dispatch(const std::string& op_name, KernelContext& ctx);
  IKernel* select_kernel(const std::string& op_name, const KernelContext& ctx);

  struct DispatchLog {
    std::string op_name;
    std::string selected_kernel;
    std::vector<int64_t> input_shape;
    std::string dtype;
    double elapsed_us;
  };
  std::vector<DispatchLog> get_dispatch_logs() const;
  void clear_dispatch_logs();
  void enable_logging(bool enable);

 private:
  Dispatcher();
  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;

  KernelPolicy policy_;
  std::vector<DispatchLog> logs_;
  bool logging_enabled_ = false;
  mutable std::mutex mutex_;
};

}  // namespace xpuruntime

#endif  // XPURUNTIME_DISPATCHER_H
