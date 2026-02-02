#ifndef XPURUNTIME_KERNEL_REGISTRY_H
#define XPURUNTIME_KERNEL_REGISTRY_H

#include "xpuruntime/common.h"

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xpuruntime {

struct KernelContext {
  std::vector<void*> inputs;
  std::vector<void*> outputs;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::string dtype;
  int device_id;
  StreamHandle stream;
  void* workspace;
  size_t workspace_size;
};

class IKernel {
 public:
  virtual ~IKernel() = default;

  virtual std::string name() const = 0;
  virtual bool supports(const KernelContext& ctx) const = 0;
  virtual size_t workspace_size(const KernelContext& ctx) const { return 0; }
  virtual void execute(const KernelContext& ctx) = 0;
};

class KernelRegistry {
 public:
  static KernelRegistry& instance();

  void register_kernel(const std::string& op_name, std::unique_ptr<IKernel> kernel);

  std::vector<IKernel*> get_kernels(const std::string& op_name) const;
  IKernel* get_kernel(const std::string& op_name, const std::string& kernel_name) const;

  std::vector<std::string> get_registered_ops() const;
  std::vector<std::string> get_kernel_names(const std::string& op_name) const;

 private:
  KernelRegistry();
  KernelRegistry(const KernelRegistry&) = delete;
  KernelRegistry& operator=(const KernelRegistry&) = delete;

  std::unordered_map<std::string, std::vector<std::unique_ptr<IKernel>>> registry_;
  mutable std::shared_mutex mutex_;
};

#define REGISTER_KERNEL(op_name, kernel_class)                                   \
  static bool _reg_##kernel_class = []() {                                        \
    xpuruntime::KernelRegistry::instance().register_kernel(                         \
        op_name, std::make_unique<kernel_class>());                                \
    return true;                                                                  \
  }()

}  // namespace xpuruntime

#endif  // XPURUNTIME_KERNEL_REGISTRY_H
