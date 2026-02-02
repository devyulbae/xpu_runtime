#ifndef XPURUNTIME_BACKENDS_IBACKEND_H
#define XPURUNTIME_BACKENDS_IBACKEND_H

#include "xpuruntime/kernel_registry.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace xpuruntime {

struct OpDescriptor {
  std::string op_type;
  std::vector<std::string> input_dtypes;
  std::vector<std::vector<int64_t>> input_shapes;
  std::map<std::string, std::string> attributes;
};

class IBackend {
 public:
  virtual ~IBackend() = default;

  virtual std::string name() const = 0;
  virtual void initialize() = 0;
  virtual void finalize() = 0;
  virtual bool supports(const OpDescriptor& op) const = 0;
  virtual void execute(const OpDescriptor& op, const KernelContext& ctx) = 0;
};

class BackendRegistry {
 public:
  static BackendRegistry& instance();

  void register_backend(std::unique_ptr<IBackend> backend);
  IBackend* get_backend(const std::string& name);
  std::vector<std::string> get_available_backends() const;

 private:
  BackendRegistry();
  BackendRegistry(const BackendRegistry&) = delete;
  BackendRegistry& operator=(const BackendRegistry&) = delete;

  std::unordered_map<std::string, std::unique_ptr<IBackend>> backends_;
};

#define REGISTER_BACKEND(backend_class)                                           \
  static bool _reg_backend_##backend_class = []() {                                \
    xpuruntime::BackendRegistry::instance().register_backend(                      \
        std::make_unique<backend_class>());                                        \
    return true;                                                                  \
  }()

}  // namespace xpuruntime

#endif  // XPURUNTIME_BACKENDS_IBACKEND_H
