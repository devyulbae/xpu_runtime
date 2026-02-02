#ifndef XPURUNTIME_PROFILER_H
#define XPURUNTIME_PROFILER_H

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace xpuruntime {

class Profiler {
 public:
  static Profiler& instance();

  void enable();
  void disable();
  bool is_enabled() const;

  void push_range(const std::string& name);
  void pop_range();

  class ScopedRange {
   public:
    explicit ScopedRange(const std::string& name);
    ~ScopedRange();

   private:
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
  };

  struct KernelProfile {
    std::string name;
    double elapsed_us;
    size_t grid_size;
    size_t block_size;
    size_t shared_memory;
  };

  struct ProfileStats {
    std::string op_name;
    int64_t call_count;
    double total_time_us;
    double avg_time_us;
    double min_time_us;
    double max_time_us;
  };
  std::vector<ProfileStats> get_stats() const;

  void export_chrome_trace(const std::string& filename);
  void export_json(const std::string& filename);
  void reset();

 private:
  Profiler();
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  bool enabled_ = false;
  std::vector<KernelProfile> kernel_profiles_;
  std::map<std::string, ProfileStats> stats_;
  mutable std::mutex mutex_;
};

#define XRT_PROFILE_SCOPE(name) \
  xpuruntime::Profiler::ScopedRange _profile_scope_##__LINE__(name)

}  // namespace xpuruntime

#endif  // XPURUNTIME_PROFILER_H
