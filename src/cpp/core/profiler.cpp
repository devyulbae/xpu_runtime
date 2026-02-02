#include "xpuruntime/profiler.h"

#include <fstream>
#include <mutex>

namespace xpuruntime {

Profiler& Profiler::instance() {
  static Profiler instance;
  return instance;
}

Profiler::Profiler() = default;

void Profiler::enable() {
  std::lock_guard lock(mutex_);
  enabled_ = true;
}

void Profiler::disable() {
  std::lock_guard lock(mutex_);
  enabled_ = false;
}

bool Profiler::is_enabled() const {
  std::lock_guard lock(mutex_);
  return enabled_;
}

void Profiler::push_range(const std::string& name) {
  (void)name;
}

void Profiler::pop_range() {}

Profiler::ScopedRange::ScopedRange(const std::string& name) {
  Profiler::instance().push_range(name);
}

Profiler::ScopedRange::~ScopedRange() {
  Profiler::instance().pop_range();
}

std::vector<Profiler::ProfileStats> Profiler::get_stats() const {
  std::lock_guard lock(mutex_);
  std::vector<ProfileStats> result;
  for (const auto& pair : stats_) {
    result.push_back(pair.second);
  }
  return result;
}

void Profiler::export_chrome_trace(const std::string& filename) {
  (void)filename;
}

void Profiler::export_json(const std::string& filename) {
  (void)filename;
}

void Profiler::reset() {
  std::lock_guard lock(mutex_);
  kernel_profiles_.clear();
  stats_.clear();
}

}  // namespace xpuruntime
