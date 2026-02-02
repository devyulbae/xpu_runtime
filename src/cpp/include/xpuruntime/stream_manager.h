#ifndef XPURUNTIME_STREAM_MANAGER_H
#define XPURUNTIME_STREAM_MANAGER_H

#include "xpuruntime/common.h"

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace xpuruntime {

struct StreamConfig {
  int priority = 0;
  bool non_blocking = false;
};

class StreamManager {
 public:
  static StreamManager& instance();

  StreamHandle create_stream(const StreamConfig& config = {});
  void destroy_stream(StreamHandle stream);
  StreamHandle get_default_stream(int device_id = -1);

  EventHandle create_event(bool enable_timing = false);
  void destroy_event(EventHandle event);
  void record_event(EventHandle event, StreamHandle stream = nullptr);
  void wait_event(StreamHandle stream, EventHandle event);
  float elapsed_time(EventHandle start, EventHandle end);

  void synchronize_stream(StreamHandle stream);
  void synchronize_device(int device_id = -1);

  struct GraphCapture {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
  };
  GraphCapture begin_capture(StreamHandle stream);
  void end_capture(GraphCapture& capture);
  void launch_graph(GraphCapture& capture, StreamHandle stream);
  void destroy_graph(GraphCapture& capture);

 private:
  StreamManager();
  ~StreamManager();
  StreamManager(const StreamManager&) = delete;
  StreamManager& operator=(const StreamManager&) = delete;

  std::vector<StreamHandle> default_streams_;
  std::unordered_set<StreamHandle> active_streams_;
  std::unordered_set<EventHandle> active_events_;
  std::mutex mutex_;
};

}  // namespace xpuruntime

#endif  // XPURUNTIME_STREAM_MANAGER_H
