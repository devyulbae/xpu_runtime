#include "xpuruntime/stream_manager.h"

#include "xpuruntime/device_manager.h"
#include "xpuruntime/exceptions.h"

#include <cuda_runtime.h>

namespace xpuruntime {

StreamManager& StreamManager::instance() {
  static StreamManager instance;
  return instance;
}

StreamManager::StreamManager() = default;

StreamManager::~StreamManager() {
  for (StreamHandle s : active_streams_) {
    cudaStreamDestroy(s);
  }
  for (EventHandle e : active_events_) {
    cudaEventDestroy(e);
  }
  for (StreamHandle s : default_streams_) {
    if (s != nullptr) {
      cudaStreamDestroy(s);
    }
  }
}

StreamHandle StreamManager::create_stream(const StreamConfig& config) {
  cudaStream_t stream = nullptr;
  unsigned int flags = config.non_blocking ? cudaStreamNonBlocking : 0;
  cudaError_t err = cudaStreamCreateWithPriority(&stream, flags, config.priority);
  if (err != cudaSuccess) {
    throw CudaError(err, "cudaStreamCreateWithPriority");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  active_streams_.insert(stream);
  return stream;
}

void StreamManager::destroy_stream(StreamHandle stream) {
  if (stream == nullptr) return;
  std::lock_guard<std::mutex> lock(mutex_);
  active_streams_.erase(stream);
  cudaStreamDestroy(stream);
}

StreamHandle StreamManager::get_default_stream(int device_id) {
  if (device_id < 0) {
    cudaGetDevice(&device_id);
  }
  size_t idx = static_cast<size_t>(device_id);
  if (default_streams_.size() <= idx) {
    default_streams_.resize(idx + 1, nullptr);
  }
  if (default_streams_[idx] == nullptr) {
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    default_streams_[idx] = s;
  }
  return default_streams_[idx];
}

EventHandle StreamManager::create_event(bool enable_timing) {
  cudaEvent_t event = nullptr;
  unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
  cudaError_t err = cudaEventCreateWithFlags(&event, flags);
  if (err != cudaSuccess) {
    throw CudaError(err, "cudaEventCreateWithFlags");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  active_events_.insert(event);
  return event;
}

void StreamManager::destroy_event(EventHandle event) {
  if (event == nullptr) return;
  std::lock_guard<std::mutex> lock(mutex_);
  active_events_.erase(event);
  cudaEventDestroy(event);
}

void StreamManager::record_event(EventHandle event, StreamHandle stream) {
  CUDA_CHECK(cudaEventRecord(event, stream));
}

void StreamManager::wait_event(StreamHandle stream, EventHandle event) {
  CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
}

float StreamManager::elapsed_time(EventHandle start, EventHandle end) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  return ms;
}

void StreamManager::synchronize_stream(StreamHandle stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void StreamManager::synchronize_device(int device_id) {
  if (device_id >= 0) {
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device_id);
    cudaDeviceSynchronize();
    cudaSetDevice(prev);
  } else {
    cudaDeviceSynchronize();
  }
}

StreamManager::GraphCapture StreamManager::begin_capture(StreamHandle stream) {
  GraphCapture cap{};
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  cap.graph = nullptr;
  cap.exec = nullptr;
  return cap;
}

void StreamManager::end_capture(GraphCapture& capture) {
  (void)capture;
  throw UnsupportedOperationError("end_capture: stub not implemented");
}

void StreamManager::launch_graph(GraphCapture& capture, StreamHandle stream) {
  (void)capture;
  (void)stream;
  throw UnsupportedOperationError("launch_graph: stub not implemented");
}

void StreamManager::destroy_graph(GraphCapture& capture) {
  if (capture.graph != nullptr) {
    cudaGraphDestroy(capture.graph);
    capture.graph = nullptr;
  }
  if (capture.exec != nullptr) {
    cudaGraphExecDestroy(capture.exec);
    capture.exec = nullptr;
  }
}

}  // namespace xpuruntime
