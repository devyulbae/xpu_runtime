# TASK_006: StreamManager 구현

> Phase 2: Core Features

---

## 개요

CUDA Stream과 Event 관리 기능을 구현한다.

## 목표

- Stream 생성/파괴/관리
- Event 생성/기록/대기
- Stream 간 동기화
- CUDA Graph 캡처 지원

## 선행 작업

- TASK_005: MemoryManager 구현

## 작업 내용

### 1. Stream 관리

```cpp
// src/cpp/core/stream_manager.cpp

StreamHandle StreamManager::create_stream(const StreamConfig& config) {
    cudaStream_t stream;
    
    if (config.non_blocking) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    } else {
        unsigned int flags = cudaStreamDefault;
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, flags, config.priority));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    active_streams_.insert(stream);
    return stream;
}

void StreamManager::destroy_stream(StreamHandle stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    std::lock_guard<std::mutex> lock(mutex_);
    active_streams_.erase(stream);
}
```

### 2. Event 관리

```cpp
EventHandle StreamManager::create_event(bool enable_timing) {
    cudaEvent_t event;
    unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
    return event;
}

void StreamManager::record_event(EventHandle event, StreamHandle stream) {
    CUDA_CHECK(cudaEventRecord(event, stream));
}

void StreamManager::wait_event(StreamHandle stream, EventHandle event) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
}

float StreamManager::elapsed_time(EventHandle start, EventHandle end) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}
```

### 3. CUDA Graph

```cpp
GraphCapture StreamManager::begin_capture(StreamHandle stream) {
    GraphCapture capture;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    return capture;
}

void StreamManager::end_capture(GraphCapture& capture) {
    CUDA_CHECK(cudaStreamEndCapture(stream_, &capture.graph));
    CUDA_CHECK(cudaGraphInstantiate(&capture.exec, capture.graph, nullptr, nullptr, 0));
}

void StreamManager::launch_graph(GraphCapture& capture, StreamHandle stream) {
    CUDA_CHECK(cudaGraphLaunch(capture.exec, stream));
}
```

### 4. Python 바인딩

```python
# src/python/xpuruntime/runtime/stream.py
class Stream:
    def __init__(self, priority: int = 0, non_blocking: bool = False):
        # ...
    
    def synchronize(self):
        # ...
```

## 완료 조건

- [ ] Stream 생성/파괴 동작
- [ ] Event 기록/대기 동작
- [ ] 시간 측정 정확성 확인
- [ ] CUDA Graph 캡처/재생 동작
- [ ] Python에서 Stream 사용 가능

## 예상 소요 시간

4-6시간

## 관련 문서

- [02_cpp_core_runtime.md](../02_cpp_core_runtime.md)
