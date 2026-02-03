# xpuruntime - C++ 코어 런타임 설계

> 이 문서는 xpuruntime의 C++ 코어 런타임 컴포넌트를 상세히 설명한다.

---

## 1. 개요

C++ 코어 런타임은 xpuruntime의 **Data Plane**으로서, 모든 성능 크리티컬한 연산을 담당한다.

### 설계 목표

- **저지연**: 함수 호출 오버헤드 최소화
- **메모리 효율**: Pool allocator, zero-copy
- **확장성**: 새 backend 추가 용이
- **안정성**: 예외 안전, RAII 패턴

---

## 2. 컴포넌트 상세

### 2.1 DeviceManager

GPU/NPU 디바이스를 탐지하고 관리하는 싱글톤 컴포넌트.

#### 책임

- 시스템의 모든 GPU/NPU 열거
- 각 디바이스의 capability 조회 (SM version, memory size 등)
- Driver/Runtime 버전 확인
- 현재 활성 디바이스 관리

#### 인터페이스

```cpp
namespace xpuruntime {

struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int sm_count;
    bool supports_fp16;
    bool supports_bf16;
    bool supports_int8;
};

class DeviceManager {
public:
    static DeviceManager& instance();
    
    // 디바이스 열거
    int get_device_count() const;
    std::vector<DeviceInfo> get_all_devices() const;
    DeviceInfo get_device_info(int device_id) const;
    
    // 활성 디바이스
    int get_current_device() const;
    void set_current_device(int device_id);
    
    // 동기화
    void synchronize(int device_id = -1);  // -1 = current
    
private:
    DeviceManager();
    ~DeviceManager();
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    
    std::vector<DeviceInfo> devices_;
    std::atomic<int> current_device_;
};

}  // namespace xpuruntime
```

#### 사용 예시

```cpp
auto& dm = DeviceManager::instance();
int count = dm.get_device_count();

for (int i = 0; i < count; ++i) {
    auto info = dm.get_device_info(i);
    std::cout << "Device " << i << ": " << info.name 
              << " (" << info.total_memory / (1024*1024*1024) << " GB)" << std::endl;
}
```

---

### 2.2 MemoryManager

GPU 메모리 할당/해제를 관리하는 컴포넌트. Caching allocator로 할당 오버헤드를 최소화.

#### 책임

- Device memory 할당/해제 (caching)
- Pinned (page-locked) memory 관리
- 메모리 풀 관리 (fragmentation 최소화)
- 메모리 사용량 추적

#### 인터페이스

```cpp
namespace xpuruntime {

enum class MemoryType {
    Device,     // GPU global memory
    Pinned,     // Page-locked host memory
    Managed,    // Unified memory
};

struct MemoryBlock {
    void* ptr;
    size_t size;
    MemoryType type;
    int device_id;
};

class MemoryManager {
public:
    static MemoryManager& instance();
    
    // 할당/해제
    void* allocate(size_t size, MemoryType type = MemoryType::Device, int device_id = -1);
    void deallocate(void* ptr);
    
    // Pinned memory
    void* allocate_pinned(size_t size);
    void deallocate_pinned(void* ptr);
    
    // 메모리 정보
    size_t get_allocated_size(int device_id = -1) const;
    size_t get_cached_size(int device_id = -1) const;
    
    // 캐시 관리
    void empty_cache(int device_id = -1);
    void trim_cache(size_t target_size, int device_id = -1);
    
    // 통계
    struct Stats {
        size_t total_allocated;
        size_t total_cached;
        size_t peak_allocated;
        int64_t allocation_count;
        int64_t deallocation_count;
    };
    Stats get_stats(int device_id = -1) const;
    
private:
    MemoryManager();
    ~MemoryManager();
    
    // Caching allocator 구현
    struct Pool {
        std::map<size_t, std::vector<void*>> free_blocks;
        std::unordered_map<void*, size_t> allocated_blocks;
        std::mutex mutex;
    };
    
    std::vector<Pool> device_pools_;
    Pool pinned_pool_;
};

}  // namespace xpuruntime
```

#### Caching Allocator 동작

1. 할당 요청 시, 캐시된 블록 중 적합한 크기 검색
2. 적합한 블록 있으면 재사용, 없으면 새로 할당
3. 해제 시 바로 반환하지 않고 캐시에 보관
4. 메모리 압박 시 캐시 정리

```cpp
// 내부 할당 로직 (의사 코드)
void* MemoryManager::allocate_impl(size_t size, Pool& pool) {
    std::lock_guard<std::mutex> lock(pool.mutex);
    
    // Round up to block size for better reuse
    size_t block_size = round_up_power_of_2(size);
    
    // Check cache
    auto it = pool.free_blocks.find(block_size);
    if (it != pool.free_blocks.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        pool.allocated_blocks[ptr] = block_size;
        return ptr;
    }
    
    // Allocate new
    void* ptr = nullptr;
    cudaMalloc(&ptr, block_size);
    pool.allocated_blocks[ptr] = block_size;
    return ptr;
}
```

---

### 2.3 StreamManager

CUDA Stream과 Event를 관리하는 컴포넌트.

#### 책임

- Stream 생성/파괴
- Event 생성/기록/대기
- Stream 간 동기화
- CUDA Graph capture 지원

#### 인터페이스

```cpp
namespace xpuruntime {

using StreamHandle = cudaStream_t;
using EventHandle = cudaEvent_t;

struct StreamConfig {
    int priority = 0;           // Stream priority
    bool non_blocking = false;  // Non-blocking with respect to host
};

class StreamManager {
public:
    static StreamManager& instance();
    
    // Stream 관리
    StreamHandle create_stream(const StreamConfig& config = {});
    void destroy_stream(StreamHandle stream);
    StreamHandle get_default_stream(int device_id = -1);
    
    // Event 관리
    EventHandle create_event(bool enable_timing = false);
    void destroy_event(EventHandle event);
    void record_event(EventHandle event, StreamHandle stream = nullptr);
    void wait_event(StreamHandle stream, EventHandle event);
    float elapsed_time(EventHandle start, EventHandle end);  // ms
    
    // 동기화
    void synchronize_stream(StreamHandle stream);
    void synchronize_device(int device_id = -1);
    
    // CUDA Graph
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
    
    // Per-device default streams
    std::vector<StreamHandle> default_streams_;
    
    // Active streams/events tracking
    std::unordered_set<StreamHandle> active_streams_;
    std::unordered_set<EventHandle> active_events_;
    std::mutex mutex_;
};

}  // namespace xpuruntime
```

#### CUDA Graph 사용 예시

```cpp
auto& sm = StreamManager::instance();
auto stream = sm.get_default_stream();

// Capture
auto capture = sm.begin_capture(stream);

// ... execute operations on stream ...
kernel_a<<<blocks, threads, 0, stream>>>(...);
kernel_b<<<blocks, threads, 0, stream>>>(...);

sm.end_capture(capture);

// Replay
for (int i = 0; i < 1000; ++i) {
    sm.launch_graph(capture, stream);
}

sm.destroy_graph(capture);
```

---

### 2.4 KernelRegistry

연산(op)과 커널 구현체를 매핑하는 레지스트리.

#### 책임

- op 이름 → 구현체 목록 매핑
- 구현체별 지원 조건 관리 (dtype, device capability 등)
- 동적 커널 등록 지원

#### 인터페이스

```cpp
namespace xpuruntime {

// 커널 실행 컨텍스트
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

// 커널 구현체 인터페이스
class IKernel {
public:
    virtual ~IKernel() = default;
    
    // 커널 이름 (예: "cublas", "flash_v2")
    virtual std::string name() const = 0;
    
    // 지원 여부 확인
    virtual bool supports(const KernelContext& ctx) const = 0;
    
    // 필요한 workspace 크기
    virtual size_t workspace_size(const KernelContext& ctx) const { return 0; }
    
    // 실행
    virtual void execute(const KernelContext& ctx) = 0;
};

class KernelRegistry {
public:
    static KernelRegistry& instance();
    
    // 커널 등록
    void register_kernel(const std::string& op_name, std::unique_ptr<IKernel> kernel);
    
    // 커널 조회
    std::vector<IKernel*> get_kernels(const std::string& op_name) const;
    IKernel* get_kernel(const std::string& op_name, const std::string& kernel_name) const;
    
    // 등록된 op 목록
    std::vector<std::string> get_registered_ops() const;
    
    // 등록된 커널 이름 목록
    std::vector<std::string> get_kernel_names(const std::string& op_name) const;
    
private:
    KernelRegistry();
    
    // op_name -> [kernel1, kernel2, ...]
    std::unordered_map<std::string, std::vector<std::unique_ptr<IKernel>>> registry_;
    mutable std::shared_mutex mutex_;
};

// 커널 자동 등록 매크로
#define REGISTER_KERNEL(op_name, kernel_class) \
    static bool _reg_##kernel_class = []() { \
        KernelRegistry::instance().register_kernel( \
            op_name, std::make_unique<kernel_class>()); \
        return true; \
    }()

}  // namespace xpuruntime
```

#### 커널 구현 예시

```cpp
// cuBLAS GEMM 커널
class CublasGemmKernel : public IKernel {
public:
    std::string name() const override { return "cublas"; }
    
    bool supports(const KernelContext& ctx) const override {
        // fp16, fp32, bf16 지원
        return ctx.dtype == "float32" || 
               ctx.dtype == "float16" || 
               ctx.dtype == "bfloat16";
    }
    
    void execute(const KernelContext& ctx) override {
        // cuBLAS 호출
        // ...
    }
};

REGISTER_KERNEL("matmul", CublasGemmKernel);
```

---

### 2.5 Dispatcher

런타임에 최적의 커널/엔진을 선택하는 컴포넌트.

#### 책임

- KernelPolicy 기반 커널 선택
- 입력 조건(shape, dtype, device) 기반 필터링
- Fallback 처리
- 선택 결과 로깅

#### 인터페이스

```cpp
namespace xpuruntime {

// 커널 정책
struct KernelPolicy {
    // op_name -> 선호 커널 이름 (없으면 자동 선택)
    std::unordered_map<std::string, std::string> preferences;
    
    // 자동 선택 전략
    enum class AutoSelectStrategy {
        FirstSupported,  // 첫 번째 지원 커널
        BestPerformance, // 벤치마크 기반 (미래)
    };
    AutoSelectStrategy auto_strategy = AutoSelectStrategy::FirstSupported;
    
    // 직렬화
    std::string to_json() const;
    static KernelPolicy from_json(const std::string& json);
};

class Dispatcher {
public:
    static Dispatcher& instance();
    
    // 정책 설정
    void set_policy(const KernelPolicy& policy);
    KernelPolicy get_policy() const;
    
    // 커널 선택 및 실행
    void dispatch(const std::string& op_name, KernelContext& ctx);
    
    // 선택만 (실행 없이)
    IKernel* select_kernel(const std::string& op_name, const KernelContext& ctx);
    
    // 실행 로그
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
    
    KernelPolicy policy_;
    std::vector<DispatchLog> logs_;
    bool logging_enabled_ = false;
    mutable std::mutex mutex_;
};

}  // namespace xpuruntime
```

#### 디스패치 로직

```cpp
IKernel* Dispatcher::select_kernel(const std::string& op_name, const KernelContext& ctx) {
    auto& registry = KernelRegistry::instance();
    auto kernels = registry.get_kernels(op_name);
    
    if (kernels.empty()) {
        throw std::runtime_error("No kernel registered for op: " + op_name);
    }
    
    // 1. 정책에 명시된 커널이 있는지 확인
    auto it = policy_.preferences.find(op_name);
    if (it != policy_.preferences.end()) {
        for (auto* kernel : kernels) {
            if (kernel->name() == it->second && kernel->supports(ctx)) {
                return kernel;
            }
        }
        // 명시된 커널이 지원하지 않으면 경고 후 fallback
        log_warning("Preferred kernel '" + it->second + 
                   "' does not support current context, falling back");
    }
    
    // 2. 자동 선택
    for (auto* kernel : kernels) {
        if (kernel->supports(ctx)) {
            return kernel;
        }
    }
    
    throw std::runtime_error("No supported kernel found for op: " + op_name);
}
```

---

### 2.6 Profiler

실행 시간/메모리 사용량을 추적하는 프로파일링 컴포넌트.

#### 책임

- NVTX 마커 삽입
- CUPTI 기반 커널 프로파일링
- 실행 시간/메모리 통계 수집
- 프로파일 결과 내보내기

#### 인터페이스

```cpp
namespace xpuruntime {

class Profiler {
public:
    static Profiler& instance();
    
    // 프로파일링 활성화/비활성화
    void enable();
    void disable();
    bool is_enabled() const;
    
    // NVTX 마커
    void push_range(const std::string& name);
    void pop_range();
    
    // Scoped marker
    class ScopedRange {
    public:
        ScopedRange(const std::string& name);
        ~ScopedRange();
    };
    
    // 커널 프로파일링
    struct KernelProfile {
        std::string name;
        double elapsed_us;
        size_t grid_size;
        size_t block_size;
        size_t shared_memory;
    };
    
    // 통계
    struct ProfileStats {
        std::string op_name;
        int64_t call_count;
        double total_time_us;
        double avg_time_us;
        double min_time_us;
        double max_time_us;
    };
    std::vector<ProfileStats> get_stats() const;
    
    // 내보내기
    void export_chrome_trace(const std::string& filename);
    void export_json(const std::string& filename);
    
    // 리셋
    void reset();
    
private:
    Profiler();
    
    bool enabled_ = false;
    std::vector<KernelProfile> kernel_profiles_;
    std::map<std::string, ProfileStats> stats_;
    mutable std::mutex mutex_;
};

// 편의 매크로
#define XRT_PROFILE_SCOPE(name) \
    xpuruntime::Profiler::ScopedRange _profile_scope_##__LINE__(name)

}  // namespace xpuruntime
```

---

## 3. Backend 인터페이스

### 3.1 IBackend 추상 클래스

모든 backend가 구현해야 하는 인터페이스.

```cpp
namespace xpuruntime {

struct OpDescriptor {
    std::string op_type;           // "matmul", "conv2d", etc.
    std::vector<std::string> input_dtypes;
    std::vector<std::vector<int64_t>> input_shapes;
    std::map<std::string, std::string> attributes;
};

class IBackend {
public:
    virtual ~IBackend() = default;
    
    // Backend 이름
    virtual std::string name() const = 0;
    
    // 초기화/정리
    virtual void initialize() = 0;
    virtual void finalize() = 0;
    
    // op 지원 여부
    virtual bool supports(const OpDescriptor& op) const = 0;
    
    // 실행
    virtual void execute(const OpDescriptor& op, const KernelContext& ctx) = 0;
};

}  // namespace xpuruntime
```

### 3.2 Backend 등록

```cpp
namespace xpuruntime {

class BackendRegistry {
public:
    static BackendRegistry& instance();
    
    void register_backend(std::unique_ptr<IBackend> backend);
    IBackend* get_backend(const std::string& name);
    std::vector<std::string> get_available_backends() const;
    
private:
    std::unordered_map<std::string, std::unique_ptr<IBackend>> backends_;
};

#define REGISTER_BACKEND(backend_class) \
    static bool _reg_backend_##backend_class = []() { \
        BackendRegistry::instance().register_backend( \
            std::make_unique<backend_class>()); \
        return true; \
    }()

}  // namespace xpuruntime
```

---

## 4. 에러 처리

### 4.1 예외 계층

```cpp
namespace xpuruntime {

// 기본 예외
class XpuRuntimeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// CUDA 에러
class CudaError : public XpuRuntimeError {
public:
    CudaError(cudaError_t error, const std::string& context = "");
    cudaError_t cuda_error() const { return error_; }
private:
    cudaError_t error_;
};

// 메모리 에러
class OutOfMemoryError : public XpuRuntimeError {
public:
    OutOfMemoryError(size_t requested, size_t available);
};

// 지원하지 않는 연산
class UnsupportedOperationError : public XpuRuntimeError {
public:
    using XpuRuntimeError::XpuRuntimeError;
};

// CUDA 에러 체크 매크로
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            throw CudaError(err, #expr); \
        } \
    } while (0)

}  // namespace xpuruntime
```

---

## 5. 스레드 안전성

### 5.1 원칙

- **싱글톤 컴포넌트**: 내부적으로 mutex로 보호
- **Stream 분리**: 각 스레드는 별도 stream 사용 권장
- **Device context**: 스레드별 current device 설정

### 5.2 가이드라인

```cpp
// 권장: 스레드별 stream 사용
void thread_function(int thread_id) {
    auto& sm = StreamManager::instance();
    auto stream = sm.create_stream();  // 스레드별 stream
    
    // ... stream에서 작업 수행 ...
    
    sm.synchronize_stream(stream);
    sm.destroy_stream(stream);
}

// 비권장: 여러 스레드가 같은 stream 공유
// (CUDA는 지원하지만 성능 저하 및 동기화 이슈 가능)
```

---

## 6. 관련 문서

- [01_architecture.md](./01_architecture.md) - 전체 아키텍처
- [03_python_binding.md](./03_python_binding.md) - Python 바인딩
- [06_kernel_policy.md](./06_kernel_policy.md) - 커널 정책 상세
