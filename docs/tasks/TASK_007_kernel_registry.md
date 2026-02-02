# TASK_007: KernelRegistry 구현

> Phase 2: Core Features

---

## 개요

연산(op)과 커널 구현체를 매핑하는 레지스트리를 구현한다.

## 목표

- IKernel 인터페이스 정의
- KernelRegistry 싱글톤 구현
- 동적 커널 등록 매크로
- 기본 cuBLAS 커널 등록

## 선행 작업

- TASK_006: StreamManager 구현

## 작업 내용

### 1. IKernel 인터페이스

```cpp
// src/cpp/include/xpuruntime/kernel_registry.h

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
```

### 2. KernelRegistry

```cpp
// src/cpp/core/kernel_registry.cpp

class KernelRegistry {
public:
    static KernelRegistry& instance() {
        static KernelRegistry registry;
        return registry;
    }
    
    void register_kernel(const std::string& op_name, 
                         std::unique_ptr<IKernel> kernel) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        registry_[op_name].push_back(std::move(kernel));
    }
    
    std::vector<IKernel*> get_kernels(const std::string& op_name) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = registry_.find(op_name);
        if (it == registry_.end()) return {};
        
        std::vector<IKernel*> result;
        for (const auto& k : it->second) {
            result.push_back(k.get());
        }
        return result;
    }
    
private:
    std::unordered_map<std::string, 
                       std::vector<std::unique_ptr<IKernel>>> registry_;
    mutable std::shared_mutex mutex_;
};
```

### 3. 등록 매크로

```cpp
#define REGISTER_KERNEL(op_name, kernel_class) \
    static bool _reg_##kernel_class = []() { \
        KernelRegistry::instance().register_kernel( \
            op_name, std::make_unique<kernel_class>()); \
        return true; \
    }()
```

### 4. 기본 cuBLAS GEMM 커널

```cpp
// src/cpp/backends/cublas/cublas_gemm.cpp

class CublasGemmKernel : public IKernel {
public:
    std::string name() const override { return "cublas"; }
    
    bool supports(const KernelContext& ctx) const override {
        return ctx.dtype == "float32" || ctx.dtype == "float16";
    }
    
    void execute(const KernelContext& ctx) override {
        // cuBLAS GEMM 호출
    }
};

REGISTER_KERNEL("matmul", CublasGemmKernel);
```

## 완료 조건

- [ ] 커널 등록 및 조회 동작
- [ ] supports() 필터링 동작
- [ ] cuBLAS GEMM 커널 실행 가능
- [ ] 스레드 안전성 확인

## 예상 소요 시간

6-8시간

## 관련 문서

- [02_cpp_core_runtime.md](../02_cpp_core_runtime.md)
- [06_kernel_policy.md](../06_kernel_policy.md)
