# TASK_012: 커널 정책 적용

> Phase 4: Training

---

## 개요

KernelPolicy와 Dispatcher를 완성하여 실제 커널 선택이 정책에 따라 동작하도록 한다.

## 목표

- Dispatcher 선택 로직 완성
- cuBLASLt GEMM 커널 구현
- Flash Attention 통합
- 정책 로깅/프로파일링

## 선행 작업

- TASK_011: PyTorch Extension

## 작업 내용

### 1. Dispatcher 완성

```cpp
// src/cpp/core/dispatcher.cpp

IKernel* Dispatcher::select_kernel(const std::string& op_name,
                                    const KernelContext& ctx) {
    auto& registry = KernelRegistry::instance();
    auto kernels = registry.get_kernels(op_name);
    
    if (kernels.empty()) {
        throw XpuRuntimeError("No kernel for op: " + op_name);
    }
    
    // 1. 정책에 명시된 커널 확인
    auto it = policy_.preferences.find(op_name);
    if (it != policy_.preferences.end()) {
        for (auto* kernel : kernels) {
            if (kernel->name() == it->second && kernel->supports(ctx)) {
                if (logging_enabled_) {
                    log_selection(op_name, kernel->name(), "policy");
                }
                return kernel;
            }
        }
        // 경고: 선호 커널이 지원하지 않음
        log_warning("Preferred kernel not supported, falling back");
    }
    
    // 2. 자동 선택
    for (auto* kernel : kernels) {
        if (kernel->supports(ctx)) {
            if (logging_enabled_) {
                log_selection(op_name, kernel->name(), "auto");
            }
            return kernel;
        }
    }
    
    throw XpuRuntimeError("No supported kernel for: " + op_name);
}
```

### 2. cuBLASLt GEMM

```cpp
// src/cpp/backends/cublas/cublaslt_gemm.cpp

class CublasLtGemmKernel : public IKernel {
public:
    std::string name() const override { return "cublasLt"; }
    
    bool supports(const KernelContext& ctx) const override {
        // SM 70+ 필요 (Tensor Core)
        auto info = DeviceManager::instance().get_device_info(ctx.device_id);
        if (info.compute_capability_major < 7) return false;
        
        // 지원 dtype
        return ctx.dtype == "float32" || 
               ctx.dtype == "float16" || 
               ctx.dtype == "bfloat16";
    }
    
    void execute(const KernelContext& ctx) override {
        cublasLtHandle_t handle;
        cublasLtCreate(&handle);
        
        // matmul 설정
        cublasLtMatmulDesc_t operationDesc;
        cublasLtMatmulDescCreate(&operationDesc, 
                                  CUBLAS_COMPUTE_32F, CUDA_R_32F);
        
        // 실행
        cublasLtMatmul(handle, operationDesc, /* ... */);
        
        cublasLtDestroy(handle);
    }
};

REGISTER_KERNEL("matmul", CublasLtGemmKernel);
```

### 3. Flash Attention 통합

```cpp
// src/cpp/backends/cuda_raw/flash_attention.cu

class FlashAttentionKernel : public IKernel {
public:
    std::string name() const override { return "flash_v2"; }
    
    bool supports(const KernelContext& ctx) const override {
        // SM 80+ 권장
        auto info = DeviceManager::instance().get_device_info(ctx.device_id);
        return info.compute_capability_major >= 8;
    }
    
    void execute(const KernelContext& ctx) override {
        // Flash Attention 커널 호출
    }
};

REGISTER_KERNEL("attention", FlashAttentionKernel);
```

### 4. 정책 로깅

```cpp
void Dispatcher::dispatch(const std::string& op_name, KernelContext& ctx) {
    auto kernel = select_kernel(op_name, ctx);
    
    if (logging_enabled_) {
        DispatchLog log;
        log.op_name = op_name;
        log.selected_kernel = kernel->name();
        log.input_shape = ctx.input_shapes[0];
        log.dtype = ctx.dtype;
        
        auto start = high_resolution_clock::now();
        kernel->execute(ctx);
        auto end = high_resolution_clock::now();
        
        log.elapsed_us = duration<double, std::micro>(end - start).count();
        logs_.push_back(log);
    } else {
        kernel->execute(ctx);
    }
}
```

### 5. Python 로깅 API

```python
def enable_dispatch_logging():
    _core.Dispatcher.instance().enable_logging(True)

def get_dispatch_logs():
    return _core.Dispatcher.instance().get_dispatch_logs()
```

## 완료 조건

- [ ] `KernelPolicy(matmul="cublasLt")` 적용 시 cuBLASLt 사용 확인
- [ ] `KernelPolicy(attention="flash_v2")` 적용 시 Flash Attention 사용
- [ ] 디스패치 로그에서 선택 근거 확인 가능
- [ ] 지원하지 않는 커널 요청 시 적절한 fallback/에러

## 예상 소요 시간

10-14시간

## 관련 문서

- [06_kernel_policy.md](../06_kernel_policy.md)
