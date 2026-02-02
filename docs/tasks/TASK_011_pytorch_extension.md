# TASK_011: PyTorch Extension

> Phase 4: Training

---

## 개요

PyTorch와 통합되는 xpuruntime.pytorch 모듈을 구현한다.  
**선택적 의존성**: `pip install xpuruntime[torch]` 시에만 설치/로드. 코어 패키지는 PyTorch 없이 가벼운 Inference 전용으로 사용 가능.

## 목표

- PyTorch dispatcher 통합
- 커널 정책 적용 메커니즘
- 커스텀 연산자 등록
- autocast/CUDA Graph 래퍼

## 선행 작업

- TASK_010: InferenceSession 구현

## 작업 내용

### 1. PyTorch Extension 빌드

```cmake
# src/cpp/bindings/CMakeLists.txt

find_package(Torch REQUIRED)

add_library(xpuruntime_torch SHARED
    pytorch/torch_extension.cpp
    pytorch/dispatcher_hook.cpp
)

target_link_libraries(xpuruntime_torch PRIVATE
    xpuruntime_core
    ${TORCH_LIBRARIES}
)
```

### 2. Dispatcher Hook

```cpp
// src/cpp/bindings/pytorch/dispatcher_hook.cpp

#include <torch/extension.h>
#include "xpuruntime/dispatcher.h"

// matmul 오버라이드
torch::Tensor xrt_matmul(const torch::Tensor& a, const torch::Tensor& b) {
    auto& dispatcher = xpuruntime::Dispatcher::instance();
    auto policy = dispatcher.get_policy();
    
    auto it = policy.preferences.find("matmul");
    if (it != policy.preferences.end()) {
        if (it->second == "cublasLt") {
            return cublasLt_matmul_impl(a, b);
        }
    }
    
    // 기본 PyTorch 구현
    return at::matmul(a, b);
}

// PyTorch dispatcher에 등록
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("matmul", xrt_matmul);
}
```

### 3. Python API

```python
# src/python/xpuruntime/pytorch/__init__.py

def set_kernel_policy(policy: KernelPolicy = None, **kwargs):
    """학습 중 커널 정책 설정"""
    global _current_policy
    
    if policy:
        _current_policy = policy
    elif kwargs:
        _current_policy = KernelPolicy(**kwargs)
    
    # C++ dispatcher에 적용
    _core.Dispatcher.instance().set_policy(_current_policy._get_internal())

def get_kernel_policy() -> Optional[KernelPolicy]:
    return _current_policy
```

### 4. Autocast 래퍼

```python
@contextmanager
def autocast(dtype=torch.float16, enabled=True):
    """혼합 정밀도 컨텍스트"""
    with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
        yield
```

### 5. CUDA Graph 래퍼

```python
@contextmanager
def capture_graph():
    """CUDA Graph 캡처 컨텍스트"""
    _core.Profiler.instance().push_range("cuda_graph_capture")
    try:
        yield
    finally:
        _core.Profiler.instance().pop_range()
```

## 완료 조건

- [ ] `import xpuruntime.pytorch` 성공
- [ ] `set_kernel_policy(matmul="cublasLt")` 적용 확인
- [ ] 학습 루프에서 정책 적용 확인
- [ ] autocast 동작
- [ ] PyTorch 버전 호환성 (2.0+)

## 예상 소요 시간

12-16시간

## 관련 문서

- [05_training_module.md](../05_training_module.md)
