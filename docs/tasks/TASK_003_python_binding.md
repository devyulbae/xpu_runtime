# TASK_003: Python 바인딩 기본

> Phase 1: Foundation

---

## 개요

pybind11을 사용하여 C++ 코어와 Python을 연결하는 기본 바인딩을 구현한다.

## 목표

- pybind11 모듈 설정
- 기본 타입 바인딩 (DeviceInfo 등)
- 예외 변환 설정
- Python 패키지 구조 구성

## 선행 작업

- TASK_002: C++ 코어 스켈레톤

## 작업 내용

### 1. pybind11 모듈 파일

```cpp
// src/cpp/bindings/module.cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_device(py::module_& m);
void bind_memory(py::module_& m);
void bind_exceptions(py::module_& m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "xpuruntime core module";
    
    bind_exceptions(m);
    bind_device(m);
    bind_memory(m);
}
```

### 2. 예외 바인딩

```cpp
// bindings/exception_binding.cpp
void bind_exceptions(py::module_& m) {
    static py::exception<XpuRuntimeError> exc_runtime(m, "XpuRuntimeError");
    static py::exception<CudaError> exc_cuda(m, "CudaError", exc_runtime.ptr());
    // ...
}
```

### 3. DeviceInfo 바인딩

```cpp
// bindings/device_binding.cpp
void bind_device(py::module_& m) {
    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def_readonly("device_id", &DeviceInfo::device_id)
        .def_readonly("name", &DeviceInfo::name)
        // ...
}
```

### 4. Python 패키지 구조

```python
# src/python/xpuruntime/__init__.py
from . import _core

__version__ = "0.1.0"

# re-export
from ._core import XpuRuntimeError, CudaError
```

### 5. CMake 바인딩 빌드

```cmake
pybind11_add_module(_core
    bindings/module.cpp
    bindings/exception_binding.cpp
    bindings/device_binding.cpp
)
```

## 완료 조건

- [ ] `import xpuruntime` 성공
- [ ] `xpuruntime.__version__` 확인 가능
- [ ] `xpuruntime.XpuRuntimeError` 접근 가능
- [ ] 기본 Python 테스트 통과

## 예상 소요 시간

4-6시간

## 관련 문서

- [03_python_binding.md](../plans/03_python_binding.md)
