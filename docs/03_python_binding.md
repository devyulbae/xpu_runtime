# xpuruntime - Python 바인딩 설계

> 이 문서는 C++ 코어 런타임과 Python SDK 간의 바인딩 설계를 설명한다.

---

## 1. 개요

xpuruntime은 **pybind11**을 사용하여 C++ 코어 런타임을 Python에 노출한다.

### 설계 목표

- **Zero-Copy**: Python ↔ C++ 간 불필요한 데이터 복사 제거
- **Pythonic API**: Python 사용자에게 자연스러운 인터페이스
- **GIL 관리**: 성능 크리티컬 구간에서 GIL 해제
- **예외 변환**: C++ 예외를 Python 예외로 적절히 변환
- **경량 코어**: PyTorch 등 무거운 의존성은 선택적. `xpuruntime.pytorch`는 lazy import.

---

## 2. pybind11 모듈 구조

### 2.1 모듈 계층

```
xpuruntime (Python Package)
├── _core                    # C++ 바인딩 모듈 (pybind11)
│   ├── device               # DeviceManager 바인딩
│   ├── memory               # MemoryManager 바인딩
│   ├── stream               # StreamManager 바인딩
│   ├── kernel               # KernelRegistry 바인딩
│   ├── dispatcher           # Dispatcher 바인딩
│   └── profiler             # Profiler 바인딩
│
├── runtime/                 # Python wrapper
├── inference/               # Python wrapper
├── policies/                # Python wrapper
└── pytorch/                 # PyTorch Extension (선택적, lazy import)
```

### 2.2 빌드 설정

```cmake
# CMakeLists.txt (bindings)
find_package(pybind11 REQUIRED)

pybind11_add_module(_core
    bindings/module.cpp
    bindings/device_binding.cpp
    bindings/memory_binding.cpp
    bindings/stream_binding.cpp
    bindings/kernel_binding.cpp
    bindings/dispatcher_binding.cpp
    bindings/profiler_binding.cpp
)

target_link_libraries(_core PRIVATE
    xpuruntime_core
    ${CUDA_LIBRARIES}
)
```

---

## 3. 주요 바인딩 구현

### 3.1 DeviceManager 바인딩

```cpp
// bindings/device_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "xpuruntime/device_manager.h"

namespace py = pybind11;
using namespace xpuruntime;

void bind_device(py::module_& m) {
    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def_readonly("device_id", &DeviceInfo::device_id)
        .def_readonly("name", &DeviceInfo::name)
        .def_readonly("total_memory", &DeviceInfo::total_memory)
        .def_readonly("free_memory", &DeviceInfo::free_memory)
        .def_readonly("compute_capability_major", &DeviceInfo::compute_capability_major)
        .def_readonly("compute_capability_minor", &DeviceInfo::compute_capability_minor)
        .def_readonly("sm_count", &DeviceInfo::sm_count)
        .def_readonly("supports_fp16", &DeviceInfo::supports_fp16)
        .def_readonly("supports_bf16", &DeviceInfo::supports_bf16)
        .def_readonly("supports_int8", &DeviceInfo::supports_int8)
        .def("__repr__", [](const DeviceInfo& info) {
            return "<DeviceInfo: " + info.name + " (" + 
                   std::to_string(info.total_memory / (1024*1024*1024)) + " GB)>";
        });
    
    py::class_<DeviceManager, std::unique_ptr<DeviceManager, py::nodelete>>(m, "DeviceManager")
        .def_static("instance", &DeviceManager::instance, py::return_value_policy::reference)
        .def("get_device_count", &DeviceManager::get_device_count)
        .def("get_all_devices", &DeviceManager::get_all_devices)
        .def("get_device_info", &DeviceManager::get_device_info)
        .def("get_current_device", &DeviceManager::get_current_device)
        .def("set_current_device", &DeviceManager::set_current_device)
        .def("synchronize", &DeviceManager::synchronize, py::arg("device_id") = -1);
}
```

### 3.2 Tensor 바인딩 (Zero-Copy)

PyTorch Tensor 또는 NumPy array와 메모리를 공유하기 위한 Tensor wrapper.

```cpp
// bindings/tensor_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// DLPack 프로토콜을 통한 zero-copy
struct TensorWrapper {
    void* data_ptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::string dtype;
    int device_id;
    bool owns_data;
    
    // NumPy array에서 생성
    static TensorWrapper from_numpy(py::array arr) {
        TensorWrapper t;
        t.data_ptr = arr.mutable_data();
        t.shape = std::vector<int64_t>(arr.shape(), arr.shape() + arr.ndim());
        t.strides = std::vector<int64_t>(arr.strides(), arr.strides() + arr.ndim());
        t.dtype = py::str(arr.dtype()).cast<std::string>();
        t.device_id = -1;  // CPU
        t.owns_data = false;
        return t;
    }
    
    // PyTorch Tensor에서 생성 (DLPack)
    static TensorWrapper from_dlpack(py::object dlpack_tensor);
    
    // NumPy array로 변환 (CPU only)
    py::array to_numpy() const;
    
    // DLPack capsule로 변환
    py::capsule to_dlpack() const;
};

void bind_tensor(py::module_& m) {
    py::class_<TensorWrapper>(m, "Tensor")
        .def(py::init<>())
        .def_static("from_numpy", &TensorWrapper::from_numpy)
        .def_static("from_dlpack", &TensorWrapper::from_dlpack)
        .def("to_numpy", &TensorWrapper::to_numpy)
        .def("to_dlpack", &TensorWrapper::to_dlpack)
        .def_property_readonly("shape", [](const TensorWrapper& t) { return t.shape; })
        .def_property_readonly("dtype", [](const TensorWrapper& t) { return t.dtype; })
        .def_property_readonly("device_id", [](const TensorWrapper& t) { return t.device_id; })
        .def_property_readonly("data_ptr", [](const TensorWrapper& t) { 
            return reinterpret_cast<uintptr_t>(t.data_ptr); 
        });
}
```

### 3.3 KernelPolicy 바인딩

```cpp
// bindings/policy_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "xpuruntime/dispatcher.h"

namespace py = pybind11;
using namespace xpuruntime;

void bind_policy(py::module_& m) {
    py::enum_<KernelPolicy::AutoSelectStrategy>(m, "AutoSelectStrategy")
        .value("FirstSupported", KernelPolicy::AutoSelectStrategy::FirstSupported)
        .value("BestPerformance", KernelPolicy::AutoSelectStrategy::BestPerformance);
    
    py::class_<KernelPolicy>(m, "KernelPolicy")
        .def(py::init<>())
        .def(py::init([](py::kwargs kwargs) {
            KernelPolicy policy;
            for (auto& item : kwargs) {
                std::string key = py::str(item.first).cast<std::string>();
                std::string value = py::str(item.second).cast<std::string>();
                policy.preferences[key] = value;
            }
            return policy;
        }))
        .def_readwrite("preferences", &KernelPolicy::preferences)
        .def_readwrite("auto_strategy", &KernelPolicy::auto_strategy)
        .def("to_json", &KernelPolicy::to_json)
        .def_static("from_json", &KernelPolicy::from_json)
        .def("save", [](const KernelPolicy& p, const std::string& path) {
            std::ofstream f(path);
            f << p.to_json();
        })
        .def_static("load", [](const std::string& path) {
            std::ifstream f(path);
            std::string json((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
            return KernelPolicy::from_json(json);
        })
        .def("__repr__", [](const KernelPolicy& p) {
            std::string s = "<KernelPolicy: {";
            for (auto& [k, v] : p.preferences) {
                s += k + "=" + v + ", ";
            }
            s += "}>";
            return s;
        });
}
```

---

## 4. GIL 관리

### 4.1 원칙

- **Python 객체 접근 시**: GIL 보유 필요
- **순수 C++/CUDA 연산 시**: GIL 해제로 병렬성 확보
- **콜백 호출 시**: GIL 재획득 필요

### 4.2 구현 패턴

```cpp
// GIL 해제 예시
void execute_kernel_binding(py::object input, py::object output) {
    // 1. Python 객체에서 포인터 추출 (GIL 보유)
    void* input_ptr = get_data_ptr(input);
    void* output_ptr = get_data_ptr(output);
    
    // 2. GIL 해제 후 연산 수행
    {
        py::gil_scoped_release release;
        
        // C++/CUDA 연산 (GIL 없이 실행)
        execute_kernel_impl(input_ptr, output_ptr);
        
        // CUDA 동기화
        cudaDeviceSynchronize();
    }
    // 3. GIL 자동 재획득
}

// 콜백 호출 시 GIL 획득
void with_profiling_callback(std::function<void()> kernel_fn, py::function callback) {
    // 커널 실행 (GIL 없이)
    {
        py::gil_scoped_release release;
        kernel_fn();
    }
    
    // 콜백 호출 (GIL 필요)
    callback();
}
```

---

## 5. 예외 변환

### 5.1 C++ → Python 예외 매핑

```cpp
// bindings/exception_binding.cpp
#include <pybind11/pybind11.h>
#include "xpuruntime/exceptions.h"

namespace py = pybind11;
using namespace xpuruntime;

void bind_exceptions(py::module_& m) {
    // 기본 예외
    static py::exception<XpuRuntimeError> exc_runtime(m, "XpuRuntimeError");
    
    // CUDA 에러
    static py::exception<CudaError> exc_cuda(m, "CudaError", exc_runtime.ptr());
    
    // 메모리 에러
    static py::exception<OutOfMemoryError> exc_oom(m, "OutOfMemoryError", exc_runtime.ptr());
    
    // 지원하지 않는 연산
    static py::exception<UnsupportedOperationError> exc_unsupported(
        m, "UnsupportedOperationError", exc_runtime.ptr());
    
    // 예외 변환 등록
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const OutOfMemoryError& e) {
            exc_oom(e.what());
        } catch (const CudaError& e) {
            exc_cuda(e.what());
        } catch (const UnsupportedOperationError& e) {
            exc_unsupported(e.what());
        } catch (const XpuRuntimeError& e) {
            exc_runtime(e.what());
        }
    });
}
```

### 5.2 Python 사용 예시

```python
import xpuruntime as xrt

try:
    sess = xrt.InferenceSession("model.onnx", device="cuda:99")
except xrt.CudaError as e:
    print(f"CUDA 에러: {e}")
except xrt.OutOfMemoryError as e:
    print(f"메모리 부족: {e}")
except xrt.XpuRuntimeError as e:
    print(f"런타임 에러: {e}")
```

---

## 6. Python SDK 래퍼

### 6.1 패키지 구조

```
src/python/xpuruntime/
├── __init__.py              # 메인 진입점
├── _core.cpython-*.so       # pybind11 모듈 (빌드 결과)
│
├── runtime/
│   ├── __init__.py
│   ├── device.py            # Device 관련 고수준 API
│   ├── stream.py            # Stream 관련 고수준 API
│   ├── memory.py            # Memory 관련 고수준 API
│   └── profiler.py          # Profiler 래퍼
│
├── inference/
│   ├── __init__.py
│   └── session.py           # InferenceSession 클래스
│
├── training/
│   ├── __init__.py
│   └── pytorch_ext.py       # PyTorch Extension
│
└── policies/
    ├── __init__.py
    ├── kernel_policy.py     # KernelPolicy 래퍼
    └── execution_policy.py  # ExecutionPolicy 래퍼
```

### 6.2 Device 래퍼 예시

```python
# src/python/xpuruntime/runtime/device.py
from typing import List, Optional
from .. import _core

class Device:
    """GPU/NPU 디바이스 정보 및 관리"""
    
    def __init__(self, device_id: int = 0):
        self._id = device_id
        self._info = _core.DeviceManager.instance().get_device_info(device_id)
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def name(self) -> str:
        return self._info.name
    
    @property
    def total_memory(self) -> int:
        """바이트 단위 총 메모리"""
        return self._info.total_memory
    
    @property
    def total_memory_gb(self) -> float:
        """GB 단위 총 메모리"""
        return self._info.total_memory / (1024 ** 3)
    
    @property
    def compute_capability(self) -> tuple:
        """(major, minor) 형태의 compute capability"""
        return (self._info.compute_capability_major, 
                self._info.compute_capability_minor)
    
    @property
    def supports_fp16(self) -> bool:
        return self._info.supports_fp16
    
    def synchronize(self):
        """이 디바이스의 모든 작업 완료 대기"""
        _core.DeviceManager.instance().synchronize(self._id)
    
    def __repr__(self) -> str:
        return f"<Device {self._id}: {self.name} ({self.total_memory_gb:.1f} GB)>"


def get_device_count() -> int:
    """사용 가능한 디바이스 수 반환"""
    return _core.DeviceManager.instance().get_device_count()


def get_all_devices() -> List[Device]:
    """모든 디바이스 목록 반환"""
    count = get_device_count()
    return [Device(i) for i in range(count)]


def get_current_device() -> Device:
    """현재 활성 디바이스 반환"""
    device_id = _core.DeviceManager.instance().get_current_device()
    return Device(device_id)


def set_current_device(device: int | Device):
    """현재 활성 디바이스 설정"""
    device_id = device if isinstance(device, int) else device.id
    _core.DeviceManager.instance().set_current_device(device_id)


def synchronize(device_id: Optional[int] = None):
    """디바이스 동기화"""
    _core.DeviceManager.instance().synchronize(device_id if device_id else -1)
```

### 6.3 KernelPolicy 래퍼 예시

```python
# src/python/xpuruntime/policies/kernel_policy.py
from typing import Dict, Optional
from pathlib import Path
import json
from .. import _core

class KernelPolicy:
    """커널 선택 정책을 정의하는 클래스"""
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: op_name=kernel_name 형태의 커널 매핑
                      예: matmul="cublasLt", attention="flash_v2"
        """
        self._policy = _core.KernelPolicy()
        for op_name, kernel_name in kwargs.items():
            self._policy.preferences[op_name] = kernel_name
    
    def set(self, op_name: str, kernel_name: str) -> "KernelPolicy":
        """커널 매핑 추가/수정 (체이닝 지원)"""
        self._policy.preferences[op_name] = kernel_name
        return self
    
    def get(self, op_name: str) -> Optional[str]:
        """특정 op의 커널 매핑 조회"""
        return self._policy.preferences.get(op_name)
    
    def remove(self, op_name: str) -> "KernelPolicy":
        """커널 매핑 제거"""
        if op_name in self._policy.preferences:
            del self._policy.preferences[op_name]
        return self
    
    @property
    def preferences(self) -> Dict[str, str]:
        """모든 커널 매핑 반환"""
        return dict(self._policy.preferences)
    
    def save(self, path: str | Path):
        """정책을 JSON 파일로 저장"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump({
                'preferences': self.preferences,
                'auto_strategy': str(self._policy.auto_strategy),
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "KernelPolicy":
        """JSON 파일에서 정책 로드"""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        policy = cls(**data.get('preferences', {}))
        return policy
    
    def _get_internal(self) -> _core.KernelPolicy:
        """내부 C++ 객체 반환 (내부용)"""
        return self._policy
    
    def __repr__(self) -> str:
        prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
        return f"KernelPolicy({prefs})"
```

---

## 7. 선택적 import (경량 패키지)

PyTorch 등 무거운 의존성은 **선택적**으로 로드한다. 코어만 설치한 환경에서는 `xpuruntime.pytorch` 접근 시에만 에러를 내거나 안내한다.

### 7.1 pytorch 서브모듈 lazy import

```python
# src/python/xpuruntime/__init__.py

# pytorch는 import 시점에 로드하지 않음
def __getattr__(name: str):
    if name == "pytorch":
        try:
            from . import pytorch
            return pytorch
        except ImportError as e:
            raise ImportError(
                "xpuruntime.pytorch requires PyTorch. "
                "Install with: pip install xpuruntime[torch]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### 7.2 명시적 사용

```python
# Inference만 사용 (PyTorch 불필요)
import xpuruntime as xrt
sess = xrt.InferenceSession("model.onnx", engine="onnxrt")

# Training 사용 시에만 pytorch 로드
import xpuruntime.pytorch as xrt_torch  # 여기서 torch 없으면 ImportError
xrt_torch.set_kernel_policy(matmul="cublasLt")
```

---

## 8. 메인 패키지 초기화

```python
# src/python/xpuruntime/__init__.py
"""
xpuruntime - Unified GPU/NPU Execution Runtime
"""

__version__ = "0.1.0"

# C++ 바인딩 모듈 (항상 로드)
from . import _core

# 예외
from ._core import (
    XpuRuntimeError,
    CudaError,
    OutOfMemoryError,
    UnsupportedOperationError,
)

# 런타임
from .runtime.device import (
    Device,
    get_device_count,
    get_all_devices,
    get_current_device,
    set_current_device,
    synchronize,
)

# 정책
from .policies.kernel_policy import KernelPolicy
from .policies.execution_policy import ExecutionPolicy

# 추론
from .inference.session import InferenceSession

# pytorch는 __all__에 포함하지 않음 (lazy import via __getattr__)
# 편의 별칭
__all__ = [
    # Version
    "__version__",
    
    # Exceptions
    "XpuRuntimeError",
    "CudaError", 
    "OutOfMemoryError",
    "UnsupportedOperationError",
    
    # Device
    "Device",
    "get_device_count",
    "get_all_devices",
    "get_current_device",
    "set_current_device",
    "synchronize",
    
    # Policies
    "KernelPolicy",
    "ExecutionPolicy",
    
    # Inference
    "InferenceSession",
]
```

---

## 9. PyTorch 통합 (선택적)

### 9.1 별도 서브패키지

```python
# src/python/xpuruntime/pytorch/__init__.py
"""
xpuruntime.pytorch - PyTorch Extension for xpuruntime
Requires: pip install xpuruntime[torch]
"""

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for xpuruntime.pytorch. "
        "Install with: pip install xpuruntime[torch]"
    )

from .extension import (
    set_kernel_policy,
    get_kernel_policy,
    autocast,
    capture_graph,
)

__all__ = [
    "set_kernel_policy",
    "get_kernel_policy", 
    "autocast",
    "capture_graph",
]
```

---

## 10. 관련 문서

- [02_cpp_core_runtime.md](./02_cpp_core_runtime.md) - C++ 코어 런타임
- [04_inference_module.md](./04_inference_module.md) - 추론 모듈
- [05_training_module.md](./05_training_module.md) - 학습 모듈
