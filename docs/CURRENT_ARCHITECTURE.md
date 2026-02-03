# 현재 구현 상태 – 시스템 아키텍처 기준 점검

> 설계 문서 [01_architecture.md](plans/01_architecture.md) 의 계층 구조에 맞춰, **지금 코드베이스가 어떤 구조로 되어 있는지** 정리한 문서입니다.

---

## 1. 아키텍처 계층과 디렉터리 매핑

설계상 3계층은 아래와 같고, 각각 **어디에 코드가 있는지** 매핑하면 다음과 같습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│  Python SDK (Control Plane)     →  src/python/xpuruntime/       │
│    training, inference, policies, runtime                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │  pybind11
                              ▼  src/cpp/bindings/*.cpp
┌─────────────────────────────────────────────────────────────────┐
│  C++ Core (Data Plane)          →  src/cpp/core/*.cpp           │
│    DeviceManager, MemoryManager, StreamManager, ...             │
│    헤더: src/cpp/include/xpuruntime/*.h                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│  Backends                       →  src/cpp/backends/ (비어 있음) │
│    cuda_raw, cuBLAS, cuDNN, TensorRT, ONNX Runtime, ...          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 계층별 현재 상태

### 2.1 Python SDK (Control Plane)

**역할**: 사용자가 쓰는 API. 설계상 `training`, `inference`, `policies`, `runtime` 4개 영역.

| 설계 모듈 | 디렉터리 | 현재 상태 | 비고 |
|-----------|----------|-----------|------|
| **진입점** | `src/python/xpuruntime/__init__.py` | ✅ 구현됨 | `__version__`, 예외 클래스(폴백 포함), `get_version()`, `_core`에서 DeviceManager/MemoryManager 등 re-export |
| **runtime** | `src/python/xpuruntime/runtime/` | ⚠️ 스켈레톤 | `__init__.py`만 있음. 설계상 device/stream/memory/profiling 고수준 API 담당 |
| **inference** | `src/python/xpuruntime/inference/` | ⚠️ 스켈레톤 | `__init__.py`만 있음. InferenceSession 등은 미구현 |
| **policies** | `src/python/xpuruntime/policies/` | ⚠️ 스켈레톤 | `__init__.py`만 있음. KernelPolicy/ExecutionPolicy 미구현 |
| **training** | `src/python/xpuruntime/training/` | ⚠️ 스켈레톤 | `__init__.py`만 있음. PyTorch Extension 미구현 |

**정리**: Control Plane의 **구조(패키지 4개 + 진입점)** 는 갖춰져 있고, **실제 로직은 진입점에서 C++ _core 노출만** 하고 있으며, 나머지는 빈 패키지입니다.

---

### 2.2 pybind11 바인딩 (Python ↔ C++ 연결)

**역할**: Python에서 C++ Core를 호출할 수 있게 하는 계층.

| 파일 | 바인딩 대상 | 현재 상태 |
|------|-------------|-----------|
| `src/cpp/bindings/module.cpp` | `_core` 모듈 진입, `get_version` | ✅ |
| `exception_binding.cpp` | XpuRuntimeError, CudaError, OutOfMemoryError, UnsupportedOperationError + 예외 변환 | ✅ |
| `device_binding.cpp` | DeviceInfo, DeviceManager (instance, get_device_count, get_device_info 등) | ✅ |
| `memory_binding.cpp` | MemoryType, MemoryStats, MemoryManager (instance, get_allocated_size, get_stats, empty_cache) | ✅ |

**아직 바인딩 안 된 C++ Core**: StreamManager, KernelRegistry, Dispatcher, Profiler, BackendRegistry.  
→ Python에서 바로 쓰는 API는 Device/Memory/예외/버전만 있는 상태입니다.

---

### 2.3 C++ Core Runtime (Data Plane)

**역할**: 실제 실행·리소스 관리. 설계상 DeviceManager, MemoryManager, StreamManager, KernelRegistry, EngineRegistry, Dispatcher, Profiler.

| 컴포넌트 | 헤더 | 구현(.cpp) | 현재 구현 수준 |
|----------|------|------------|----------------|
| **DeviceManager** | `device_manager.h` | `device_manager.cpp` | ✅ **실제 동작**: cudaGetDeviceCount, cudaGetDeviceProperties로 GPU 열거, DeviceInfo 채움 (fp8/int4 포함) |
| **MemoryManager** | `memory_manager.h` | `memory_manager.cpp` | ⚠️ **최소 동작**: cudaMalloc/cudaFree, cudaMallocHost. 캐싱/풀/통계는 스텁 |
| **StreamManager** | `stream_manager.h` | `stream_manager.cpp` | ⚠️ **부분**: stream/event 생성·동기화는 구현, Graph capture/launch는 스텁(예외) |
| **KernelRegistry** | `kernel_registry.h` | `kernel_registry.cpp` | ⚠️ **스텁**: 등록/조회 구조만 있고, 등록된 커널 없음 |
| **Dispatcher** | `dispatcher.h` | `dispatcher.cpp` | ⚠️ **스텁**: 정책 보관·select_kernel 로직만 있고, 실제 커널 실행 연동 없음 |
| **Profiler** | `profiler.h` | `profiler.cpp` | ⚠️ **스텁**: push_range/pop_range 등 빈 구현, NVTX/CUPTI 미연동 |
| **EngineRegistry** | (설계상) | 없음 | ❌ 헤더/구현 없음. BackendRegistry만 있음 |
| **BackendRegistry** | `backends/ibackend.h` | `backend_registry.cpp` | ⚠️ **스텁**: 등록/조회만 가능, 등록된 백엔드 없음 |

**공통/기반**:
- `common.h`: DeviceId, StreamHandle, EventHandle, TensorInfo
- `exceptions.h` + `exceptions.cpp`: XpuRuntimeError, CudaError, OutOfMemoryError, UnsupportedOperationError
- `config.hpp` + `version.cpp`: 버전 문자열

**정리**: Data Plane은 **인터페이스와 디렉터리 구조는 설계대로** 있고, **실제로 채워진 것은 DeviceManager** 이며, Memory/Stream은 기본 수준, 나머지는 스텁입니다.

---

### 2.4 Backends

**역할**: cuBLAS, cuDNN, TensorRT, ONNX Runtime 등 실제 가속 라이브러리와의 연동.

| 설계 Backend | 현재 상태 |
|--------------|-----------|
| **인터페이스** | `include/xpuruntime/backends/ibackend.h` (IBackend, OpDescriptor, BackendRegistry) ✅ |
| **구현 디렉터리** | `src/cpp/backends/` → `.gitkeep`만 있음 |
| **cuda_raw / cuBLAS / cuDNN / TensorRT / ONNX Runtime 등** | ❌ 미구현 |

**정리**: Backends 계층은 **인터페이스(IBackend)만** 있고, 구현체는 아직 없습니다. TASK_008(ORT), TASK_009(TensorRT) 등에서 채울 예정입니다.

---

## 3. 데이터가 지나가는 경로 (현재)

### 3.1 지금 가능한 흐름

```
[Python]
  import xpuruntime
  dm = xpuruntime.DeviceManager.instance()
  n = dm.get_device_count()        # 또는 get_device_info(0) 등
  mm = xpuruntime.MemoryManager.instance()
  size = mm.get_allocated_size()
       │
       ▼  pybind11 (_core)
[C++ Core]
  DeviceManager::instance() → get_device_count()  (실제 CUDA 호출)
  MemoryManager::instance() → get_allocated_size() (실제 구현)
```

- **Inference/Training 흐름**(Session → EngineRegistry → Dispatcher → Backend)은 아직 없음.
- **커널/엔진 선택**(KernelRegistry, Dispatcher)은 C++에 타입만 있고, Python 바인딩도 없음.

### 3.2 설계 대비 요약

| 구분 | 설계 | 현재 |
|------|------|------|
| Control Plane 구조 | training / inference / policies / runtime | ✅ 패키지 4개 + `__init__` |
| Control Plane 로직 | 고수준 API, Session, Policy | ❌ 대부분 없음, `__init__`에서 _core만 노출 |
| 바인딩 | Core 전부 노출 | Device, Memory, 예외, get_version 만 |
| Data Plane | 7개 컴포넌트 | 7개 타입/클래스 존재, DeviceManager만 본 구현, 나머지 스텁 |
| Backends | 다수 백엔드 | IBackend 인터페이스만, 구현 0개 |

---

## 4. 한눈에 보는 구조 요약

- **Python**: `xpuruntime` 패키지 + `training` / `inference` / `policies` / `runtime` 빈 패키지. 진입점에서 `_core`(C++) 로 Device/Memory/예외/버전만 노출.
- **연결**: pybind11으로 `_core` 모듈 하나만 빌드하고, 예외·Device·Memory만 바인딩.
- **C++ Core**: DeviceManager(실제 GPU 열거), MemoryManager/StreamManager(기본 동작), KernelRegistry/Dispatcher/Profiler/BackendRegistry(스텁). EngineRegistry는 미도입.
- **Backends**: `ibackend.h`로 계약만 있고, `backends/` 아래 구현은 없음.

즉, **시스템 아키텍처(3계층 + pybind11)는 설계대로 잡혀 있고**, **실제 동작하는 부분은 “디바이스 열거 + 메모리/스트림 기본 + 예외/버전”** 까지이며, 나머지는 스켈레톤 또는 미구현입니다.
