# xpuruntime - 전체 아키텍처

> 이 문서는 xpuruntime의 전체 시스템 아키텍처를 설명한다.

---

## 1. 아키텍처 개요

xpuruntime은 **계층화된 아키텍처**를 채택한다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python SDK (Control Plane)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ training │  │inference │  │ policies │  │     runtime      │ │
│  │ (PyTorch │  │(Session) │  │ (Kernel/ │  │ (device/stream/  │ │
│  │Extension)│  │          │  │  Exec)   │  │ memory/profile)  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
└───────┼─────────────┼─────────────┼─────────────────┼───────────┘
        │             │             │                 │
        └─────────────┴─────────────┴─────────────────┘
                              │
                      [ pybind11 Binding ]
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    C++ Core Runtime (Data Plane)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │DeviceManager │  │MemoryManager │  │StreamManager │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │KernelRegistry│  │EngineRegistry│  │  Dispatcher  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐                                               │
│  │   Profiler   │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                         Backends                                │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────────┐ │
│  │cuda_raw│ │ cuBLAS │ │ cuDNN  │ │TensorRT│ │  ONNX Runtime  │ │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────────────┘ │
│  ┌────────┐ ┌────────┐ ┌────────────┐                           │
│  │  NCCL  │ │OpenVINO│ │    QNN     │                           │
│  └────────┘ └────────┘ └────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 계층별 역할

### 2.1 Python SDK (Control Plane)

사용자가 직접 상호작용하는 계층. 모든 API는 Python으로 제공된다.

| 모듈 | 역할 | 의존성 |
|------|------|--------|
| `xpuruntime.pytorch` | PyTorch Extension, 커널 정책 적용 | **선택적** (`pip install xpuruntime[torch]`) |
| `xpuruntime.inference` | InferenceSession, 모델 로드/실행 | 코어 |
| `xpuruntime.policies` | KernelPolicy, ExecutionPolicy 정의 | 코어 |
| `xpuruntime.runtime` | 저수준 device/stream/memory/profiling API | 코어 |

코어 패키지만 설치 시 PyTorch 없이 Inference 전용 사용 가능 (경량).

### 2.2 C++ Core Runtime (Data Plane)

실제 실행이 이루어지는 계층. 모든 성능 크리티컬한 연산을 담당한다.

| 컴포넌트 | 역할 |
|----------|------|
| `DeviceManager` | GPU/NPU 탐지, capability 조회, driver 버전 확인 |
| `MemoryManager` | Pool allocator, caching allocator, pinned memory 관리 |
| `StreamManager` | CUDA stream, event, graph capture 관리 |
| `KernelRegistry` | op → kernel 구현체 매핑 테이블 |
| `EngineRegistry` | model/graph → engine/provider 매핑 |
| `Dispatcher` | 런타임 커널/엔진 선택 로직 |
| `Profiler` | NVTX/CUPTI 기반 프로파일링, 실행 로깅 |

### 2.3 Backends

실제 하드웨어 가속을 수행하는 라이브러리/런타임 래퍼.

| Backend | 용도 |
|---------|------|
| `cuda_raw` | 커스텀 CUDA 커널 |
| `cublas` / `cublasLt` | GEMM 연산 |
| `cudnn` | Convolution, BatchNorm 등 |
| `tensorrt` | TensorRT 엔진 빌드/캐시/실행 |
| `onnxruntime` | ONNX Runtime (CUDA EP) |
| `nccl` | 분산 통신 (Multi-GPU) |
| `openvino` | Intel NPU/CPU/iGPU |
| `qnn` | Qualcomm NPU |

---

## 3. 데이터 흐름

### 3.1 Inference 흐름

```
User Python Code
       │
       ▼
┌──────────────────┐
│ InferenceSession │  ← model.onnx, device, engine, policy
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  EngineRegistry  │  ← 엔진 선택 (TensorRT vs ORT vs ...)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Dispatcher     │  ← KernelPolicy 적용
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Backend       │  ← 실제 실행 (TRT engine / ORT session)
└────────┬─────────┘
         │
         ▼
     Output Tensor
```

### 3.2 Training 흐름

```
PyTorch Model + xpuruntime.pytorch
       │
       ▼
┌──────────────────┐
│   KernelPolicy   │  ← attention="flash_v2", matmul="cublasLt"
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  KernelRegistry  │  ← op → kernel 구현체 조회
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Dispatcher     │  ← shape/dtype/device 기반 최종 선택
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CUDA Backend    │  ← custom kernel / cuBLAS / cuDNN
└────────┬─────────┘
         │
         ▼
  PyTorch Autograd
```

---

## 4. 핵심 설계 원칙

### 4.1 Policy as Code

- 모든 실행 결정(커널/엔진/정밀도)은 **명시적 정책**으로 표현
- 정책은 **직렬화 가능** (JSON/YAML)
- 동일한 정책 → 동일한 실행 결과 (재현성)

```python
policy = KernelPolicy(
    matmul="cublasLt",
    attention="flash_v2",
)

# 정책 저장/로드
policy.save("policy.json")
policy = KernelPolicy.load("policy.json")
```

### 4.2 Zero-Copy 원칙

- Python ↔ C++ 간 데이터 이동 최소화
- PyTorch Tensor와 메모리 공유 (DLPack 또는 직접 포인터)
- 불필요한 GPU ↔ CPU 전송 제거

### 4.3 Explicit over Implicit

- 자동 최적화보다 **명시적 제어**를 우선
- 사용자가 원하면 자동 선택도 가능 (`policy="auto"`)
- 모든 실행 결정은 **로깅/프로파일링**으로 추적 가능

### 4.4 Backend Abstraction

- 각 backend는 **공통 인터페이스** 구현
- 새 backend 추가 시 기존 코드 변경 최소화
- 플러그인 구조로 확장 가능

```cpp
// Backend Interface (추상)
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual void execute(const OpContext& ctx) = 0;
    virtual bool supports(const OpDescriptor& op) = 0;
};
```

---

## 5. 디렉토리 구조

```
xpuruntime/
├── CMakeLists.txt              # 최상위 CMake
├── pyproject.toml              # Python 패키지 설정
├── README.md
├── LICENSE
│
├── docs/                       # 설계 문서
│   ├── 00_overview.md
│   ├── 01_architecture.md      # (본 문서)
│   └── ...
│
├── src/
│   ├── cpp/                    # C++ 코어 런타임
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── xpuruntime/
│   │   │       ├── device_manager.h
│   │   │       ├── memory_manager.h
│   │   │       ├── stream_manager.h
│   │   │       ├── kernel_registry.h
│   │   │       ├── engine_registry.h
│   │   │       ├── dispatcher.h
│   │   │       ├── profiler.h
│   │   │       └── backends/
│   │   │           ├── ibackend.h
│   │   │           ├── cublas_backend.h
│   │   │           └── ...
│   │   ├── core/
│   │   │   ├── device_manager.cpp
│   │   │   ├── memory_manager.cpp
│   │   │   └── ...
│   │   ├── backends/
│   │   │   ├── cublas/
│   │   │   ├── cudnn/
│   │   │   ├── tensorrt/
│   │   │   └── onnxruntime/
│   │   └── bindings/
│   │       └── python_module.cpp
│   │
│   └── python/                 # Python SDK
│       └── xpuruntime/
│           ├── __init__.py
│           ├── inference/
│           │   ├── __init__.py
│           │   └── session.py
│           ├── training/
│           │   ├── __init__.py
│           │   └── pytorch_ext.py
│           ├── policies/
│           │   ├── __init__.py
│           │   ├── kernel_policy.py
│           │   └── execution_policy.py
│           └── runtime/
│               ├── __init__.py
│               ├── device.py
│               ├── stream.py
│               ├── memory.py
│               └── profiler.py
│
├── tests/
│   ├── cpp/
│   │   └── test_device_manager.cpp
│   └── python/
│       └── test_inference_session.py
│
└── examples/
    ├── inference_basic.py
    ├── training_kernel_policy.py
    └── npu_inference.py
```

---

## 6. 의존성

### 필수 의존성 (MVP)

| 라이브러리 | 버전 | 용도 |
|------------|------|------|
| CUDA Toolkit | 11.8+ | CUDA 런타임, NVRTC |
| cuBLAS | (CUDA 포함) | GEMM |
| cuDNN | 8.x+ | CNN 연산 |
| TensorRT | 8.6+ | 추론 엔진 |
| ONNX Runtime | 1.16+ | ONNX 실행 |
| pybind11 | 2.11+ | Python 바인딩 |
| CMake | 3.24+ | 빌드 시스템 |

### 선택적 의존성 (Phase 2+)

| 라이브러리 | 용도 |
|------------|------|
| OpenVINO | Intel NPU |
| QNN SDK | Qualcomm NPU |
| NCCL | Multi-GPU 통신 |

### Python 의존성

| 패키지 | 용도 |
|--------|------|
| numpy | 배열 연산 |
| torch | Training 모듈용 (optional) |
| onnx | 모델 검증 |

---

## 7. 관련 문서

- [00_overview.md](./00_overview.md) - 프로젝트 개요
- [02_cpp_core_runtime.md](./02_cpp_core_runtime.md) - C++ 코어 상세 설계
- [03_python_binding.md](./03_python_binding.md) - Python 바인딩 설계
