# GPU/NPU Execution Runtime Library – Design Specification (Draft)

> 목표: **연구(학습) ↔ 서비스(추론/호스팅)** 전 구간에서, 실행 엔진/커널 선택을 **명시적·재현 가능**하게 만들고
> GPU/NPU/CPU 등 다양한 디바이스를 **동일한 Python API**로 사용할 수 있는 오픈소스 런타임을 구축한다.

---

## 1. 프로젝트 목적 (Why)

본 프로젝트의 목표는 **모델 프레임워크를 새로 만드는 것**이 아니라,  
**GPU/NPU 실행을 통제(Control)하는 런타임 라이브러리**를 만드는 것이다.

- 사용자는 **Python**으로 사용한다.
- 실제 실행, 연산, 메모리, 스트림, 커널/엔진 선택은 **전부 C++(및 CUDA 등)** 에서 수행한다.
- **연구자(Training)**와 **MLOps/ML Engineer(Inference & Hosting)** 모두에게 실질적인 가치를 제공한다.

> 핵심 메시지  
> **“연구에서 검증한 실행 전략(커널/정밀도/메모리 정책)을 그대로 서비스까지 가져간다.”**

---

## 2. 타깃 사용자

### 2.1 연구자 (Research / Training)
- PyTorch 등 기존 학습 프레임워크를 유지하면서 GPU 실행을 **명시적으로 통제**하고 싶음
- 커널/정밀도/메모리 전략을 실험 대상으로 삼고 싶음
- 실험 결과의 **재현성**과 **설명 가능성**이 중요

### 2.2 MLOps / ML Engineer (Inference / Hosting)
- 학습(PyTorch)과 서비스(TensorRT / ONNX Runtime / 서버)의 단절 문제 해결 필요
- TensorRT/ORT의 블랙박스 최적화에 대한 통제/가시성 필요
- 성능 튜닝을 **코드/정책**으로 관리하고, CI/CD에 포함 가능한 파이프라인 필요
- 서버(GPU)뿐 아니라 클라이언트(노트북/PC/엣지)의 **NPU 추론**까지 고려

---

## 3. 해결하려는 핵심 문제

### 기존 생태계의 한계
1. 커널/엔진 선택을 사용자가 제어하기 어려움(대부분 내부 heuristic/black box)
2. 학습과 추론 최적화 전략이 단절되어 동일한 “빠름”이 유지되지 않음
3. 프레임워크 변경 시 성능 튜닝 자산(노하우, 설정)이 소실
4. GPU/NPU 성능 결과에 대한 설명(어떤 커널/엔진/정밀도/동기화/메모리 이동)이 부족
5. TensorRT / ONNX / CUDA / (NPU 런타임)을 혼합해서 쓰기 어렵고 운영이 복잡

### 본 라이브러리의 해결 방향
- GPU/NPU 실행 정책을 **명시적·코드화**(Policy as Code)
- 학습과 추론이 **동일한 Runtime/Policy 개념**을 공유
- 프레임워크 독립적 최적화 자산 축적
- 실행 선택/성능 근거를 로깅/프로파일링으로 제공(설명 가능)
- 여러 엔진/커널을 통합하는 **Orchestrator/Control Plane**

---

## 4. 기본 철학 및 원칙

### 4.1 언어 및 계층 분리
- **Python**: 사용자 UX, API, 파이프라인 정의(컨트롤 플레인)
- **C++**: 실행 런타임(메모리, 스트림, 커널, 엔진, 디바이스 관리)
- **CUDA 등**: 성능 상한 결정 계층(엔진/커널 내부)

> Python = Control Plane  
> C++/CUDA = Data Plane

### 4.2 지원 범위(단계적)
- 1차: **NVIDIA CUDA 중심**(Training + Inference 핵심)
- 2차: **NPU(Inference 중심)** 추가
  - Intel NPU: OpenVINO 기반
  - Qualcomm NPU: QNN 기반
- 장기: 멀티 백엔드 확장(AMD/Metal 등) — 플러그인 구조로

---

## 5. 전체 아키텍처 개요

### 5.1 레이어 구조

```text
[ Python SDK ]
 ├─ training/
 │    └─ PyTorch extension (커널 정책, 커스텀 op, graph capture)
 ├─ inference/
 │    └─ InferenceSession (ONNX / TensorRT / NPU)
 ├─ policies/
 │    ├─ KernelPolicy (GPU op 단위)
 │    └─ ExecutionPolicy (NPU/엔진 단위)
 ├─ kernels/
 │    └─ 커스텀 커널 등록 API (선택)
 └─ runtime/
      └─ device, stream, memory, profiling API

[ C++ Core Runtime ]
 ├─ DeviceManager      (GPU/NPU 탐지, capability, driver/SDK)
 ├─ MemoryManager      (pool, caching allocator, pinned memory)
 ├─ StreamManager      (CUDA stream, event, graph capture)
 ├─ KernelRegistry     (op → kernel 구현체 매핑)
 ├─ EngineRegistry     (model/graph → engine/provider 매핑)
 ├─ Dispatcher         (실행 시점 커널/엔진 선택)
 ├─ Profiler           (NVTX/CUPTI + backend별 profiling hook)
 └─ Backends/
      ├─ cuda_raw          (custom CUDA kernels)
      ├─ cublas / cublasLt (GEMM)
      ├─ cudnn             (conv 등)
      ├─ tensorrt          (TRT 엔진 빌드/캐시/런)
      ├─ onnxruntime       (ORT CUDA EP + graph partition)
      ├─ nccl              (분산 통신)
      ├─ openvino          (Intel NPU / CPU / iGPU)
      ├─ qnn               (Qualcomm NPU)
      └─ cpu_fallback      (디버깅/비가속 환경)
```

---

## 6. Training (학습) 지원 전략

### 6.1 학습 프레임워크를 대체하지 않는다
- PyTorch/TF 자체를 새로 만들지 않음(Autograd/Optimizer 재구현 지양)
- 대신 **PyTorch Extension(커스텀 op + 런타임 최적화)**로 학습 가속과 제어를 제공

### 6.2 Training 제공 기능(우선순위)
- 커널 선택 정책(KernelPolicy)
- 커스텀 CUDA op(예: attention, layernorm, fused ops)
- allocator/stream 통합 및 불필요 sync 최소화
- CUDA Graph capture(런치 오버헤드 절감)
- NCCL helper(선택)

### 6.3 사용자 예시(Training)

```python
import gpukit.pytorch as gkt

gkt.set_kernel_policy(
    attention="flash_v2",
    matmul="cublasLt",
)

with gkt.autocast(fp16=True), gkt.capture_graph():
    y = model(x)
    loss = criterion(y, t)
    loss.backward()
```

> 주의: NPU는 일반적으로 **학습 대상이 아니라 추론 대상**이므로 Training은 GPU 중심으로 설계한다.

---

## 7. Inference (추론) 지원 전략

### 7.1 모델 진입 경로(공통)
- PyTorch/TF 모델을 **ONNX로 Export**(권장)
- ONNX를 기준 IR로 사용
- InferenceSession에서 실행(엔진/디바이스 선택)

### 7.2 지원 엔진/런타임
- GPU: TensorRT / ONNX Runtime(CUDA EP) / custom CUDA
- NPU:
  - Intel NPU: OpenVINO (ONNX/IR 경로)
  - Qualcomm NPU: QNN (ONNX → QNN IR 변환/컴파일)

### 7.3 사용자 예시(Inference)

```python
import gpukit as gk

policy = gk.KernelPolicy(
    matmul="cublasLt",
    attention="flash_v2",
)

sess = gk.InferenceSession(
    model="model.onnx",
    device="cuda:0",
    engine="tensorrt",   # or "onnxrt"
    kernel_policy=policy,
    fp16=True,
)

out = sess.run({"input": x})
```

---

## 8. KernelPolicy & Dispatcher (GPU 핵심 차별점)

### 8.1 KernelPolicy
- 연산(op) 단위로 커널/구현체 선택을 명시
- 실험/운영에서 재현 가능한 성능 튜닝(Policy as Code)

```python
KernelPolicy(
    matmul="cublasLt",
    conv="cudnn",
    attention="flash_v2",
)
```

### 8.2 Dispatcher
- 실행 시점에 다음 정보를 종합해 최종 구현체를 선택:
  - 입력 shape / dtype
  - device capability(SM 등)
  - 사용자 policy(강제/선호)
  - workspace 제한 등

---

## 9. ExecutionPolicy (NPU/엔진 중심)

NPU는 GPU처럼 “커널 단위 선택”이 아니라 **그래프 컴파일/정밀도/shape 제약**이 핵심이다.  
따라서 NPU 쪽은 KernelPolicy 대신 ExecutionPolicy 중심으로 설계한다.

예시:

```python
ExecutionPolicy(
    target="npu",
    precision="int8",
    static_shape=True,
    allow_fallback="cpu",
)
```

---

## 10. Intel NPU 지원 (OpenVINO Backend)

### 10.1 개요
- Intel NPU는 OpenVINO 생태계를 통해 접근하는 것이 현실적/성숙함
- ONNX 기반 추론 워크플로우와 결합이 좋음

### 10.2 설계 포인트
- backend: `openvino`
- device: `npu`(또는 `AUTO`/`CPU`/`GPU` 선택)
- quantization/int8 정책 적용 가능

### 10.3 사용자 예시

```python
sess = gk.InferenceSession(
    "model.onnx",
    device="npu",
    backend="openvino",
    exec_policy=gk.ExecutionPolicy(precision="int8", static_shape=True),
)
```

---

## 11. Qualcomm NPU 지원 (QNN Backend)

### 11.1 개요
- Snapdragon 계열 NPU는 QNN SDK 기반
- ONNX → QNN IR 변환 및 컴파일 단계 필요
- Inference 중심(Training 비대상)

### 11.2 설계 포인트
- backend: `qnn`
- device: `npu`
- 모델/입력 shape 고정이 강하게 요구될 수 있음
- fallback(CPU/GPU)을 정책으로 제어

### 11.3 사용자 예시

```python
sess = gk.InferenceSession(
    "model.onnx",
    device="npu",
    backend="qnn",
    exec_policy=gk.ExecutionPolicy(precision="int8", static_shape=True, allow_fallback="cpu"),
)
```

---

## 12. TensorRT / ONNX Runtime 관계

- TensorRT: 최고 성능(특히 NVIDIA), 다만 내부 tactic이 블랙박스 성격
- ONNX Runtime: 범용성과 provider 확장성
- 본 라이브러리: 두 엔진을
  - 선택적으로 사용하거나
  - 그래프/연산 단위로 혼합(partition)
하는 **Orchestrator** 역할을 수행

---

## 13. 차별화 포인트 요약

- 커널/엔진 선택을 **사용자가 통제**(정책으로 고정/선호/자동 선택)
- 학습–추론을 관통하는 **일관된 정책 모델**
- 프레임워크 독립적인 성능 튜닝 자산(Policy as Code)
- 실행 근거(선택된 커널/엔진, sync/메모리 이동)를 기록 → 설명 가능
- GPU + NPU를 동일한 Python API로 통합(서버/클라이언트/엣지 모두)

> 우리는 “모델 라이브러리”가 아니라  
> **AI Accelerator Execution Control Plane**이다.

---

## 14. MVP 범위(현실적 1차 목표)

1. NVIDIA CUDA only
2. InferenceSession: ONNX Runtime(CUDA EP) + TensorRT
3. KernelPolicy: matmul(cuBLAS / cuBLASLt)부터
4. MemoryPool + Stream 관리 + 프로파일링 기본
5. Python 바인딩(pybind11) + C++ 런타임 스켈레톤

### 14.1 NPU는 2차(MVP 이후)
- Intel NPU(OpenVINO) → 비교적 빠르게 추가 가능
- Qualcomm NPU(QNN) → SDK/배포/변환 파이프라인 포함하여 단계적으로

---

## 15. 구현 우선순위 제안

1. C++ Core Runtime skeleton(Device/Memory/Stream/Registry/Dispatcher)
2. Python ↔ C++ 바인딩(pybind11) 및 최소 Tensor abstraction
3. InferenceSession: ORT 실행 → TRT 실행(엔진 캐시 포함)
4. KernelRegistry/Dispatcher를 통한 커널 정책 적용
5. PyTorch Extension: 커스텀 op + 정책 적용(Training-A)

---

## 16. README 첫 줄 후보

> **A unified GPU/NPU execution runtime that bridges research and production by making engine and kernel selection explicit, reproducible, and framework-agnostic.**
