# xpuruntime - 프로젝트 개요

> **A unified GPU/NPU execution runtime that bridges research and production by making engine and kernel selection explicit, reproducible, and framework-agnostic.**

---

## 1. 프로젝트 비전

**xpuruntime**은 모델 프레임워크를 새로 만드는 것이 아니라,  
**GPU/NPU 실행을 통제(Control)하는 런타임 라이브러리**이다.

### 핵심 메시지

> **"연구에서 검증한 실행 전략(커널/정밀도/메모리 정책)을 그대로 서비스까지 가져간다."**

### 언어 및 계층 분리

| 계층 | 역할 | 언어 |
|------|------|------|
| Control Plane | 사용자 UX, API, 파이프라인 정의 | Python |
| Data Plane | 실행 런타임 (메모리, 스트림, 커널, 엔진) | C++ |
| Performance Layer | 성능 상한 결정 (엔진/커널 내부) | CUDA 등 |

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

1. 커널/엔진 선택을 사용자가 제어하기 어려움 (대부분 내부 heuristic/black box)
2. 학습과 추론 최적화 전략이 단절되어 동일한 "빠름"이 유지되지 않음
3. 프레임워크 변경 시 성능 튜닝 자산(노하우, 설정)이 소실
4. GPU/NPU 성능 결과에 대한 설명 부족 (어떤 커널/엔진/정밀도/동기화/메모리 이동)
5. TensorRT / ONNX / CUDA / NPU 런타임을 혼합해서 쓰기 어렵고 운영이 복잡

### xpuruntime의 해결 방향

- GPU/NPU 실행 정책을 **명시적·코드화** (Policy as Code)
- 학습과 추론이 **동일한 Runtime/Policy 개념**을 공유
- 프레임워크 독립적 최적화 자산 축적
- 실행 선택/성능 근거를 로깅/프로파일링으로 제공 (설명 가능)
- 여러 엔진/커널을 통합하는 **Orchestrator/Control Plane**

---

## 4. 차별화 포인트

| 특징 | 설명 |
|------|------|
| **Policy as Code** | 커널/엔진 선택을 정책으로 고정/선호/자동 선택 |
| **일관된 정책 모델** | 학습–추론을 관통하는 동일한 정책 개념 |
| **프레임워크 독립** | PyTorch, TensorFlow 등에 종속되지 않는 성능 튜닝 자산 |
| **설명 가능** | 실행 근거(선택된 커널/엔진, sync/메모리 이동)를 기록 |
| **통합 API** | GPU + NPU를 동일한 Python API로 (서버/클라이언트/엣지) |

> 우리는 "모델 라이브러리"가 아니라  
> **AI Accelerator Execution Control Plane**이다.

---

## 5. 모듈형 경량 패키지 (PyTorch 대비 경량)

xpuruntime은 **모듈형 구조**로 설계하여, 필요한 기능만 설치할 수 있도록 한다.  
PyTorch(~2GB+) 대비 **코어만 설치 시 약 200-300MB** 수준으로 가볍게 사용할 수 있다.

### 5.1 패키지 크기 비교

| 설치 방식 | 예상 크기 | 용도 |
|----------|----------|------|
| **xpuruntime** (코어) | ~200-300MB | Inference 기본 (CUDA + ONNX Runtime) |
| + [tensorrt] | +100-200MB | TensorRT 백엔드 |
| + [torch] | +2GB | Training (PyTorch Extension) |
| + [openvino] | +200MB | Intel NPU |
| + [qnn] | +100MB | Qualcomm NPU |
| PyTorch (참고) | ~2-2.5GB | 전체 ML 프레임워크 |

### 5.2 설치 시나리오

```bash
# 기본 (Inference, ORT 백엔드만) - 가벼움
pip install xpuruntime

# TensorRT 백엔드 추가
pip install xpuruntime[tensorrt]

# Training 사용 시에만 PyTorch 포함
pip install xpuruntime[torch]

# 전체 백엔드
pip install xpuruntime[all]
```

### 5.3 설계 원칙

- **코어 패키지**: CUDA 런타임 + ONNX Runtime만 포함. Inference 전용 사용자는 PyTorch 없이 동작.
- **Training 모듈**: `xpuruntime.pytorch`는 **선택적**. `pip install xpuruntime[torch]` 시에만 로드 가능.
- **백엔드**: TensorRT, OpenVINO, QNN 등은 각각 optional extra로 분리.

---

## 6. 지원 범위 (단계적)

### Phase 1: MVP (NVIDIA CUDA 중심, 경량 코어)

- NVIDIA CUDA only
- InferenceSession: ONNX Runtime (CUDA EP) + TensorRT
- KernelPolicy: matmul (cuBLAS / cuBLASLt) 부터
- MemoryPool + Stream 관리 + 프로파일링 기본
- Python 바인딩 (pybind11) + C++ 런타임

### Phase 2: NPU 추가

- Intel NPU: OpenVINO 기반 (비교적 빠르게 추가 가능)
- Qualcomm NPU: QNN 기반 (SDK/배포/변환 파이프라인 포함)

### Phase 3: 멀티 백엔드 확장

- AMD ROCm
- Apple Metal
- 플러그인 구조로 확장

---

## 7. 빠른 시작 예시

### Inference

```python
import xpuruntime as xrt

policy = xrt.KernelPolicy(
    matmul="cublasLt",
    attention="flash_v2",
)

sess = xrt.InferenceSession(
    model="model.onnx",
    device="cuda:0",
    engine="tensorrt",
    kernel_policy=policy,
    fp16=True,
)

out = sess.run({"input": x})
```

### Training (PyTorch Extension, 선택적)

`pip install xpuruntime[torch]` 설치 후 사용 가능.

```python
import xpuruntime.pytorch as xrt_torch

xrt_torch.set_kernel_policy(
    attention="flash_v2",
    matmul="cublasLt",
)

with xrt_torch.autocast(fp16=True), xrt_torch.capture_graph():
    y = model(x)
    loss = criterion(y, t)
    loss.backward()
```

### NPU Inference

```python
import xpuruntime as xrt

sess = xrt.InferenceSession(
    "model.onnx",
    device="npu",
    backend="openvino",  # or "qnn"
    exec_policy=xrt.ExecutionPolicy(precision="int8", static_shape=True),
)
```

---

## 8. 관련 문서

- [01_architecture.md](./01_architecture.md) - 전체 아키텍처 상세
- [02_cpp_core_runtime.md](./02_cpp_core_runtime.md) - C++ 코어 런타임 설계
- [03_python_binding.md](./03_python_binding.md) - Python 바인딩 설계
- [04_inference_module.md](./04_inference_module.md) - 추론 모듈 설계
- [05_training_module.md](./05_training_module.md) - 학습 모듈 설계
- [06_kernel_policy.md](./06_kernel_policy.md) - 커널 정책 상세
- [07_build_packaging.md](./07_build_packaging.md) - 빌드/패키징 설계
- [08_testing_strategy.md](./08_testing_strategy.md) - 테스트 전략

---

## 9. 라이선스

Apache License 2.0 (예정)

---

## 10. 기여

[CONTRIBUTING.md](../CONTRIBUTING.md) 참조 (작성 예정)
