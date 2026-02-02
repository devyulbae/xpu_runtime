# xpuruntime 구현 작업 목록

> 이 폴더는 xpuruntime 구현을 위한 세부 작업 문서를 포함합니다.

---

## 작업 흐름

```
Phase 1: Foundation
├── TASK_001: 프로젝트 스캐폴드
├── TASK_002: C++ 코어 스켈레톤
└── TASK_003: Python 바인딩 기본
         │
         ▼
Phase 2: Core Features
├── TASK_004: DeviceManager 구현
├── TASK_005: MemoryManager 구현
├── TASK_006: StreamManager 구현
└── TASK_007: KernelRegistry 구현
         │
         ▼
Phase 3: Inference
├── TASK_008: ONNX Runtime 통합
├── TASK_009: TensorRT 통합
└── TASK_010: InferenceSession 구현
         │
         ▼
Phase 4: Training
├── TASK_011: PyTorch Extension
└── TASK_012: 커널 정책 적용
         │
         ▼
Phase 5: Release
├── TASK_013: CI/CD 설정
└── TASK_014: PyPI 배포
```

---

## 작업 목록

### Phase 1: Foundation

| Task | 이름 | 예상 시간 | 상태 |
|------|------|----------|------|
| [TASK_001](./TASK_001_project_scaffold.md) | 프로젝트 스캐폴드 | 2-4h | 대기 |
| [TASK_002](./TASK_002_cpp_core_skeleton.md) | C++ 코어 스켈레톤 | 4-6h | 대기 |
| [TASK_003](./TASK_003_python_binding.md) | Python 바인딩 기본 | 4-6h | 대기 |

### Phase 2: Core Features

| Task | 이름 | 예상 시간 | 상태 |
|------|------|----------|------|
| [TASK_004](./TASK_004_device_manager.md) | DeviceManager | 4-6h | 대기 |
| [TASK_005](./TASK_005_memory_manager.md) | MemoryManager | 6-8h | 대기 |
| [TASK_006](./TASK_006_stream_manager.md) | StreamManager | 4-6h | 대기 |
| [TASK_007](./TASK_007_kernel_registry.md) | KernelRegistry | 6-8h | 대기 |

### Phase 3: Inference

| Task | 이름 | 예상 시간 | 상태 |
|------|------|----------|------|
| [TASK_008](./TASK_008_ort_integration.md) | ONNX Runtime 통합 | 8-12h | 대기 |
| [TASK_009](./TASK_009_tensorrt_integration.md) | TensorRT 통합 | 10-14h | 대기 |
| [TASK_010](./TASK_010_inference_session.md) | InferenceSession | 8-10h | 대기 |

### Phase 4: Training

| Task | 이름 | 예상 시간 | 상태 |
|------|------|----------|------|
| [TASK_011](./TASK_011_pytorch_extension.md) | PyTorch Extension | 12-16h | 대기 |
| [TASK_012](./TASK_012_kernel_policy_apply.md) | 커널 정책 적용 | 10-14h | 대기 |

### Phase 5: Release

| Task | 이름 | 예상 시간 | 상태 |
|------|------|----------|------|
| [TASK_013](./TASK_013_cicd_setup.md) | CI/CD 설정 | 6-8h | 대기 |
| [TASK_014](./TASK_014_pypi_publish.md) | PyPI 배포 | 4-6h | 대기 |

---

## 총 예상 시간

- **Phase 1**: 10-16시간
- **Phase 2**: 20-28시간
- **Phase 3**: 26-36시간
- **Phase 4**: 22-30시간
- **Phase 5**: 10-14시간

**총계**: 88-124시간 (약 11-16일 @ 8시간/일)

---

## 작업 가이드

### 작업 시작 전

1. 선행 작업 완료 확인
2. 관련 설계 문서 재검토
3. 필요한 의존성 설치 확인

### 작업 중

1. 완료 조건 체크리스트 확인
2. 테스트 작성 병행
3. 문서화 (docstring, 주석)

### 작업 완료 후

1. 모든 완료 조건 충족 확인
2. 테스트 통과 확인
3. 코드 리뷰 요청
4. 상태 업데이트
