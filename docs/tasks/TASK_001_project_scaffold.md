# TASK_001: 프로젝트 스캐폴드

> Phase 1: Foundation

---

## 개요

프로젝트의 기본 디렉토리 구조와 빌드 시스템을 구성한다.

## 목표

- 표준 Python/C++ 프로젝트 구조 생성
- CMake 빌드 시스템 설정
- pyproject.toml 설정
- 기본 README, LICENSE 파일 생성

## 선행 작업

- 없음 (첫 번째 작업)

## 작업 내용

### 1. 디렉토리 구조 생성

```
xpuruntime/
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── cmake/
├── src/
│   ├── cpp/
│   │   ├── CMakeLists.txt
│   │   ├── include/xpuruntime/
│   │   ├── core/
│   │   ├── backends/
│   │   └── bindings/
│   └── python/
│       └── xpuruntime/
│           ├── __init__.py
│           ├── inference/
│           ├── training/
│           ├── policies/
│           └── runtime/
├── tests/
│   ├── cpp/
│   └── python/
└── examples/
```

### 2. CMakeLists.txt (최상위)

- CMake 3.24+ 설정
- CUDA, cuBLAS, cuDNN 찾기
- 옵션 정의 (XRT_BUILD_TESTS, XRT_BUILD_PYTHON 등)

### 3. pyproject.toml

- scikit-build-core 설정
- 패키지 메타데이터
- 의존성 정의

### 4. 기타 파일

- README.md: 프로젝트 소개
- LICENSE: Apache 2.0
- .gitignore: Python, C++, CMake 빌드 파일 무시

## 완료 조건

- [ ] 모든 디렉토리 생성됨
- [ ] `cmake ..` 실행 시 오류 없음 (의존성 설치 후)
- [ ] `pip install -e .` 실행 시 빈 패키지 설치됨
- [ ] README에 기본 정보 포함

## 예상 소요 시간

2-4시간

## 관련 문서

- [01_architecture.md](../plans/01_architecture.md)
- [07_build_packaging.md](../plans/07_build_packaging.md)
