# xpuruntime

A unified GPU/NPU execution runtime that bridges research and production by making engine and kernel selection explicit, reproducible, and framework-agnostic.

> **"연구에서 검증한 실행 전략(커널/정밀도/메모리 정책)을 그대로 서비스까지 가져간다."**

## Overview

**xpuruntime** is not a new model framework—it is a **runtime library that controls GPU/NPU execution**. It provides:

- **Policy as Code**: Kernel/engine selection as explicit, versioned policy
- **Unified policy model**: Same policy concepts across training and inference
- **Framework-agnostic**: Reusable tuning assets across PyTorch, TensorFlow, ONNX Runtime, TensorRT
- **Explainable execution**: Logging and profiling of which kernels/engines/precision were used

| Layer            | Role                          | Language |
|------------------|-------------------------------|----------|
| Control Plane    | User UX, API, pipeline definition | Python   |
| Data Plane       | Execution runtime (memory, stream, kernel, engine) | C++      |
| Performance Layer| Engine/kernel internals       | CUDA etc.|

## Install

Default install is **pure Python** (no CUDA/CMake required):

```bash
pip install -e ".[dev]"   # editable + dev deps
```

The C++/CUDA extension is optional; when CUDA and CMake are available, see [07_build_packaging.md](docs/plans/07_build_packaging.md) for native build.

## Documentation

- **설계/계획**: [`docs/plans/`](docs/plans/) – Vision, architecture, C++/Python/빌드/테스트 등
- **작업**: [`docs/tasks/`](docs/tasks/) – TASK_001~014, task_log
- **이슈 초안**: [`docs/issues/`](docs/issues/) – GitHub Issue용 초안
- **현재 구조**: [CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) – 구현 상태 점검

| 문서 | 설명 |
|------|------|
| [00_overview](docs/plans/00_overview.md) | 비전, 타깃 사용자, 차별화 |
| [01_architecture](docs/plans/01_architecture.md) | 아키텍처 |
| [02_cpp_core_runtime](docs/plans/02_cpp_core_runtime.md) | C++ 코어 |
| [03_python_binding](docs/plans/03_python_binding.md) | Python 바인딩 |
| [04~08](docs/plans/) | Inference, Training, Kernel policy, Build, Testing |
| [tasks/](docs/tasks/) | TASK_001~014, task_log |

## Open Source & Contributing

xpuruntime is **open source** and maintained as a **nonprofit, community-driven** project. We welcome contributions: bug reports, feature ideas, docs, and code. Issues tagged with `good first issue` or `help wanted` are a good place to start.

- **[CONTRIBUTING.md](CONTRIBUTING.md)** – 개발 환경 설정, PR 절차, 코드 스타일
- **[Issues](https://github.com/devyulbae/xpu_runtime/issues)** – 버그 리포트, 기능 제안, 기여 이슈

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.
