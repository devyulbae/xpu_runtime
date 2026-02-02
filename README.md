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

The C++/CUDA extension is optional; when CUDA and CMake are available, see [07_build_packaging.md](docs/07_build_packaging.md) for native build.

## Documentation

Design and architecture docs are in [`docs/`](docs/):

- [00_overview.md](docs/00_overview.md) – Vision, target users, differentiation
- [01_architecture.md](docs/01_architecture.md) – Architecture
- [02_cpp_core_runtime.md](docs/02_cpp_core_runtime.md) – C++ core
- [03_python_binding.md](docs/03_python_binding.md) – Python bindings
- [04_inference_module.md](docs/04_inference_module.md) – Inference module
- [05_training_module.md](docs/05_training_module.md) – Training module
- [06_kernel_policy.md](docs/06_kernel_policy.md) – Kernel policy
- [07_build_packaging.md](docs/07_build_packaging.md) – Build & packaging
- [08_testing_strategy.md](docs/08_testing_strategy.md) – Testing strategy
- [docs/tasks/](docs/tasks/) – Task breakdown (TASK_001–014)

## Open Source & Contributing

xpuruntime is **open source** and maintained as a **nonprofit, community-driven** project. We welcome contributions: bug reports, feature ideas, docs, and code. Issues tagged with `good first issue` or `help wanted` are a good place to start.

- **[CONTRIBUTING.md](CONTRIBUTING.md)** – 개발 환경 설정, PR 절차, 코드 스타일
- **[Issues](https://github.com/devyulbae/xpu_runtime/issues)** – 버그 리포트, 기능 제안, 기여 이슈

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.
