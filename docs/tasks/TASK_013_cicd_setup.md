# TASK_013: CI/CD 설정

> Phase 5: Release

---

## 개요

GitHub Actions를 사용한 CI/CD 파이프라인을 구성한다.

## 목표

- 자동 빌드/테스트 파이프라인
- 코드 품질 검사 (lint, type check)
- manylinux wheel 빌드
- 테스트 커버리지 리포트

## 선행 작업

- TASK_012: 커널 정책 적용

## 작업 내용

### 1. 기본 CI 워크플로우

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install tools
        run: pip install black ruff mypy
      
      - name: Black
        run: black --check src/python tests
      
      - name: Ruff
        run: ruff check src/python tests
      
      - name: MyPy
        run: mypy src/python/xpuruntime

  test-python:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -e ".[dev]"
      - name: Test (no GPU)
        run: pytest tests/python -v -m "not requires_gpu"

  test-gpu:
    runs-on: [self-hosted, gpu]
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -e ".[dev]"
      - name: Test (GPU)
        run: pytest tests/python -v -m "requires_gpu"
      - name: Coverage
        run: pytest tests/python --cov=xpuruntime --cov-report=xml
      - uses: codecov/codecov-action@v3

  test-cpp:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: |
          mkdir build && cd build
          cmake .. -DXRT_BUILD_TESTS=ON
          make -j$(nproc)
      - name: Test
        run: cd build && ctest --output-on-failure
```

### 2. Wheel 빌드 워크플로우

```yaml
# .github/workflows/build-wheels.yml
name: Build Wheels

on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build-wheels:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build manylinux wheels
        run: |
          docker build -t builder -f Dockerfile.manylinux .
          docker run --rm -v $PWD/wheelhouse:/out builder \
            cp -r /workspace/wheelhouse/* /out/
      
      - name: Upload wheels
        uses: actions/upload-artifact@v3
        with:
          name: wheels
          path: wheelhouse/*.whl
```

### 3. Dockerfile.manylinux

```dockerfile
# Dockerfile.manylinux
FROM quay.io/pypa/manylinux_2_28_x86_64

# CUDA Toolkit
RUN yum install -y cuda-toolkit-12-2

# TensorRT
COPY TensorRT-*.tar.gz /tmp/
RUN tar -xzf /tmp/TensorRT-*.tar.gz -C /usr/local/

# ONNX Runtime
RUN pip install onnxruntime-gpu

# 빌드
WORKDIR /workspace
COPY . .

RUN for PYTHON in cp310 cp311 cp312; do \
    /opt/python/${PYTHON}-${PYTHON}/bin/pip wheel . -w dist/; \
done

RUN for whl in dist/*.whl; do \
    auditwheel repair "$whl" -w wheelhouse/; \
done
```

### 4. Pre-commit 설정

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml

  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix]
```

### 5. 테스트 구성

```ini
# pytest.ini
[pytest]
testpaths = tests/python
markers =
    requires_gpu: requires GPU
    benchmark: performance test
addopts = -v --tb=short
```

## 완료 조건

- [ ] PR 생성 시 자동 lint/test 실행
- [ ] main 브랜치 push 시 GPU 테스트 실행
- [ ] 태그 push 시 wheel 빌드
- [ ] pre-commit hook 동작
- [ ] 커버리지 리포트 생성

## 예상 소요 시간

6-8시간

## 관련 문서

- [07_build_packaging.md](../plans/07_build_packaging.md)
- [08_testing_strategy.md](../plans/08_testing_strategy.md)
