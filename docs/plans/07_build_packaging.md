# xpuruntime - 빌드 및 패키징 설계

> 이 문서는 xpuruntime의 빌드 시스템과 PyPI 패키징 전략을 설명한다.

---

## 1. 개요

xpuruntime은 C++/CUDA 코어와 Python SDK로 구성되어 있어,  
**네이티브 확장 빌드**와 **Python 패키징**을 모두 고려해야 한다.

### 모듈형 경량 패키지 전략

- **코어 패키지**: CUDA 런타임 + ONNX Runtime만 기본 포함 (~200-300MB). PyTorch 없이 Inference 전용 사용 가능.
- **선택적 extras**: TensorRT, PyTorch(Training), OpenVINO, QNN 등은 `pip install xpuruntime[extra]` 로만 설치.
- **PyTorch 대비**: 코어만 설치 시 약 10배 이상 가벼움 (~2GB vs ~200-300MB).

### 빌드 스택

| 컴포넌트 | 도구 |
|----------|------|
| C++ 빌드 | CMake 3.24+ |
| CUDA 컴파일 | nvcc (CUDA Toolkit) |
| Python 바인딩 | pybind11 |
| Python 패키징 | scikit-build-core |
| 배포 | PyPI (wheel) |

### 1.1 설치 시나리오별 크기

| 설치 명령 | 예상 크기 | 용도 |
|----------|----------|------|
| `pip install xpuruntime` | ~200-300MB | Inference (ORT 백엔드) |
| `pip install xpuruntime[tensorrt]` | +100-200MB | TensorRT 백엔드 |
| `pip install xpuruntime[torch]` | +2GB | Training (PyTorch) |
| `pip install xpuruntime[all]` | 전체 | 모든 백엔드 + 개발 도구 |

---

## 2. 프로젝트 구조

```
xpuruntime/
├── CMakeLists.txt              # 최상위 CMake
├── pyproject.toml              # Python 패키지 설정
├── setup.py                    # (선택) legacy 호환
├── MANIFEST.in                 # 소스 배포 포함 파일
│
├── cmake/
│   ├── FindTensorRT.cmake      # TensorRT 찾기
│   ├── FindOnnxRuntime.cmake   # ORT 찾기
│   └── cuda_utils.cmake        # CUDA 유틸리티
│
├── src/
│   ├── cpp/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   ├── core/
│   │   ├── backends/
│   │   └── bindings/
│   │
│   └── python/
│       └── xpuruntime/
│
└── tests/
```

---

## 3. CMake 설정

### 3.1 최상위 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.24)
project(xpuruntime VERSION 0.1.0 LANGUAGES CXX CUDA)

# C++ 표준
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 옵션
option(XRT_BUILD_TESTS "Build tests" ON)
option(XRT_BUILD_PYTHON "Build Python bindings" ON)
option(XRT_USE_TENSORRT "Enable TensorRT backend" ON)
option(XRT_USE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)

# CUDA 설정
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# 의존성 찾기
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# cuBLAS, cuDNN
find_package(CUDAToolkit REQUIRED COMPONENTS cublas cudnn)

# TensorRT (선택)
if(XRT_USE_TENSORRT)
    find_package(TensorRT REQUIRED)
endif()

# ONNX Runtime (선택)
if(XRT_USE_ONNXRUNTIME)
    find_package(OnnxRuntime REQUIRED)
endif()

# pybind11
if(XRT_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
endif()

# 서브디렉토리
add_subdirectory(src/cpp)

if(XRT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
```

### 3.2 C++ 코어 CMakeLists.txt

```cmake
# src/cpp/CMakeLists.txt

# 코어 라이브러리
add_library(xpuruntime_core SHARED
    core/device_manager.cpp
    core/memory_manager.cpp
    core/stream_manager.cpp
    core/kernel_registry.cpp
    core/dispatcher.cpp
    core/profiler.cpp
)

target_include_directories(xpuruntime_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(xpuruntime_core PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
)

# CUDA 커널 라이브러리
add_library(xpuruntime_kernels STATIC
    backends/cuda_raw/flash_attention.cu
    backends/cuda_raw/fused_layernorm.cu
    backends/cuda_raw/gemm_kernels.cu
)

set_target_properties(xpuruntime_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(xpuruntime_core PRIVATE xpuruntime_kernels)

# Backend 라이브러리들
add_subdirectory(backends/cublas)
add_subdirectory(backends/cudnn)

if(XRT_USE_TENSORRT)
    add_subdirectory(backends/tensorrt)
    target_link_libraries(xpuruntime_core PRIVATE xpuruntime_tensorrt)
endif()

if(XRT_USE_ONNXRUNTIME)
    add_subdirectory(backends/onnxruntime)
    target_link_libraries(xpuruntime_core PRIVATE xpuruntime_onnxrt)
endif()

# Python 바인딩
if(XRT_BUILD_PYTHON)
    pybind11_add_module(_core
        bindings/module.cpp
        bindings/device_binding.cpp
        bindings/memory_binding.cpp
        bindings/stream_binding.cpp
        bindings/kernel_binding.cpp
        bindings/dispatcher_binding.cpp
        bindings/profiler_binding.cpp
        bindings/tensor_binding.cpp
        bindings/session_binding.cpp
    )
    
    target_link_libraries(_core PRIVATE xpuruntime_core)
    
    # 설치 경로 설정
    install(TARGETS _core
        LIBRARY DESTINATION xpuruntime
    )
endif()

# 설치
install(TARGETS xpuruntime_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/xpuruntime
    DESTINATION include
)
```

### 3.3 TensorRT Backend CMake

```cmake
# src/cpp/backends/tensorrt/CMakeLists.txt

add_library(xpuruntime_tensorrt STATIC
    tensorrt_engine.cpp
    trt_logger.cpp
    trt_utils.cpp
)

target_include_directories(xpuruntime_tensorrt PRIVATE
    ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries(xpuruntime_tensorrt PRIVATE
    ${TensorRT_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)
```

---

## 4. Python 패키징

### 4.1 pyproject.toml

```toml
[build-system]
requires = [
    "scikit-build-core>=0.8",
    "pybind11>=2.11",
]
build-backend = "scikit_build_core.build"

[project]
name = "xpuruntime"
version = "0.1.0"
description = "Unified GPU/NPU execution runtime for ML workloads"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "xpuruntime contributors"}
]
keywords = ["gpu", "npu", "cuda", "tensorrt", "onnx", "inference", "training"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: C++",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.21",
]

# 선택적 의존성: 필요한 기능만 설치하여 패키지 크기 최소화
[project.optional-dependencies]
# TensorRT 백엔드 (~+100-200MB)
tensorrt = []
# PyTorch Extension (Training) - 무거움 (~+2GB)
torch = ["torch>=2.0"]
# ONNX 모델 검증용
onnx = ["onnx>=1.14"]
# Intel NPU (Phase 2)
openvino = ["openvino>=2023.0"]
# Qualcomm NPU (Phase 2)
qnn = []
# 개발/테스트
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "ruff",
    "mypy",
]
docs = [
    "sphinx",
    "sphinx-rtd-theme",
]
# 전체 백엔드 (Training 포함 시 무거움)
all = ["xpuruntime[tensorrt,torch,onnx,openvino,qnn,dev,docs]"]

[project.urls]
Homepage = "https://github.com/yourusername/xpuruntime"
Documentation = "https://xpuruntime.readthedocs.io"
Repository = "https://github.com/yourusername/xpuruntime"
Issues = "https://github.com/yourusername/xpuruntime/issues"

[tool.scikit-build]
cmake.minimum-version = "3.24"
cmake.build-type = "Release"
cmake.args = [
    "-DXRT_BUILD_TESTS=OFF",
    "-DXRT_BUILD_PYTHON=ON",
]
wheel.packages = ["src/python/xpuruntime"]
wheel.install-dir = "xpuruntime"

[tool.scikit-build.cmake.define]
XRT_USE_TENSORRT = "ON"
XRT_USE_ONNXRUNTIME = "ON"

# 플랫폼별 설정
[[tool.scikit-build.overrides]]
if.platform-system = "linux"
cmake.args = ["-DCMAKE_CUDA_ARCHITECTURES=70;75;80;86;89;90"]
```

### 4.2 MANIFEST.in

```
include LICENSE
include README.md
include CMakeLists.txt
include pyproject.toml

recursive-include src *.cpp *.h *.hpp *.cu *.cuh
recursive-include src/python *.py
recursive-include cmake *.cmake
recursive-include tests *.cpp *.py

prune build
prune dist
prune *.egg-info
global-exclude __pycache__
global-exclude *.pyc
global-exclude *.so
```

---

## 5. 의존성 관리

### 5.1 시스템 의존성

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libcudnn8 \
    libcudnn8-dev

# CUDA Toolkit (별도 설치)
# https://developer.nvidia.com/cuda-downloads

# TensorRT (별도 설치)
# https://developer.nvidia.com/tensorrt
```

### 5.2 CMake Find 모듈

```cmake
# cmake/FindTensorRT.cmake

# TensorRT 찾기
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    PATH_SUFFIXES include
)

find_library(TensorRT_LIBRARY nvinfer
    HINTS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64
)

find_library(TensorRT_ONNX_LIBRARY nvonnxparser
    HINTS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARY} ${TensorRT_ONNX_LIBRARY})
    
    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES
            IMPORTED_LOCATION "${TensorRT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
endif()
```

---

## 6. Wheel 빌드

### 6.1 로컬 빌드

```bash
# 개발 모드 설치
pip install -e ".[dev]"

# wheel 빌드
pip wheel . --no-deps -w dist/

# 빌드 + 설치
pip install .
```

### 6.2 manylinux wheel

```dockerfile
# Dockerfile.manylinux
FROM quay.io/pypa/manylinux_2_28_x86_64

# CUDA 설치
RUN yum install -y cuda-toolkit-12-2

# TensorRT 설치
COPY TensorRT-*.tar.gz /tmp/
RUN tar -xzf /tmp/TensorRT-*.tar.gz -C /usr/local/

# 빌드 의존성
RUN pip install scikit-build-core pybind11

# 소스 복사
COPY . /workspace
WORKDIR /workspace

# wheel 빌드
RUN for PYTHON in cp310 cp311 cp312; do \
    /opt/python/${PYTHON}-${PYTHON}/bin/pip wheel . -w dist/; \
done

# auditwheel repair
RUN for whl in dist/*.whl; do \
    auditwheel repair "$whl" -w wheelhouse/; \
done
```

### 6.3 빌드 스크립트

```bash
#!/bin/bash
# scripts/build_wheels.sh

set -e

# 빌드 디렉토리 정리
rm -rf build dist wheelhouse

# Docker 빌드
docker build -t xpuruntime-builder -f Dockerfile.manylinux .

# wheel 추출
docker run --rm -v $(pwd)/wheelhouse:/output xpuruntime-builder \
    cp -r /workspace/wheelhouse/* /output/

echo "Wheels built in wheelhouse/"
ls -la wheelhouse/
```

---

## 7. CI/CD 파이프라인

### 7.1 GitHub Actions

```yaml
# .github/workflows/build.yml
name: Build and Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y python3-pip cmake ninja-build
          pip3 install scikit-build-core pybind11 pytest
      
      - name: Install TensorRT
        run: |
          # TensorRT 설치 스크립트
          ./scripts/install_tensorrt.sh
      
      - name: Build
        run: pip3 install -e ".[dev]"
      
      - name: Test
        run: pytest tests/python -v

  build-wheel:
    runs-on: ubuntu-latest
    needs: build
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build wheels
        run: |
          docker build -t builder -f Dockerfile.manylinux .
          docker run --rm -v $(pwd)/wheelhouse:/out builder cp -r /workspace/wheelhouse/* /out/
      
      - name: Upload wheels
        uses: actions/upload-artifact@v3
        with:
          name: wheels
          path: wheelhouse/*.whl
```

### 7.2 PyPI 배포

```yaml
# .github/workflows/release.yml
name: Release to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Download wheels
        uses: actions/download-artifact@v3
        with:
          name: wheels
          path: dist/
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

---

## 8. 개발 환경 설정

### 8.1 개발 환경 구성

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/xpuruntime.git
cd xpuruntime

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate

# 3. 개발 의존성 설치
pip install -e ".[dev]"

# 4. pre-commit 설정
pip install pre-commit
pre-commit install
```

### 8.2 CMake 직접 빌드

```bash
# CMake 빌드 (개발/디버깅용)
mkdir build && cd build

cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DXRT_BUILD_TESTS=ON \
    -DXRT_BUILD_PYTHON=ON

ninja

# 테스트 실행
ctest --output-on-failure
```

### 8.3 IDE 설정 (VSCode)

```json
// .vscode/settings.json
{
    "cmake.configureArgs": [
        "-DXRT_BUILD_TESTS=ON",
        "-DXRT_BUILD_PYTHON=ON"
    ],
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/python"]
}
```

---

## 9. 버전 관리

### 9.1 버전 체계

- **MAJOR.MINOR.PATCH** (SemVer)
- MAJOR: 호환성 깨지는 변경
- MINOR: 새 기능 (하위 호환)
- PATCH: 버그 수정

### 9.2 버전 동기화

```python
# src/python/xpuruntime/__init__.py
__version__ = "0.1.0"  # pyproject.toml과 동기화
```

```cmake
# CMakeLists.txt
project(xpuruntime VERSION 0.1.0 ...)
```

---

## 10. 관련 문서

- [01_architecture.md](./01_architecture.md) - 전체 아키텍처
- [08_testing_strategy.md](./08_testing_strategy.md) - 테스트 전략
- [02_cpp_core_runtime.md](./02_cpp_core_runtime.md) - C++ 코어
