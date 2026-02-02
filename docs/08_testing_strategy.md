# xpuruntime - 테스트 전략

> 이 문서는 xpuruntime의 테스트 전략과 품질 보증 방안을 설명한다.

---

## 1. 개요

xpuruntime은 C++/CUDA 코어와 Python SDK로 구성되어 있어,  
**두 레이어 모두**에서 철저한 테스트가 필요하다.

### 테스트 피라미드

```
            ┌─────────────────┐
            │   E2E Tests     │  ← 전체 워크플로우
            ├─────────────────┤
            │ Integration     │  ← Python + C++ 통합
            ├─────────────────┤
            │   Unit Tests    │  ← 개별 컴포넌트
            └─────────────────┘
            
            Python Tests + C++ Tests
```

---

## 2. 테스트 구조

```
tests/
├── cpp/                        # C++ 단위 테스트
│   ├── CMakeLists.txt
│   ├── test_device_manager.cpp
│   ├── test_memory_manager.cpp
│   ├── test_stream_manager.cpp
│   ├── test_kernel_registry.cpp
│   ├── test_dispatcher.cpp
│   └── backends/
│       ├── test_cublas_backend.cpp
│       └── test_tensorrt_engine.cpp
│
├── python/                     # Python 테스트
│   ├── conftest.py             # pytest fixtures
│   ├── unit/                   # 단위 테스트
│   │   ├── test_device.py
│   │   ├── test_policy.py
│   │   └── test_session.py
│   ├── integration/            # 통합 테스트
│   │   ├── test_inference_e2e.py
│   │   └── test_training_e2e.py
│   └── benchmark/              # 성능 테스트
│       ├── bench_matmul.py
│       └── bench_inference.py
│
└── models/                     # 테스트용 모델
    ├── simple_mlp.onnx
    └── resnet18.onnx
```

---

## 3. C++ 테스트

### 3.1 테스트 프레임워크

**Google Test (gtest)** 사용

```cmake
# tests/cpp/CMakeLists.txt
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# 테스트 실행 파일
add_executable(xpuruntime_tests
    test_device_manager.cpp
    test_memory_manager.cpp
    test_stream_manager.cpp
    test_kernel_registry.cpp
    test_dispatcher.cpp
)

target_link_libraries(xpuruntime_tests
    xpuruntime_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(xpuruntime_tests)
```

### 3.2 DeviceManager 테스트

```cpp
// tests/cpp/test_device_manager.cpp
#include <gtest/gtest.h>
#include "xpuruntime/device_manager.h"

using namespace xpuruntime;

class DeviceManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        dm_ = &DeviceManager::instance();
    }
    
    DeviceManager* dm_;
};

TEST_F(DeviceManagerTest, GetDeviceCount) {
    int count = dm_->get_device_count();
    EXPECT_GE(count, 0);
}

TEST_F(DeviceManagerTest, GetDeviceInfo) {
    int count = dm_->get_device_count();
    if (count == 0) {
        GTEST_SKIP() << "No GPU available";
    }
    
    auto info = dm_->get_device_info(0);
    
    EXPECT_EQ(info.device_id, 0);
    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.total_memory, 0);
    EXPECT_GE(info.compute_capability_major, 5);  // 최소 SM 5.0
}

TEST_F(DeviceManagerTest, SetCurrentDevice) {
    int count = dm_->get_device_count();
    if (count < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for this test";
    }
    
    dm_->set_current_device(1);
    EXPECT_EQ(dm_->get_current_device(), 1);
    
    dm_->set_current_device(0);
    EXPECT_EQ(dm_->get_current_device(), 0);
}

TEST_F(DeviceManagerTest, InvalidDeviceThrows) {
    EXPECT_THROW(
        dm_->get_device_info(999),
        std::out_of_range
    );
}
```

### 3.3 MemoryManager 테스트

```cpp
// tests/cpp/test_memory_manager.cpp
#include <gtest/gtest.h>
#include "xpuruntime/memory_manager.h"

using namespace xpuruntime;

class MemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mm_ = &MemoryManager::instance();
        // 테스트 전 캐시 비우기
        mm_->empty_cache();
    }
    
    void TearDown() override {
        mm_->empty_cache();
    }
    
    MemoryManager* mm_;
};

TEST_F(MemoryManagerTest, AllocateAndDeallocate) {
    size_t size = 1024 * 1024;  // 1MB
    
    void* ptr = mm_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    
    // 메모리가 할당되었는지 확인
    EXPECT_GE(mm_->get_allocated_size(), size);
    
    mm_->deallocate(ptr);
    
    // 캐시에 보관되므로 allocated는 0이지만 cached는 유지
    EXPECT_GE(mm_->get_cached_size(), size);
}

TEST_F(MemoryManagerTest, CachingAllocator) {
    size_t size = 1024 * 1024;
    
    // 첫 번째 할당
    void* ptr1 = mm_->allocate(size);
    mm_->deallocate(ptr1);
    
    // 두 번째 할당 (캐시에서 재사용)
    void* ptr2 = mm_->allocate(size);
    
    // 같은 포인터가 재사용되어야 함
    EXPECT_EQ(ptr1, ptr2);
    
    mm_->deallocate(ptr2);
}

TEST_F(MemoryManagerTest, PinnedMemory) {
    size_t size = 1024 * 1024;
    
    void* ptr = mm_->allocate_pinned(size);
    ASSERT_NE(ptr, nullptr);
    
    // 호스트에서 접근 가능한지 확인
    memset(ptr, 0, size);
    
    mm_->deallocate_pinned(ptr);
}

TEST_F(MemoryManagerTest, EmptyCache) {
    void* ptr = mm_->allocate(1024 * 1024);
    mm_->deallocate(ptr);
    
    EXPECT_GT(mm_->get_cached_size(), 0);
    
    mm_->empty_cache();
    
    EXPECT_EQ(mm_->get_cached_size(), 0);
}
```

### 3.4 KernelRegistry 테스트

```cpp
// tests/cpp/test_kernel_registry.cpp
#include <gtest/gtest.h>
#include "xpuruntime/kernel_registry.h"

using namespace xpuruntime;

// 테스트용 Mock 커널
class MockKernel : public IKernel {
public:
    explicit MockKernel(const std::string& name) : name_(name) {}
    
    std::string name() const override { return name_; }
    
    bool supports(const KernelContext& ctx) const override {
        return ctx.dtype == "float32";
    }
    
    void execute(const KernelContext& ctx) override {
        // No-op for testing
    }
    
private:
    std::string name_;
};

class KernelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = &KernelRegistry::instance();
    }
    
    KernelRegistry* registry_;
};

TEST_F(KernelRegistryTest, RegisterAndGet) {
    // Mock 커널 등록
    registry_->register_kernel("test_op", std::make_unique<MockKernel>("mock1"));
    registry_->register_kernel("test_op", std::make_unique<MockKernel>("mock2"));
    
    auto kernels = registry_->get_kernels("test_op");
    EXPECT_EQ(kernels.size(), 2);
}

TEST_F(KernelRegistryTest, GetByName) {
    registry_->register_kernel("named_op", std::make_unique<MockKernel>("specific"));
    
    auto kernel = registry_->get_kernel("named_op", "specific");
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->name(), "specific");
}

TEST_F(KernelRegistryTest, UnknownOp) {
    auto kernels = registry_->get_kernels("nonexistent_op");
    EXPECT_TRUE(kernels.empty());
}
```

---

## 4. Python 테스트

### 4.1 pytest 설정

```python
# tests/python/conftest.py
import pytest
import numpy as np
import os

# GPU 가용성 확인
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )

@pytest.fixture(scope="session")
def has_gpu():
    """GPU 사용 가능 여부"""
    try:
        import xpuruntime as xrt
        return xrt.get_device_count() > 0
    except Exception:
        return False

@pytest.fixture(autouse=True)
def skip_if_no_gpu(request, has_gpu):
    """GPU 필요한 테스트 스킵"""
    if request.node.get_closest_marker("requires_gpu"):
        if not has_gpu:
            pytest.skip("GPU not available")

@pytest.fixture
def simple_model_path():
    """테스트용 간단한 ONNX 모델 경로"""
    return os.path.join(os.path.dirname(__file__), "../models/simple_mlp.onnx")

@pytest.fixture
def random_input():
    """랜덤 입력 생성 fixture"""
    def _create(shape, dtype=np.float32):
        return np.random.randn(*shape).astype(dtype)
    return _create
```

### 4.2 단위 테스트 - Device

```python
# tests/python/unit/test_device.py
import pytest
import xpuruntime as xrt

class TestDevice:
    @pytest.mark.requires_gpu
    def test_get_device_count(self):
        count = xrt.get_device_count()
        assert count >= 1
    
    @pytest.mark.requires_gpu
    def test_get_all_devices(self):
        devices = xrt.get_all_devices()
        assert len(devices) > 0
        
        for device in devices:
            assert isinstance(device, xrt.Device)
            assert device.name
            assert device.total_memory > 0
    
    @pytest.mark.requires_gpu
    def test_device_properties(self):
        device = xrt.Device(0)
        
        assert device.id == 0
        assert isinstance(device.name, str)
        assert device.total_memory_gb > 0
        assert isinstance(device.compute_capability, tuple)
        assert len(device.compute_capability) == 2
    
    @pytest.mark.requires_gpu
    def test_set_current_device(self):
        original = xrt.get_current_device()
        
        xrt.set_current_device(0)
        assert xrt.get_current_device().id == 0
        
        # 복원
        xrt.set_current_device(original)
    
    def test_invalid_device(self):
        with pytest.raises(Exception):  # CudaError or similar
            xrt.Device(9999)
```

### 4.3 단위 테스트 - Policy

```python
# tests/python/unit/test_policy.py
import pytest
import tempfile
import json
from pathlib import Path
import xpuruntime as xrt

class TestKernelPolicy:
    def test_creation(self):
        policy = xrt.KernelPolicy(
            matmul="cublasLt",
            attention="flash_v2",
        )
        
        assert policy.get("matmul") == "cublasLt"
        assert policy.get("attention") == "flash_v2"
        assert policy.get("nonexistent") is None
    
    def test_set_method(self):
        policy = xrt.KernelPolicy()
        policy.set("conv", "cudnn")
        
        assert policy.get("conv") == "cudnn"
    
    def test_chaining(self):
        policy = xrt.KernelPolicy()
        policy.set("a", "1").set("b", "2").set("c", "3")
        
        assert policy.preferences == {"a": "1", "b": "2", "c": "3"}
    
    def test_save_and_load(self):
        policy = xrt.KernelPolicy(
            matmul="cublasLt",
            attention="flash_v2",
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        
        try:
            policy.save(path)
            
            # 파일 내용 확인
            with open(path) as f:
                data = json.load(f)
            assert "preferences" in data
            
            # 로드
            loaded = xrt.KernelPolicy.load(path)
            assert loaded.preferences == policy.preferences
        finally:
            path.unlink()
    
    def test_merge(self):
        base = xrt.KernelPolicy(matmul="cublas", conv="cudnn")
        override = xrt.KernelPolicy(matmul="cublasLt")
        
        merged = base.merge(override)
        
        assert merged.get("matmul") == "cublasLt"  # overridden
        assert merged.get("conv") == "cudnn"       # preserved
    
    def test_repr(self):
        policy = xrt.KernelPolicy(matmul="cublasLt")
        repr_str = repr(policy)
        
        assert "KernelPolicy" in repr_str
        assert "matmul" in repr_str
```

### 4.4 통합 테스트 - Inference

```python
# tests/python/integration/test_inference_e2e.py
import pytest
import numpy as np
import xpuruntime as xrt

class TestInferenceE2E:
    @pytest.mark.requires_gpu
    def test_basic_inference(self, simple_model_path, random_input):
        """기본 추론 테스트"""
        sess = xrt.InferenceSession(
            simple_model_path,
            device="cuda:0",
            engine="onnxrt",
        )
        
        # 입출력 정보 확인
        assert len(sess.input_info) > 0
        assert len(sess.output_info) > 0
        
        # 추론 실행
        input_shape = sess.input_info[0]["shape"]
        input_data = random_input(input_shape)
        
        output = sess.run({"input": input_data})
        
        assert "output" in output
        assert output["output"].shape[0] == input_shape[0]
    
    @pytest.mark.requires_gpu
    def test_inference_with_policy(self, simple_model_path, random_input):
        """정책 적용 추론 테스트"""
        policy = xrt.KernelPolicy(matmul="cublasLt")
        
        sess = xrt.InferenceSession(
            simple_model_path,
            device="cuda:0",
            engine="onnxrt",
            kernel_policy=policy,
        )
        
        input_shape = sess.input_info[0]["shape"]
        input_data = random_input(input_shape)
        
        output = sess.run({"input": input_data})
        assert output is not None
    
    @pytest.mark.requires_gpu
    def test_fp16_inference(self, simple_model_path, random_input):
        """FP16 추론 테스트"""
        sess = xrt.InferenceSession(
            simple_model_path,
            device="cuda:0",
            engine="onnxrt",
            fp16=True,
        )
        
        input_shape = sess.input_info[0]["shape"]
        input_data = random_input(input_shape, dtype=np.float16)
        
        output = sess.run({"input": input_data})
        assert output is not None
    
    @pytest.mark.requires_gpu
    def test_tensorrt_inference(self, simple_model_path, random_input):
        """TensorRT 추론 테스트"""
        sess = xrt.InferenceSession(
            simple_model_path,
            device="cuda:0",
            engine="tensorrt",
        )
        
        assert sess.engine_name == "tensorrt"
        
        input_shape = sess.input_info[0]["shape"]
        input_data = random_input(input_shape)
        
        output = sess.run({"input": input_data})
        assert output is not None
    
    def test_invalid_model_path(self):
        """존재하지 않는 모델 경로"""
        with pytest.raises(FileNotFoundError):
            xrt.InferenceSession("nonexistent.onnx")
    
    @pytest.mark.requires_gpu
    def test_session_repr(self, simple_model_path):
        """세션 문자열 표현"""
        sess = xrt.InferenceSession(simple_model_path)
        repr_str = repr(sess)
        
        assert "InferenceSession" in repr_str
        assert simple_model_path in repr_str
```

---

## 5. 성능 테스트

### 5.1 벤치마크 프레임워크

```python
# tests/python/benchmark/bench_utils.py
import time
import statistics
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    
    def __str__(self):
        return (f"{self.name}: {self.mean_ms:.3f}ms "
                f"(± {self.std_ms:.3f}ms, min={self.min_ms:.3f}ms, "
                f"max={self.max_ms:.3f}ms, n={self.iterations})")


def benchmark(
    fn: Callable,
    warmup: int = 10,
    iterations: int = 100,
    name: str = "benchmark",
) -> BenchmarkResult:
    """함수 실행 시간 벤치마크"""
    
    # Warmup
    for _ in range(warmup):
        fn()
    
    # 동기화
    try:
        import torch
        torch.cuda.synchronize()
    except ImportError:
        pass
    
    # 측정
    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        
        # CUDA 동기화
        try:
            import torch
            torch.cuda.synchronize()
        except ImportError:
            pass
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return BenchmarkResult(
        name=name,
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_ms=min(times),
        max_ms=max(times),
        iterations=iterations,
    )
```

### 5.2 MatMul 벤치마크

```python
# tests/python/benchmark/bench_matmul.py
import pytest
import numpy as np
import xpuruntime as xrt
from .bench_utils import benchmark

SIZES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

@pytest.mark.requires_gpu
@pytest.mark.benchmark
@pytest.mark.parametrize("m,n,k", SIZES)
def test_matmul_cublas(m, n, k):
    """cuBLAS MatMul 벤치마크"""
    import torch
    
    xrt.set_kernel_policy(matmul="cublas")
    
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    
    def fn():
        return torch.matmul(a, b)
    
    result = benchmark(fn, name=f"cublas_{m}x{n}x{k}")
    print(result)


@pytest.mark.requires_gpu
@pytest.mark.benchmark
@pytest.mark.parametrize("m,n,k", SIZES)
def test_matmul_cublaslt(m, n, k):
    """cuBLASLt MatMul 벤치마크"""
    import torch
    
    xrt.set_kernel_policy(matmul="cublasLt")
    
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    
    def fn():
        return torch.matmul(a, b)
    
    result = benchmark(fn, name=f"cublasLt_{m}x{n}x{k}")
    print(result)
```

### 5.3 Inference 벤치마크

```python
# tests/python/benchmark/bench_inference.py
import pytest
import numpy as np
import xpuruntime as xrt
from .bench_utils import benchmark

@pytest.mark.requires_gpu
@pytest.mark.benchmark
def test_inference_latency(simple_model_path, random_input):
    """추론 지연 시간 벤치마크"""
    sess = xrt.InferenceSession(
        simple_model_path,
        device="cuda:0",
        engine="tensorrt",
        fp16=True,
    )
    
    input_shape = sess.input_info[0]["shape"]
    input_data = random_input(input_shape)
    inputs = {"input": input_data}
    
    def fn():
        return sess.run(inputs)
    
    result = benchmark(fn, warmup=50, iterations=500, name="inference")
    print(result)
    
    # 지연 시간 요구사항 확인 (예: 10ms 이하)
    assert result.mean_ms < 10.0, f"Inference too slow: {result.mean_ms}ms"


@pytest.mark.requires_gpu
@pytest.mark.benchmark
def test_throughput(simple_model_path, random_input):
    """처리량 벤치마크"""
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        sess = xrt.InferenceSession(
            simple_model_path,
            device="cuda:0",
            engine="tensorrt",
            max_batch_size=batch_size,
        )
        
        input_shape = list(sess.input_info[0]["shape"])
        input_shape[0] = batch_size
        input_data = random_input(input_shape)
        inputs = {"input": input_data}
        
        def fn():
            return sess.run(inputs)
        
        result = benchmark(fn, name=f"throughput_batch{batch_size}")
        
        samples_per_sec = (batch_size / result.mean_ms) * 1000
        print(f"Batch {batch_size}: {samples_per_sec:.1f} samples/sec")
```

---

## 6. CI 테스트 설정

### 6.1 pytest 설정

```ini
# pytest.ini
[pytest]
testpaths = tests/python
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    requires_gpu: mark test as requiring GPU
    benchmark: mark test as benchmark (slow)

addopts = -v --tb=short

filterwarnings =
    ignore::DeprecationWarning
```

### 6.2 GitHub Actions 테스트

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Run unit tests (no GPU)
        run: pytest tests/python/unit -v -m "not requires_gpu"

  gpu-tests:
    runs-on: [self-hosted, gpu]  # GPU runner 필요
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Run GPU tests
        run: pytest tests/python -v -m "requires_gpu"
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  cpp-tests:
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
      
      - name: Run C++ tests
        run: cd build && ctest --output-on-failure
```

---

## 7. 코드 품질

### 7.1 정적 분석

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json

  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [numpy]
```

### 7.2 커버리지 설정

```toml
# pyproject.toml (추가)
[tool.coverage.run]
source = ["src/python/xpuruntime"]
omit = ["*/__pycache__/*", "*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
fail_under = 80
```

---

## 8. 관련 문서

- [07_build_packaging.md](./07_build_packaging.md) - 빌드/패키징
- [02_cpp_core_runtime.md](./02_cpp_core_runtime.md) - C++ 코어
- [03_python_binding.md](./03_python_binding.md) - Python 바인딩
