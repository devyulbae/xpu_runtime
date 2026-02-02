# xpuruntime - 커널 정책 설계

> 이 문서는 xpuruntime의 핵심 차별점인 KernelPolicy와 ExecutionPolicy를 상세히 설명한다.

---

## 1. 개요

### 1.1 Policy as Code

xpuruntime의 핵심 철학은 **"실행 결정을 코드로 명시화"**하는 것이다.

| 기존 방식 | xpuruntime 방식 |
|----------|----------------|
| 프레임워크가 자동으로 커널 선택 | 사용자가 명시적으로 지정 |
| 선택 근거 불투명 (black box) | 선택 과정 로깅/추적 가능 |
| 재현성 보장 어려움 | 동일 정책 → 동일 결과 |
| 성능 튜닝 자산 휘발성 | 정책 파일로 버전 관리 |

### 1.2 두 가지 정책 유형

| 정책 | 대상 | 용도 |
|------|------|------|
| **KernelPolicy** | GPU (커널 단위) | matmul, attention 등 개별 op |
| **ExecutionPolicy** | NPU (그래프 단위) | 정밀도, shape 제약, fallback |

---

## 2. KernelPolicy

### 2.1 개념

KernelPolicy는 **연산(op) 단위**로 어떤 커널 구현체를 사용할지 지정한다.

```python
# 예시: matmul에는 cuBLASLt, attention에는 Flash Attention v2 사용
policy = KernelPolicy(
    matmul="cublasLt",
    attention="flash_v2",
)
```

### 2.2 지원 연산 및 커널

#### matmul (행렬 곱셈)

| 커널 | 설명 | 특징 |
|------|------|------|
| `cublas` | 기본 cuBLAS | 안정적, 범용 |
| `cublasLt` | cuBLAS Light | 더 많은 튜닝 옵션, Tensor Core 최적화 |
| `cutlass` | NVIDIA CUTLASS | 커스터마이징 가능, FP8 지원 |

#### attention (어텐션)

| 커널 | 설명 | 특징 |
|------|------|------|
| `torch` | PyTorch 기본 | 호환성 |
| `flash_v2` | Flash Attention v2 | 메모리 효율적, 긴 시퀀스 |
| `xformers` | xFormers | 다양한 attention 패턴 |

#### conv (합성곱)

| 커널 | 설명 | 특징 |
|------|------|------|
| `cudnn` | cuDNN | 기본, auto-tune |
| `cudnn_implicit_gemm` | cuDNN GEMM | 특정 shape에서 빠름 |
| `cudnn_winograd` | cuDNN Winograd | 작은 필터에 효율적 |

### 2.3 C++ 구현

```cpp
namespace xpuruntime {

struct KernelPolicy {
    // op_name -> kernel_name 매핑
    std::unordered_map<std::string, std::string> preferences;
    
    // 자동 선택 전략
    enum class AutoSelectStrategy {
        FirstSupported,   // 첫 번째 지원 커널
        BestPerformance,  // 벤치마크 기반 (future)
        LowestMemory,     // 메모리 사용량 최소화 (future)
    };
    AutoSelectStrategy auto_strategy = AutoSelectStrategy::FirstSupported;
    
    // 기본 정밀도 힌트
    std::string default_precision = "fp32";  // fp32, fp16, bf16, int8
    
    // 직렬화
    std::string to_json() const {
        nlohmann::json j;
        j["preferences"] = preferences;
        j["auto_strategy"] = static_cast<int>(auto_strategy);
        j["default_precision"] = default_precision;
        return j.dump(2);
    }
    
    static KernelPolicy from_json(const std::string& json) {
        auto j = nlohmann::json::parse(json);
        KernelPolicy policy;
        policy.preferences = j["preferences"].get<std::unordered_map<std::string, std::string>>();
        policy.auto_strategy = static_cast<AutoSelectStrategy>(j["auto_strategy"].get<int>());
        policy.default_precision = j.value("default_precision", "fp32");
        return policy;
    }
    
    // 병합 (다른 정책과 합치기)
    KernelPolicy merge(const KernelPolicy& other) const {
        KernelPolicy merged = *this;
        for (const auto& [k, v] : other.preferences) {
            merged.preferences[k] = v;
        }
        return merged;
    }
};

}  // namespace xpuruntime
```

### 2.4 Python API

```python
# src/python/xpuruntime/policies/kernel_policy.py
from typing import Dict, Optional, Literal
from pathlib import Path
import json
from dataclasses import dataclass, field
from enum import Enum

class AutoSelectStrategy(Enum):
    FIRST_SUPPORTED = 0
    BEST_PERFORMANCE = 1
    LOWEST_MEMORY = 2

@dataclass
class KernelPolicy:
    """GPU 커널 선택 정책"""
    
    preferences: Dict[str, str] = field(default_factory=dict)
    auto_strategy: AutoSelectStrategy = AutoSelectStrategy.FIRST_SUPPORTED
    default_precision: Literal["fp32", "fp16", "bf16", "int8"] = "fp32"
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: op_name=kernel_name 형태의 커널 매핑
        
        Examples:
            policy = KernelPolicy(
                matmul="cublasLt",
                attention="flash_v2",
                conv="cudnn",
            )
        """
        self.preferences = {}
        self.auto_strategy = AutoSelectStrategy.FIRST_SUPPORTED
        self.default_precision = "fp32"
        
        for op_name, kernel_name in kwargs.items():
            self.preferences[op_name] = kernel_name
    
    def set(self, op_name: str, kernel_name: str) -> "KernelPolicy":
        """커널 매핑 추가/수정"""
        self.preferences[op_name] = kernel_name
        return self
    
    def get(self, op_name: str) -> Optional[str]:
        """특정 op의 커널 조회"""
        return self.preferences.get(op_name)
    
    def merge(self, other: "KernelPolicy") -> "KernelPolicy":
        """다른 정책과 병합 (other가 우선)"""
        merged = KernelPolicy(**self.preferences)
        merged.preferences.update(other.preferences)
        return merged
    
    def save(self, path: str | Path):
        """JSON 파일로 저장"""
        data = {
            "preferences": self.preferences,
            "auto_strategy": self.auto_strategy.value,
            "default_precision": self.default_precision,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "KernelPolicy":
        """JSON 파일에서 로드"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        policy = cls(**data.get("preferences", {}))
        policy.auto_strategy = AutoSelectStrategy(data.get("auto_strategy", 0))
        policy.default_precision = data.get("default_precision", "fp32")
        return policy
    
    def __repr__(self) -> str:
        prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
        return f"KernelPolicy({prefs})"
```

---

## 3. ExecutionPolicy

### 3.1 개념

ExecutionPolicy는 **NPU/엔진 수준**에서 실행 방식을 제어한다.  
NPU는 커널 단위 선택이 아니라 **그래프 컴파일** 기반이므로 다른 접근이 필요하다.

```python
# 예시: NPU에서 INT8 정밀도로 실행, CPU fallback 허용
policy = ExecutionPolicy(
    target="npu",
    precision="int8",
    static_shape=True,
    allow_fallback="cpu",
)
```

### 3.2 설정 항목

| 항목 | 타입 | 설명 |
|------|------|------|
| `target` | str | 실행 대상 ("cuda", "npu", "cpu") |
| `precision` | str | 정밀도 ("fp32", "fp16", "int8") |
| `static_shape` | bool | 정적 shape 강제 여부 |
| `allow_fallback` | str/None | fallback 허용 디바이스 |
| `optimize_level` | int | 최적화 레벨 (0-3) |
| `cache_compiled` | bool | 컴파일된 모델 캐시 여부 |

### 3.3 C++ 구현

```cpp
namespace xpuruntime {

struct ExecutionPolicy {
    // 실행 대상
    std::string target = "cuda";  // cuda, npu, cpu
    
    // 정밀도
    std::string precision = "fp32";  // fp32, fp16, bf16, int8
    
    // Shape 정책
    bool static_shape = false;
    std::vector<std::vector<int64_t>> fixed_shapes;  // static_shape=true일 때
    
    // Fallback 정책
    std::optional<std::string> allow_fallback;  // cpu, cuda, etc.
    
    // 최적화 레벨 (0=none, 1=basic, 2=extended, 3=aggressive)
    int optimize_level = 2;
    
    // 캐시 정책
    bool cache_compiled = true;
    std::string cache_dir = "";
    
    // 타임아웃 (컴파일/실행)
    int compile_timeout_ms = 300000;  // 5분
    int execution_timeout_ms = 30000;  // 30초
    
    // 직렬화
    std::string to_json() const;
    static ExecutionPolicy from_json(const std::string& json);
};

}  // namespace xpuruntime
```

### 3.4 Python API

```python
# src/python/xpuruntime/policies/execution_policy.py
from typing import Optional, List, Literal
from pathlib import Path
import json
from dataclasses import dataclass, field

@dataclass
class ExecutionPolicy:
    """NPU/엔진 실행 정책"""
    
    target: Literal["cuda", "npu", "cpu"] = "cuda"
    precision: Literal["fp32", "fp16", "bf16", "int8"] = "fp32"
    static_shape: bool = False
    fixed_shapes: Optional[List[List[int]]] = None
    allow_fallback: Optional[str] = None
    optimize_level: int = 2
    cache_compiled: bool = True
    cache_dir: str = ""
    compile_timeout_ms: int = 300000
    execution_timeout_ms: int = 30000
    
    def __post_init__(self):
        if self.static_shape and self.fixed_shapes is None:
            raise ValueError("fixed_shapes required when static_shape=True")
    
    def with_precision(self, precision: str) -> "ExecutionPolicy":
        """정밀도 변경한 새 정책 반환"""
        return ExecutionPolicy(
            target=self.target,
            precision=precision,
            static_shape=self.static_shape,
            fixed_shapes=self.fixed_shapes,
            allow_fallback=self.allow_fallback,
            optimize_level=self.optimize_level,
            cache_compiled=self.cache_compiled,
        )
    
    def with_fallback(self, device: Optional[str]) -> "ExecutionPolicy":
        """fallback 설정 변경한 새 정책 반환"""
        return ExecutionPolicy(
            target=self.target,
            precision=self.precision,
            static_shape=self.static_shape,
            fixed_shapes=self.fixed_shapes,
            allow_fallback=device,
            optimize_level=self.optimize_level,
            cache_compiled=self.cache_compiled,
        )
    
    def save(self, path: str | Path):
        """JSON 파일로 저장"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "ExecutionPolicy":
        """JSON 파일에서 로드"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
```

---

## 4. Dispatcher 동작

### 4.1 디스패치 흐름

```
요청: matmul(A, B)
       │
       ▼
┌──────────────────────────┐
│    KernelPolicy 확인     │
│ preferences["matmul"]?   │
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     │           │
  지정됨      미지정
     │           │
     ▼           ▼
 지정된 커널  AutoSelect
 조회         전략 적용
     │           │
     └─────┬─────┘
           │
           ▼
┌──────────────────────────┐
│    지원 여부 확인        │
│ kernel.supports(ctx)?    │
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     │           │
  지원함      미지원
     │           │
     ▼           ▼
  실행!      Fallback
             또는 에러
```

### 4.2 컨텍스트 기반 필터링

```cpp
bool CublasLtKernel::supports(const KernelContext& ctx) const {
    // dtype 확인
    if (ctx.dtype != "float32" && 
        ctx.dtype != "float16" && 
        ctx.dtype != "bfloat16") {
        return false;
    }
    
    // Tensor Core 요구 시 SM 버전 확인
    auto& dm = DeviceManager::instance();
    auto info = dm.get_device_info(ctx.device_id);
    if (info.compute_capability_major < 7) {
        return false;  // Tensor Core는 SM70+
    }
    
    // Shape 제약 (예: 8의 배수)
    for (const auto& shape : ctx.input_shapes) {
        if (shape.back() % 8 != 0) {
            return false;  // cuBLASLt는 8의 배수 선호
        }
    }
    
    return true;
}
```

### 4.3 로깅

```cpp
void Dispatcher::dispatch(const std::string& op_name, KernelContext& ctx) {
    auto kernel = select_kernel(op_name, ctx);
    
    if (logging_enabled_) {
        DispatchLog log;
        log.op_name = op_name;
        log.selected_kernel = kernel->name();
        log.input_shape = ctx.input_shapes[0];
        log.dtype = ctx.dtype;
        
        auto start = std::chrono::high_resolution_clock::now();
        kernel->execute(ctx);
        auto end = std::chrono::high_resolution_clock::now();
        
        log.elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        logs_.push_back(log);
    } else {
        kernel->execute(ctx);
    }
}
```

---

## 5. 정책 관리

### 5.1 정책 파일 구조

```json
// kernel_policy.json
{
  "preferences": {
    "matmul": "cublasLt",
    "attention": "flash_v2",
    "conv": "cudnn"
  },
  "auto_strategy": 0,
  "default_precision": "fp16"
}
```

```json
// execution_policy.json
{
  "target": "npu",
  "precision": "int8",
  "static_shape": true,
  "fixed_shapes": [[1, 3, 224, 224]],
  "allow_fallback": "cpu",
  "optimize_level": 2,
  "cache_compiled": true
}
```

### 5.2 환경별 정책

```python
# 개발 환경 정책
dev_policy = KernelPolicy(
    matmul="cublas",  # 안정적인 기본 커널
)

# 프로덕션 환경 정책
prod_policy = KernelPolicy(
    matmul="cublasLt",
    attention="flash_v2",
)

# 환경에 따라 로드
import os
env = os.getenv("ENVIRONMENT", "dev")
policy = KernelPolicy.load(f"policies/{env}_kernel_policy.json")
```

### 5.3 정책 검증

```python
def validate_policy(policy: KernelPolicy) -> List[str]:
    """정책의 유효성 검증"""
    warnings = []
    
    from .. import _core
    registry = _core.KernelRegistry.instance()
    
    for op_name, kernel_name in policy.preferences.items():
        # op 존재 확인
        if op_name not in registry.get_registered_ops():
            warnings.append(f"Unknown op: {op_name}")
            continue
        
        # 커널 존재 확인
        available = registry.get_kernel_names(op_name)
        if kernel_name not in available:
            warnings.append(
                f"Unknown kernel '{kernel_name}' for op '{op_name}'. "
                f"Available: {available}"
            )
    
    return warnings
```

---

## 6. 고급 사용법

### 6.1 컨텍스트별 정책

```python
from contextlib import contextmanager

@contextmanager
def with_policy(policy: KernelPolicy):
    """일시적으로 다른 정책 적용"""
    from .. import _core
    dispatcher = _core.Dispatcher.instance()
    
    # 현재 정책 저장
    old_policy = dispatcher.get_policy()
    
    # 새 정책 적용
    dispatcher.set_policy(policy._get_internal())
    
    try:
        yield
    finally:
        # 원래 정책 복원
        dispatcher.set_policy(old_policy)


# 사용 예시
default_policy = KernelPolicy(matmul="cublas")
fast_policy = KernelPolicy(matmul="cublasLt")

xrt.set_kernel_policy(default_policy)

# 특정 구간만 다른 정책
with xrt.with_policy(fast_policy):
    output = model(input)  # cublasLt 사용
# 여기서는 cublas 사용
```

### 6.2 정책 병합

```python
# 기본 정책
base = KernelPolicy(
    matmul="cublas",
    conv="cudnn",
)

# 오버라이드 정책
override = KernelPolicy(
    matmul="cublasLt",  # matmul만 변경
)

# 병합 (override가 우선)
merged = base.merge(override)
# 결과: matmul="cublasLt", conv="cudnn"
```

### 6.3 조건부 정책

```python
def get_optimal_policy(model_type: str, batch_size: int) -> KernelPolicy:
    """모델 타입과 배치 크기에 따른 최적 정책"""
    
    if model_type == "transformer":
        if batch_size >= 32:
            return KernelPolicy(
                matmul="cublasLt",
                attention="flash_v2",
            )
        else:
            return KernelPolicy(
                matmul="cublas",
                attention="torch",  # 작은 배치에서는 오버헤드
            )
    elif model_type == "cnn":
        return KernelPolicy(
            conv="cudnn",
            matmul="cublas",
        )
    
    return KernelPolicy()  # 기본
```

---

## 7. 디버깅

### 7.1 디스패치 로그 확인

```python
import xpuruntime as xrt

# 로깅 활성화
xrt.enable_dispatch_logging()

# 모델 실행
output = model(input)

# 로그 확인
logs = xrt.get_dispatch_logs()
for log in logs:
    print(f"{log.op_name}: {log.selected_kernel} "
          f"({log.elapsed_us:.2f}us)")

# 로그 초기화
xrt.clear_dispatch_logs()
```

### 7.2 지원 커널 목록

```python
import xpuruntime as xrt

# 특정 op의 사용 가능한 커널 목록
available = xrt.get_available_kernels("matmul")
print(available)  # ["cublas", "cublasLt", "cutlass"]

# 모든 op와 커널 목록
all_ops = xrt.get_registered_ops()
for op in all_ops:
    kernels = xrt.get_available_kernels(op)
    print(f"{op}: {kernels}")
```

---

## 8. 관련 문서

- [02_cpp_core_runtime.md](./02_cpp_core_runtime.md) - Dispatcher 구현
- [04_inference_module.md](./04_inference_module.md) - 추론에서 정책 적용
- [05_training_module.md](./05_training_module.md) - 학습에서 정책 적용
