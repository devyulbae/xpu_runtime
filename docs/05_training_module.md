# xpuruntime - 학습 모듈 설계

> 이 문서는 xpuruntime의 Training 모듈 (PyTorch Extension) 설계를 설명한다.

---

## 1. 개요

Training 모듈은 **PyTorch를 대체하지 않고**, PyTorch Extension으로서 학습 과정에서의 GPU 실행을 제어한다.

### 선택적 의존성 (경량 패키지)

- **PyTorch는 필수 아님**: 코어 패키지(`pip install xpuruntime`)만 설치 시 PyTorch는 설치되지 않음.
- **Training 사용 시에만**: `pip install xpuruntime[torch]` 로 설치한 경우에만 `xpuruntime.pytorch` 사용 가능.
- **lazy import**: `import xpuruntime.pytorch` 시점에 PyTorch 존재 여부 확인. 없으면 `ImportError`와 설치 안내 메시지 반환.

```python
# PyTorch 없이 설치한 환경에서
import xpuruntime.pytorch  # ImportError: xpuruntime.pytorch requires PyTorch. Install with: pip install xpuruntime[torch]
```

### 설계 원칙

- **PyTorch 호환**: 기존 PyTorch 코드 최소 수정
- **비침습적**: 필요한 부분만 선택적으로 적용
- **커널 정책**: KernelPolicy를 통한 명시적 커널 선택
- **성능 최적화**: CUDA Graph, 커스텀 op, 메모리 최적화

### 미지원 범위

- Autograd 재구현 (PyTorch 그대로 사용)
- Optimizer 재구현 (PyTorch 그대로 사용)
- 분산 학습 프레임워크 (NCCL helper만 제공)

---

## 2. 모듈 구조

```
xpuruntime.pytorch
├── set_kernel_policy()        # 전역 커널 정책 설정
├── get_kernel_policy()        # 현재 정책 조회
├── autocast()                 # 혼합 정밀도 컨텍스트
├── capture_graph()            # CUDA Graph 캡처
├── ops/                       # 커스텀 연산자
│   ├── fused_attention()
│   ├── fused_layernorm()
│   └── ...
└── allocator                  # 커스텀 allocator
    ├── set_allocator()
    └── reset_allocator()
```

---

## 3. PyTorch Extension 통합

### 3.1 커널 정책 적용 방식

PyTorch의 dispatcher를 활용하여 특정 연산에 xpuruntime 커널을 삽입한다.

```
PyTorch Forward Pass
       │
       ▼
┌──────────────────────┐
│   torch.matmul()     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  xpuruntime dispatch │  ← KernelPolicy 확인
│  (if policy set)     │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
 cuBLAS      cuBLASLt
 (default)   (policy)
```

### 3.2 구현 방식

```cpp
// PyTorch C++ Extension
#include <torch/extension.h>
#include "xpuruntime/dispatcher.h"

// matmul 오버라이드
torch::Tensor xrt_matmul(const torch::Tensor& a, const torch::Tensor& b) {
    auto& dispatcher = xpuruntime::Dispatcher::instance();
    auto policy = dispatcher.get_policy();
    
    // 정책에 따른 커널 선택
    auto it = policy.preferences.find("matmul");
    if (it != policy.preferences.end()) {
        if (it->second == "cublasLt") {
            return cublasLt_matmul(a, b);
        }
    }
    
    // 기본 PyTorch 구현
    return torch::matmul(a, b);
}

// PyTorch dispatcher에 등록
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("matmul", xrt_matmul);
}
```

---

## 4. Python API

### 4.1 커널 정책 설정

```python
# src/python/xpuruntime/pytorch/policy.py
import torch
from typing import Optional
from ..policies import KernelPolicy
from .. import _core

# 전역 정책 저장소
_current_policy: Optional[KernelPolicy] = None

def set_kernel_policy(
    policy: Optional[KernelPolicy] = None,
    **kwargs
) -> None:
    """
    학습 중 사용할 커널 정책 설정
    
    Args:
        policy: KernelPolicy 객체
        **kwargs: op_name=kernel_name 형태의 직접 지정
    
    Examples:
        # KernelPolicy 객체 사용
        policy = KernelPolicy(matmul="cublasLt", attention="flash_v2")
        set_kernel_policy(policy)
        
        # 직접 지정
        set_kernel_policy(matmul="cublasLt", attention="flash_v2")
    """
    global _current_policy
    
    if policy is not None:
        _current_policy = policy
    elif kwargs:
        _current_policy = KernelPolicy(**kwargs)
    else:
        _current_policy = None
    
    # C++ Dispatcher에 정책 적용
    if _current_policy:
        _core.Dispatcher.instance().set_policy(_current_policy._get_internal())
    else:
        _core.Dispatcher.instance().set_policy(_core.KernelPolicy())


def get_kernel_policy() -> Optional[KernelPolicy]:
    """현재 적용된 커널 정책 반환"""
    return _current_policy


def clear_kernel_policy() -> None:
    """커널 정책 초기화"""
    set_kernel_policy(None)
```

### 4.2 혼합 정밀도 (Autocast)

```python
# src/python/xpuruntime/pytorch/autocast.py
import torch
from contextlib import contextmanager
from typing import Optional

@contextmanager
def autocast(
    dtype: torch.dtype = torch.float16,
    enabled: bool = True,
    cache_enabled: bool = True,
):
    """
    혼합 정밀도 학습 컨텍스트
    
    PyTorch의 autocast와 유사하지만, xpuruntime 커널 정책과 통합됨.
    
    Args:
        dtype: 캐스팅할 dtype (float16, bfloat16)
        enabled: autocast 활성화 여부
        cache_enabled: 캐스팅 캐시 활성화 여부
    
    Examples:
        with xrt.pytorch.autocast(dtype=torch.float16):
            y = model(x)
            loss = criterion(y, t)
            loss.backward()
    """
    if not enabled:
        yield
        return
    
    # PyTorch autocast 사용
    with torch.cuda.amp.autocast(dtype=dtype, cache_enabled=cache_enabled):
        # xpuruntime 정밀도 힌트 설정
        _core.set_precision_hint("fp16" if dtype == torch.float16 else "bf16")
        try:
            yield
        finally:
            _core.set_precision_hint("fp32")
```

### 4.3 CUDA Graph 캡처

```python
# src/python/xpuruntime/pytorch/graph.py
import torch
from contextlib import contextmanager
from typing import Optional, Callable

class CUDAGraphWrapper:
    """CUDA Graph 래퍼"""
    
    def __init__(self):
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False
    
    def capture(self, fn: Callable, *args, **kwargs):
        """함수 실행을 그래프로 캡처"""
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            fn(*args, **kwargs)
        
        torch.cuda.current_stream().wait_stream(s)
        
        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            fn(*args, **kwargs)
        
        self._captured = True
    
    def replay(self):
        """캡처된 그래프 재실행"""
        if not self._captured:
            raise RuntimeError("Graph not captured yet")
        self._graph.replay()


@contextmanager
def capture_graph(
    warmup_iters: int = 3,
    pool: Optional[torch.cuda.graphs.graph_pool_handle] = None,
):
    """
    CUDA Graph 캡처 컨텍스트
    
    캡처된 연산들을 하나의 그래프로 묶어 런치 오버헤드를 최소화한다.
    
    Args:
        warmup_iters: 캡처 전 warmup 반복 횟수
        pool: 그래프 메모리 풀 핸들
    
    Examples:
        # 기본 사용
        static_input = torch.randn(32, 3, 224, 224, device='cuda')
        static_target = torch.randint(0, 1000, (32,), device='cuda')
        
        # Warmup
        for _ in range(3):
            y = model(static_input)
            loss = criterion(y, static_target)
            loss.backward()
        
        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = model(static_input)
            loss = criterion(y, static_target)
            loss.backward()
        
        # Replay
        for data, target in dataloader:
            static_input.copy_(data)
            static_target.copy_(target)
            g.replay()
            optimizer.step()
    """
    # xpuruntime 프로파일링 마커
    from .. import _core
    _core.Profiler.instance().push_range("cuda_graph_capture")
    
    try:
        yield
    finally:
        _core.Profiler.instance().pop_range()
```

---

## 5. 커스텀 연산자 (Custom Ops)

### 5.1 Fused Attention

Flash Attention 스타일의 융합 attention 연산자.

```python
# src/python/xpuruntime/pytorch/ops/attention.py
import torch
from torch import Tensor
from typing import Optional

def fused_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Fused Multi-Head Attention
    
    Flash Attention v2 기반의 메모리 효율적인 attention 구현.
    
    Args:
        query: (batch, heads, seq_len, head_dim)
        key: (batch, heads, seq_len, head_dim)
        value: (batch, heads, seq_len, head_dim)
        attn_mask: attention mask
        dropout_p: dropout 확률
        is_causal: causal masking 여부
        scale: attention scale (default: 1/sqrt(head_dim))
    
    Returns:
        (batch, heads, seq_len, head_dim)
    """
    # 커널 정책 확인
    from ..policy import get_kernel_policy
    policy = get_kernel_policy()
    
    kernel = "flash_v2"  # default
    if policy and "attention" in policy.preferences:
        kernel = policy.preferences["attention"]
    
    if kernel == "flash_v2":
        return _flash_attention_v2(query, key, value, attn_mask, 
                                   dropout_p, is_causal, scale)
    elif kernel == "torch":
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal)
    else:
        raise ValueError(f"Unknown attention kernel: {kernel}")


def _flash_attention_v2(query, key, value, attn_mask, dropout_p, is_causal, scale):
    """Flash Attention v2 구현 (C++ 바인딩 호출)"""
    from ... import _core
    
    # C++ 커널 호출
    return _core.flash_attention_forward(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask,
        dropout_p,
        is_causal,
        scale or (1.0 / (query.shape[-1] ** 0.5)),
    )
```

### 5.2 C++ 커널 구현 (Flash Attention)

```cpp
// src/cpp/backends/cuda_raw/flash_attention.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

// Flash Attention v2 커널 (간략화)
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Tiled attention 구현
    // ...
}

torch::Tensor flash_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    c10::optional<torch::Tensor> attn_mask,
    double dropout_p,
    bool is_causal,
    double scale
) {
    // 입력 검증
    TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
    TORCH_CHECK(query.dim() == 4, "query must be 4D");
    
    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto seq_len = query.size(2);
    auto head_dim = query.size(3);
    
    // 출력 텐서 생성
    auto output = torch::empty_like(query);
    
    // 커널 런치
    dim3 blocks(batch_size, num_heads);
    dim3 threads(256);
    
    flash_attention_kernel<<<blocks, threads>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_heads, seq_len, head_dim,
        static_cast<float>(scale)
    );
    
    return output;
}

// PyTorch 바인딩
TORCH_LIBRARY(xpuruntime, m) {
    m.def("flash_attention_forward", &flash_attention_forward);
}
```

### 5.3 Fused LayerNorm

```python
# src/python/xpuruntime/pytorch/ops/layernorm.py
import torch
from torch import Tensor
from typing import Optional, List

def fused_layernorm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Fused Layer Normalization
    
    단일 커널로 mean, variance, normalize, scale, bias를 수행.
    
    Args:
        input: 입력 텐서
        normalized_shape: 정규화할 shape
        weight: scale 파라미터
        bias: bias 파라미터
        eps: numerical stability를 위한 epsilon
    
    Returns:
        정규화된 텐서
    """
    from ... import _core
    
    return _core.fused_layernorm_forward(
        input.contiguous(),
        normalized_shape,
        weight,
        bias,
        eps,
    )
```

---

## 6. 메모리 관리

### 6.1 커스텀 Allocator

```python
# src/python/xpuruntime/pytorch/allocator.py
import torch
from typing import Optional

class XpuRuntimeAllocator:
    """xpuruntime 기반 PyTorch 메모리 allocator"""
    
    def __init__(self):
        self._original_allocator = None
    
    def enable(self):
        """커스텀 allocator 활성화"""
        from .. import _core
        
        # 현재 allocator 저장
        self._original_allocator = torch.cuda.memory._get_current_allocator()
        
        # xpuruntime allocator로 교체
        _core.enable_pytorch_allocator()
    
    def disable(self):
        """원래 allocator로 복원"""
        if self._original_allocator:
            from .. import _core
            _core.disable_pytorch_allocator()


_allocator = XpuRuntimeAllocator()

def set_allocator(enabled: bool = True):
    """xpuruntime allocator 설정"""
    if enabled:
        _allocator.enable()
    else:
        _allocator.disable()

def reset_allocator():
    """allocator를 기본값으로 리셋"""
    _allocator.disable()
```

### 6.2 메모리 최적화 팁

```python
import xpuruntime.pytorch as xrt_torch

# 1. 커스텀 allocator 사용 (fragmentation 감소)
xrt_torch.set_allocator(enabled=True)

# 2. 정적 shape으로 CUDA Graph 활용
# 동적 shape이 필요 없다면 그래프 캡처로 오버헤드 제거

# 3. 불필요한 동기화 제거
# xpuruntime은 명시적 sync만 수행

# 4. Pinned memory 활용
input_pinned = torch.empty(size, pin_memory=True)
```

---

## 7. 프로파일링 통합

### 7.1 PyTorch Profiler와 통합

```python
# src/python/xpuruntime/pytorch/profiling.py
import torch
from contextlib import contextmanager

@contextmanager
def profile(
    activities=None,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
    export_chrome_trace: str = None,
):
    """
    xpuruntime + PyTorch 통합 프로파일링
    
    Examples:
        with xrt_torch.profile(export_chrome_trace="trace.json"):
            for batch in dataloader:
                output = model(batch)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    """
    from .. import _core
    
    # xpuruntime 프로파일러 활성화
    profiler = _core.Profiler.instance()
    profiler.enable()
    
    # PyTorch 프로파일러
    activities = activities or [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        yield prof
    
    # 프로파일러 비활성화
    profiler.disable()
    
    # Chrome trace 내보내기
    if export_chrome_trace:
        prof.export_chrome_trace(export_chrome_trace)
        # xpuruntime 데이터도 같이 내보내기
        profiler.export_chrome_trace(export_chrome_trace.replace(".json", "_xrt.json"))
```

---

## 8. 사용 예시

### 8.1 기본 학습 루프

```python
import torch
import xpuruntime.pytorch as xrt_torch

# 모델 정의
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# xpuruntime 커널 정책 설정
xrt_torch.set_kernel_policy(
    matmul="cublasLt",
    attention="flash_v2",
)

# 학습 루프
for epoch in range(num_epochs):
    for batch, target in dataloader:
        batch, target = batch.cuda(), target.cuda()
        
        with xrt_torch.autocast(dtype=torch.float16):
            output = model(batch)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 8.2 CUDA Graph 활용

```python
import torch
import xpuruntime.pytorch as xrt_torch

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# 정적 입출력 버퍼
static_input = torch.randn(32, 3, 224, 224, device='cuda')
static_target = torch.randint(0, 1000, (32,), device='cuda')

# Warmup
xrt_torch.set_kernel_policy(matmul="cublasLt")
for _ in range(3):
    output = model(static_input)
    loss = criterion(output, static_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Graph capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)
    loss = criterion(output, static_target)
    loss.backward()

# 학습 (Graph replay)
for batch, target in dataloader:
    static_input.copy_(batch)
    static_target.copy_(target)
    g.replay()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 9. 관련 문서

- [04_inference_module.md](./04_inference_module.md) - 추론 모듈
- [06_kernel_policy.md](./06_kernel_policy.md) - 커널 정책
- [02_cpp_core_runtime.md](./02_cpp_core_runtime.md) - C++ 코어 런타임
