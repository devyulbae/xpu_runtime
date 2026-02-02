# TASK_002: C++ 코어 스켈레톤

> Phase 1: Foundation

---

## 개요

C++ 코어 런타임의 기본 구조와 인터페이스를 정의한다.

## 목표

- 핵심 컴포넌트 헤더 파일 작성
- 인터페이스/추상 클래스 정의
- 기본 예외 클래스 정의
- 빈 구현체 스텁 작성

## 선행 작업

- TASK_001: 프로젝트 스캐폴드

## 작업 내용

### 1. 헤더 파일 생성

```
src/cpp/include/xpuruntime/
├── common.h              # 공통 타입, 매크로
├── exceptions.h          # 예외 클래스들
├── device_manager.h      # DeviceManager
├── memory_manager.h      # MemoryManager
├── stream_manager.h      # StreamManager
├── kernel_registry.h     # KernelRegistry, IKernel
├── dispatcher.h          # Dispatcher, KernelPolicy
├── profiler.h            # Profiler
└── backends/
    └── ibackend.h        # IBackend 인터페이스
```

### 2. 핵심 타입 정의

```cpp
// common.h
namespace xpuruntime {
    using DeviceId = int;
    using StreamHandle = cudaStream_t;
    using EventHandle = cudaEvent_t;
    
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        std::string dtype;
    };
}
```

### 3. 예외 클래스

```cpp
// exceptions.h
class XpuRuntimeError : public std::runtime_error;
class CudaError : public XpuRuntimeError;
class OutOfMemoryError : public XpuRuntimeError;
class UnsupportedOperationError : public XpuRuntimeError;
```

### 4. 각 매니저 인터페이스

- DeviceManager: 싱글톤, 디바이스 열거/조회
- MemoryManager: 싱글톤, 할당/해제
- StreamManager: 싱글톤, 스트림/이벤트 관리

### 5. 스텁 구현

```cpp
// device_manager.cpp
DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

int DeviceManager::get_device_count() const {
    // TODO: 구현
    return 0;
}
```

## 완료 조건

- [ ] 모든 헤더 파일 작성됨
- [ ] 컴파일 오류 없음
- [ ] 기본 테스트 통과 (빈 구현이지만 호출 가능)

## 예상 소요 시간

4-6시간

## 관련 문서

- [02_cpp_core_runtime.md](../02_cpp_core_runtime.md)
