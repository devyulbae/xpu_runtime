# 완료된 작업 로그 (Task Log)

> **기록 위치**: 태스크 완료 시 **이 파일(`docs/tasks/task_log.md`)** 에 기록합니다. **형식**: **맞닥뜨린 문제 → 해결 방법 → 얻은 효과**.  
> BE/개발자 관점에서 “무슨 상황이었고, 어떻게 풀었고, 그 결과 무엇이 좋아졌는지”를 남깁니다.

---

## 목록 (요약)

| 완료일 | Task | 작업명 |
|--------|------|--------|
| 2026-02-02 | [TASK_001](./TASK_001_project_scaffold.md) | 프로젝트 스캐폴드 |
| 2026-02-02 | [TASK_002](./TASK_002_cpp_core_skeleton.md) | C++ 코어 스켈레톤 |
| 2026-02-03 | [TASK_003](./TASK_003_python_binding.md) | Python 바인딩 기본 |

---

## TASK_001: 프로젝트 스캐폴드 (2026-02-02)

### 맞닥뜨린 문제

- **환경**: Windows, Python venv. CUDA/Visual Studio 미설치.
- **상황**: `pip install -e ".[dev]"` 시 빌드 백엔드가 scikit-build-core(CMake)를 사용하면서 **C++/CUDA 네이티브 확장**을 빌드하려 함.
- **결과**: NMake/C++/CUDA 컴파일러 미설정으로 CMake 설정 실패 → editable 설치 자체가 안 됨. `import xpuruntime` 불가, pytest도 실행 불가.

즉, “의존성(CUDA, VS) 없이는 아무것도 못 쓴다”는 상태였음.

### 해결 방법

1. **기본 빌드 백엔드를 setuptools로 변경**  
   - scikit-build-core 대신 setuptools 사용.  
   - 기본 `pip install -e .` 시 **순수 Python 패키지만** 설치 (CMake/CUDA 불필요).
2. **네이티브 확장은 선택 사항으로 유지**  
   - `xpuruntime.__init__.py`에서 `_core` import 실패 시 `ImportError` 무시하고, `get_version()`은 Python `__version__`으로 폴백.  
   - 나중에 CUDA/CMake 있는 환경에서 C++ 확장 빌드하면 그때부터 `_core` 사용 가능.
3. **패키지 레이아웃**  
   - `[tool.setuptools.packages.find]`로 `src/python` 아래 패키지 인식.  
   - README에 “기본은 순수 Python, 네이티브 빌드는 07_build_packaging 참고” 안내 추가.

### 얻은 효과

- **CUDA/Visual Studio 없이** Windows에서 `pip install -e ".[dev]"` 성공.
- `import xpuruntime`, `xpuruntime.get_version()` 동작 (현재는 Python 버전 "0.1.0" 반환).
- `pytest tests/python -v` 통과 → **스캐폴드 단계에서 개발·테스트 사이클 확보**.
- “일단 Python만으로 개발하고, 나중에 Linux/CUDA 환경에서 네이티브 확장 빌드”라는 단계적 전략을 문서·코드로 정리함.

---

## TASK_002: C++ 코어 스켈레톤 (2026-02-02)

### 맞닥뜨린 문제

- **상황**: TASK_001으로 Python 패키지·스캐폴드는 갖춰졌지만, C++ 쪽에는 `version.cpp`와 `config.hpp`만 있는 상태.
- **결과**: DeviceManager, MemoryManager, StreamManager, KernelRegistry, Dispatcher, Profiler, IBackend 등 **인터페이스·타입 정의가 없어** 이후 TASK_004~007 구현이나 Python 바인딩(TASK_003)을 진행할 수 없음. “어디에 무엇을 구현할지” 공통 계약이 없던 상태.

### 해결 방법

1. **설계 문서(02_cpp_core_runtime) 기준으로 헤더 정의**  
   - `common.h`(DeviceId, StreamHandle, EventHandle, TensorInfo), `exceptions.h`(XpuRuntimeError, CudaError, OutOfMemoryError, UnsupportedOperationError, CUDA_CHECK).  
   - `device_manager.h`, `memory_manager.h`, `stream_manager.h`, `kernel_registry.h`, `dispatcher.h`, `profiler.h`, `backends/ibackend.h`에 싱글톤/인터페이스·구조체 선언.
2. **스텁 구현**  
   - 각 매니저·레지스트리·프로파일러에 대한 `.cpp` 추가. DeviceManager는 `cudaGetDeviceCount`/`cudaGetDeviceProperties` 등으로 실제 열거(환경에 GPU 없으면 빈 목록), MemoryManager는 `cudaMalloc`/`cudaFree` 직접 호출 수준, StreamManager는 create/destroy/record/wait 등 최소 구현. KernelRegistry/Dispatcher/Profiler/BackendRegistry는 “호출 가능한 빈 구현” 수준.
3. **CMake·테스트**  
   - `xpuruntime_core`에 새 소스 파일들 추가, `tests/cpp/test_core.cpp`에서 각 `instance()` 및 `get_device_count()` 등 호출해 **컴파일·링크·실행** 가능 여부만 검증(GPU 0개일 때는 StreamManager 기본 스트림 생성 생략).

### 얻은 효과

- **컴파일 가능한 C++ 코어 스켈레톤** 확보. CUDA/CMake 있는 환경에서 `cmake .. && cmake --build .` 및 `ctest`로 “빈 구현이지만 호출 가능”함을 확인 가능.
- 이후 TASK_004~007에서 각 매니저·레지스트리 구현을 채울 때 **공통 헤더·네임스페이스·예외 계층**을 그대로 사용할 수 있음.
- Python 바인딩(TASK_003)에서 바인딩할 C++ API가 명확해짐.

---

## TASK_003: Python 바인딩 기본 (2026-02-02)

### 맞닥뜨린 문제

- **상황**: TASK_002로 C++ 코어 스켈레톤(DeviceManager, MemoryManager, 예외 등)은 있지만, Python에서 **직접 쓸 수 있는 바인딩**이 없음. `module.cpp`에는 `get_version()`만 노출된 상태.
- **결과**: Python 사용자가 `xpuruntime.XpuRuntimeError`, `DeviceManager.instance()`, `MemoryManager` 등을 쓸 수 없음. 예외 변환(C++ → Python)도 없어 C++에서 던진 예외가 Python에서 제대로 잡히지 않음.

### 해결 방법

1. **예외 바인딩**  
   - `exception_binding.cpp`: `XpuRuntimeError`, `CudaError`, `OutOfMemoryError`, `UnsupportedOperationError`를 pybind11로 등록하고, `register_exception_translator`로 C++ 예외를 Python 예외로 변환.
2. **Device / Memory 바인딩**  
   - `device_binding.cpp`: `DeviceInfo`(readonly 속성, `__repr__`), `DeviceManager`(싱글톤, nodelete holder, get_device_count 등).  
   - `memory_binding.cpp`: `MemoryType` enum, `MemoryManager::Stats`, `MemoryManager`(instance, get_allocated_size, get_cached_size, get_stats, empty_cache).
3. **모듈 통합**  
   - `module.cpp`에서 `bind_exceptions`, `bind_device`, `bind_memory` 호출 후 `get_version` 노출. CMake에 새 바인딩 소스 추가.
4. **Python 패키지**  
   - `__init__.py`: `_core` 있으면 해당 예외·클래스·함수 re-export, 없으면 **Python 폴백 예외 클래스** 정의 후 Device/Memory는 None. `get_version()`은 기존처럼 네이티브/폴백 분기.
5. **테스트**  
   - 예외 클래스 존재·raise/catch, `_core` 있을 때만 DeviceManager/MemoryManager 호출 테스트(skipif).

### 얻은 효과

- **`import xpuruntime` 후** `xpuruntime.XpuRuntimeError`, `CudaError`, `OutOfMemoryError`, `UnsupportedOperationError` 사용 가능(네이티브 빌드 시 C++ 예외가 Python으로 변환됨, 순수 Python 시 폴백 클래스).
- **네이티브 빌드 환경**에서는 `DeviceManager.instance()`, `get_device_count()`, `MemoryManager.instance()`, `get_allocated_size()` 등 호출 가능.
- **순수 Python 환경**에서도 `pytest tests/python -v` 통과(예외 테스트 + 네이티브 전용 테스트는 skip).

---

## 요약

- **Phase 1: Foundation** – TASK_001 완료 (스캐폴드 + Windows 순수 Python 경로 확보), TASK_002 완료 (C++ 코어 스켈레톤), TASK_003 완료 (Python 바인딩 기본)
