# TASK_004: DeviceManager 구현

> Phase 2: Core Features

---

## 개요

GPU 디바이스 탐지 및 관리 기능을 구현한다.

## 목표

- CUDA Runtime API를 사용한 디바이스 열거
- 디바이스 정보 조회 (이름, 메모리, compute capability 등)
- 현재 디바이스 설정/조회
- 동기화 기능

## 선행 작업

- TASK_003: Python 바인딩 기본

## 작업 내용

### 1. C++ 구현

```cpp
// src/cpp/core/device_manager.cpp

DeviceManager::DeviceManager() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        DeviceInfo info;
        info.device_id = i;
        info.name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.sm_count = prop.multiProcessorCount;
        // ...
        
        devices_.push_back(info);
    }
}

int DeviceManager::get_device_count() const {
    return static_cast<int>(devices_.size());
}

void DeviceManager::set_current_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    current_device_ = device_id;
}
```

### 2. Python 바인딩 완성

```cpp
py::class_<DeviceManager, ...>(m, "DeviceManager")
    .def_static("instance", &DeviceManager::instance, ...)
    .def("get_device_count", &DeviceManager::get_device_count)
    .def("get_all_devices", &DeviceManager::get_all_devices)
    // ...
```

### 3. Python 래퍼

```python
# src/python/xpuruntime/runtime/device.py
class Device:
    def __init__(self, device_id: int = 0):
        # ...

def get_device_count() -> int:
    return _core.DeviceManager.instance().get_device_count()
```

### 4. 테스트

```cpp
// tests/cpp/test_device_manager.cpp
TEST_F(DeviceManagerTest, GetDeviceCount) { ... }
TEST_F(DeviceManagerTest, GetDeviceInfo) { ... }
```

```python
# tests/python/unit/test_device.py
def test_get_device_count(): ...
def test_device_properties(): ...
```

## 완료 조건

- [ ] `xrt.get_device_count()` 정상 동작
- [ ] `xrt.Device(0)` 생성 및 속성 접근 가능
- [ ] `xrt.set_current_device()` 동작
- [ ] C++ 테스트 통과
- [ ] Python 테스트 통과

## 예상 소요 시간

4-6시간

## 관련 문서

- [02_cpp_core_runtime.md](../02_cpp_core_runtime.md)
