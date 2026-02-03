# TASK_005: MemoryManager 구현

> Phase 2: Core Features

---

## 개요

GPU 메모리 할당/해제를 관리하는 Caching Allocator를 구현한다.

## 목표

- Device memory 할당/해제
- Caching allocator (재사용을 통한 할당 오버헤드 감소)
- Pinned memory 지원
- 메모리 통계 수집

## 선행 작업

- TASK_004: DeviceManager 구현

## 작업 내용

### 1. Caching Allocator 구현

```cpp
// src/cpp/core/memory_manager.cpp

void* MemoryManager::allocate(size_t size, MemoryType type, int device_id) {
    if (device_id < 0) {
        device_id = DeviceManager::instance().get_current_device();
    }
    
    auto& pool = device_pools_[device_id];
    std::lock_guard<std::mutex> lock(pool.mutex);
    
    // 블록 크기 정규화 (2의 거듭제곱으로)
    size_t block_size = round_up_power_of_2(size);
    
    // 캐시에서 검색
    auto it = pool.free_blocks.find(block_size);
    if (it != pool.free_blocks.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        pool.allocated_blocks[ptr] = block_size;
        return ptr;
    }
    
    // 새로 할당
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, block_size));
    pool.allocated_blocks[ptr] = block_size;
    
    return ptr;
}

void MemoryManager::deallocate(void* ptr) {
    // 캐시에 보관 (즉시 해제하지 않음)
    // ...
}
```

### 2. 캐시 관리

```cpp
void MemoryManager::empty_cache(int device_id) {
    // 모든 캐시된 블록 해제
}

void MemoryManager::trim_cache(size_t target_size, int device_id) {
    // 지정된 크기까지만 캐시 유지
}
```

### 3. Pinned Memory

```cpp
void* MemoryManager::allocate_pinned(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}
```

### 4. 통계

```cpp
Stats MemoryManager::get_stats(int device_id) const {
    Stats stats;
    stats.total_allocated = ...;
    stats.total_cached = ...;
    stats.peak_allocated = ...;
    return stats;
}
```

## 완료 조건

- [ ] 기본 할당/해제 동작
- [ ] 캐싱 동작 확인 (같은 크기 재요청 시 기존 포인터 반환)
- [ ] Pinned memory 할당/해제
- [ ] 통계 정상 수집
- [ ] 메모리 누수 없음 (valgrind/compute-sanitizer 확인)

## 예상 소요 시간

6-8시간

## 관련 문서

- [02_cpp_core_runtime.md](../plans/02_cpp_core_runtime.md)
