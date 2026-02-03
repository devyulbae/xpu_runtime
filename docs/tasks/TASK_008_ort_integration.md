# TASK_008: ONNX Runtime 통합

> Phase 3: Inference

---

## 개요

ONNX Runtime을 xpuruntime의 inference 백엔드로 통합한다.

## 목표

- ONNX Runtime C++ API 래핑
- CUDA Execution Provider 설정
- 세션 생성/실행 구현
- 입출력 바인딩

## 선행 작업

- TASK_007: KernelRegistry 구현

## 작업 내용

### 1. ORT Engine 클래스

```cpp
// src/cpp/backends/onnxruntime/ort_engine.cpp

class OnnxRuntimeEngine : public IInferenceEngine {
public:
    std::string name() const override { return "onnxrt"; }
    
    void load(const std::string& model_path, 
              const SessionConfig& config) override {
        // 세션 옵션 설정
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        
        // CUDA Provider 추가
        if (config.device.find("cuda") != std::string::npos) {
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = parse_device_id(config.device);
            options.AppendExecutionProvider_CUDA(cuda_opts);
        }
        
        // 세션 생성
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
        
        // 입출력 정보 캐시
        cache_io_info();
    }
    
    void run(const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override {
        // Ort::Value 변환 및 실행
    }
    
private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "xpuruntime"};
    std::unique_ptr<Ort::Session> session_;
};

REGISTER_ENGINE("onnxrt", OnnxRuntimeEngine);
```

### 2. CMake 설정

```cmake
# src/cpp/backends/onnxruntime/CMakeLists.txt

find_package(OnnxRuntime REQUIRED)

add_library(xpuruntime_onnxrt STATIC
    ort_engine.cpp
    ort_utils.cpp
)

target_link_libraries(xpuruntime_onnxrt PRIVATE
    OnnxRuntime::OnnxRuntime
)
```

### 3. 입출력 변환

```cpp
Ort::Value tensor_to_ort_value(const Tensor& tensor) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    return Ort::Value::CreateTensor(
        memory_info,
        tensor.data_ptr(),
        tensor.numel() * sizeof(float),
        tensor.shape().data(),
        tensor.shape().size()
    );
}
```

### 4. 테스트

```python
def test_ort_inference():
    sess = xrt.InferenceSession(
        "model.onnx",
        device="cuda:0",
        engine="onnxrt",
    )
    
    output = sess.run({"input": input_data})
    assert output is not None
```

## 완료 조건

- [ ] ONNX 모델 로드 가능
- [ ] CUDA EP로 추론 실행
- [ ] 입출력 형식 정확
- [ ] 에러 처리 (모델 없음, 잘못된 입력 등)

## 예상 소요 시간

8-12시간

## 관련 문서

- [04_inference_module.md](../plans/04_inference_module.md)
