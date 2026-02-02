# TASK_009: TensorRT 통합

> Phase 3: Inference

---

## 개요

TensorRT를 xpuruntime의 inference 백엔드로 통합한다.

## 목표

- ONNX → TensorRT 엔진 빌드
- 엔진 직렬화/캐싱
- FP16/INT8 지원
- Dynamic shape 지원

## 선행 작업

- TASK_008: ONNX Runtime 통합

## 작업 내용

### 1. TensorRT Engine 클래스

```cpp
// src/cpp/backends/tensorrt/trt_engine.cpp

class TensorRTEngine : public IInferenceEngine {
public:
    std::string name() const override { return "tensorrt"; }
    
    void load(const std::string& model_path,
              const SessionConfig& config) override {
        // 캐시 확인
        std::string cache_path = get_cache_path(model_path, config);
        if (load_from_cache(cache_path)) {
            return;
        }
        
        // ONNX → TRT 빌드
        build_engine(model_path, config);
        
        // 캐시 저장
        save_to_cache(cache_path);
    }
    
    void run(const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override {
        // 입출력 바인딩 및 실행
        for (size_t i = 0; i < inputs.size(); ++i) {
            context_->setTensorAddress(input_names_[i].c_str(),
                                       inputs[i].data_ptr());
        }
        
        for (size_t i = 0; i < outputs.size(); ++i) {
            context_->setTensorAddress(output_names_[i].c_str(),
                                       outputs[i].data_ptr());
        }
        
        context_->enqueueV3(stream_);
    }
    
private:
    void build_engine(const std::string& onnx_path, 
                      const SessionConfig& config);
    bool load_from_cache(const std::string& path);
    void save_to_cache(const std::string& path);
    
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;
};
```

### 2. 엔진 빌드

```cpp
void TensorRTEngine::build_engine(const std::string& onnx_path,
                                   const SessionConfig& config) {
    auto builder = nvinfer1::createInferBuilder(logger_);
    auto network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    
    auto parser = nvonnxparser::createParser(*network, logger_);
    parser->parseFromFile(onnx_path.c_str(), 
                          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    
    auto builder_config = builder->createBuilderConfig();
    builder_config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE, config.workspace_size);
    
    // FP16 설정
    if (config.fp16 && builder->platformHasFastFp16()) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    engine_.reset(builder->buildEngineWithConfig(*network, *builder_config));
    context_.reset(engine_->createExecutionContext());
}
```

### 3. 캐싱

```cpp
std::string TensorRTEngine::get_cache_path(const std::string& model_path,
                                            const SessionConfig& config) {
    // 해시 기반 캐시 키 생성
    std::string key = model_path + "_" +
                     std::to_string(config.fp16) + "_" +
                     std::to_string(config.int8) + "_" +
                     get_trt_version();
    
    return config.cache_dir + "/" + sha256(key).substr(0, 16) + ".engine";
}
```

### 4. CMake 설정

```cmake
find_package(TensorRT REQUIRED)

add_library(xpuruntime_tensorrt STATIC
    trt_engine.cpp
    trt_logger.cpp
    trt_utils.cpp
)

target_link_libraries(xpuruntime_tensorrt PRIVATE
    ${TensorRT_LIBRARIES}
    nvinfer
    nvonnxparser
)
```

## 완료 조건

- [ ] ONNX → TRT 빌드 성공
- [ ] 엔진 캐시 저장/로드 동작
- [ ] FP16 추론 동작
- [ ] 추론 결과 정확성 검증 (ORT 결과와 비교)

## 예상 소요 시간

10-14시간

## 관련 문서

- [04_inference_module.md](../04_inference_module.md)
