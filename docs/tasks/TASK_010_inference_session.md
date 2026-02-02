# TASK_010: InferenceSession 구현

> Phase 3: Inference

---

## 개요

통합 추론 인터페이스인 InferenceSession을 완성한다.

## 목표

- 엔진 추상화 완성
- EngineRegistry 구현
- Python InferenceSession API 완성
- IOBinding 고급 기능

## 선행 작업

- TASK_009: TensorRT 통합

## 작업 내용

### 1. EngineRegistry

```cpp
// src/cpp/core/engine_registry.cpp

class EngineRegistry {
public:
    static EngineRegistry& instance() {
        static EngineRegistry registry;
        return registry;
    }
    
    void register_engine(
        const std::string& name,
        std::function<std::unique_ptr<IInferenceEngine>()> factory) {
        factories_[name] = std::move(factory);
    }
    
    std::unique_ptr<IInferenceEngine> create_engine(const std::string& name) {
        auto it = factories_.find(name);
        if (it == factories_.end()) {
            throw UnsupportedOperationError("Unknown engine: " + name);
        }
        return it->second();
    }
    
private:
    std::map<std::string, 
             std::function<std::unique_ptr<IInferenceEngine>()>> factories_;
};
```

### 2. InferenceSession C++ 구현

```cpp
// src/cpp/core/inference_session.cpp

InferenceSession::InferenceSession(const std::string& model_path,
                                   const SessionConfig& config)
    : config_(config) {
    
    // 엔진 생성
    engine_ = EngineRegistry::instance().create_engine(config.engine);
    
    // 모델 로드
    engine_->load(model_path, config);
}

std::map<std::string, Tensor> InferenceSession::run(
    const std::map<std::string, Tensor>& inputs) {
    
    // 입력 변환
    std::vector<Tensor> input_tensors;
    for (const auto& info : engine_->get_input_info()) {
        auto it = inputs.find(info.name);
        if (it == inputs.end()) {
            throw XpuRuntimeError("Missing input: " + info.name);
        }
        input_tensors.push_back(it->second);
    }
    
    // 출력 준비
    std::vector<Tensor> output_tensors;
    for (const auto& info : engine_->get_output_info()) {
        output_tensors.push_back(allocate_tensor(info));
    }
    
    // 실행
    engine_->run(input_tensors, output_tensors);
    
    // 결과 매핑
    std::map<std::string, Tensor> result;
    auto output_info = engine_->get_output_info();
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        result[output_info[i].name] = std::move(output_tensors[i]);
    }
    
    return result;
}
```

### 3. Python API 완성

```python
# src/python/xpuruntime/inference/session.py

class InferenceSession:
    def __init__(
        self,
        model: str | Path,
        device: str = "cuda:0",
        engine: str = "tensorrt",
        kernel_policy: Optional[KernelPolicy] = None,
        fp16: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        # C++ 세션 생성
        config = _core.SessionConfig()
        config.device = device
        config.engine = engine
        config.fp16 = fp16
        # ...
        
        self._session = _core.InferenceSession(str(model), config)
    
    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # ...
```

### 4. IOBinding

```python
class IOBinding:
    def bind_input(self, name: str, tensor, device: str = "cuda"):
        # ...
    
    def bind_output(self, name: str, tensor, device: str = "cuda"):
        # ...

# 사용 예시
binding = sess.create_io_binding()
binding.bind_input("input", gpu_tensor)
binding.bind_output("output", output_tensor)
sess.run_with_io_binding(binding)
```

## 완료 조건

- [ ] `InferenceSession("model.onnx", engine="tensorrt")` 동작
- [ ] `InferenceSession("model.onnx", engine="onnxrt")` 동작
- [ ] `sess.run({"input": data})` 동작
- [ ] IOBinding zero-copy 동작
- [ ] 에러 처리 (잘못된 입력, 없는 엔진 등)

## 예상 소요 시간

8-10시간

## 관련 문서

- [04_inference_module.md](../04_inference_module.md)
