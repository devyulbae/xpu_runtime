# xpuruntime - 추론 모듈 설계

> 이 문서는 xpuruntime의 Inference 모듈 설계를 설명한다.

---

## 1. 개요

Inference 모듈은 학습된 모델을 다양한 엔진(TensorRT, ONNX Runtime 등)에서 실행할 수 있게 하는 **통합 추론 인터페이스**를 제공한다.

### 설계 목표

- **통합 API**: 엔진에 관계없이 동일한 인터페이스
- **엔진 선택**: TensorRT, ONNX Runtime, OpenVINO, QNN 중 선택
- **정책 기반**: KernelPolicy/ExecutionPolicy 적용
- **캐싱**: 컴파일된 엔진 캐시로 재시작 시간 단축

---

## 2. InferenceSession 설계

### 2.1 클래스 구조

```
InferenceSession
├── IInferenceEngine (인터페이스)
│   ├── TensorRTEngine
│   ├── OnnxRuntimeEngine
│   ├── OpenVINOEngine (Phase 2)
│   └── QNNEngine (Phase 2)
│
├── EngineRegistry
├── KernelPolicy
└── ExecutionPolicy
```

### 2.2 C++ 인터페이스

```cpp
namespace xpuruntime {

// 추론 엔진 인터페이스
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    
    // 엔진 이름
    virtual std::string name() const = 0;
    
    // 모델 로드
    virtual void load(const std::string& model_path, 
                      const SessionConfig& config) = 0;
    
    // 입출력 정보
    virtual std::vector<TensorInfo> get_input_info() const = 0;
    virtual std::vector<TensorInfo> get_output_info() const = 0;
    
    // 실행
    virtual void run(const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) = 0;
    
    // I/O 바인딩 (zero-copy)
    virtual void bind_input(const std::string& name, const Tensor& tensor) = 0;
    virtual void bind_output(const std::string& name, Tensor& tensor) = 0;
    virtual void run_with_bindings() = 0;
};

// 텐서 정보
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;  // -1 for dynamic
    std::string dtype;
};

// 세션 설정
struct SessionConfig {
    std::string device;              // "cuda:0", "npu", "cpu"
    std::string engine;              // "tensorrt", "onnxrt", "openvino", "qnn"
    KernelPolicy kernel_policy;
    ExecutionPolicy exec_policy;
    bool fp16 = false;
    bool int8 = false;
    std::string cache_dir = "";      // 엔진 캐시 디렉토리
    int max_batch_size = 1;
    size_t workspace_size = 1ULL << 30;  // 1GB default
};

// 추론 세션
class InferenceSession {
public:
    InferenceSession(const std::string& model_path, const SessionConfig& config);
    ~InferenceSession();
    
    // 입출력 정보
    std::vector<TensorInfo> get_input_info() const;
    std::vector<TensorInfo> get_output_info() const;
    
    // 실행 (딕셔너리 스타일)
    std::map<std::string, Tensor> run(const std::map<std::string, Tensor>& inputs);
    
    // 실행 (순서 기반)
    std::vector<Tensor> run(const std::vector<Tensor>& inputs);
    
    // I/O 바인딩 (고급)
    IOBinding create_io_binding();
    void run_with_io_binding(IOBinding& binding);
    
    // 세션 정보
    std::string engine_name() const;
    SessionConfig config() const;
    
private:
    std::unique_ptr<IInferenceEngine> engine_;
    SessionConfig config_;
};

}  // namespace xpuruntime
```

---

## 3. TensorRT Engine

### 3.1 설계 포인트

- ONNX → TensorRT 엔진 빌드
- 엔진 직렬화/역직렬화 (캐싱)
- Dynamic shape 지원
- FP16/INT8 양자화

### 3.2 구현

```cpp
namespace xpuruntime {

class TensorRTEngine : public IInferenceEngine {
public:
    std::string name() const override { return "tensorrt"; }
    
    void load(const std::string& model_path, 
              const SessionConfig& config) override {
        // 1. 캐시 확인
        std::string cache_path = get_cache_path(model_path, config);
        if (load_from_cache(cache_path)) {
            return;
        }
        
        // 2. ONNX → TRT 빌드
        build_engine(model_path, config);
        
        // 3. 캐시 저장
        save_to_cache(cache_path);
    }
    
    void run(const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override {
        // 입력 바인딩
        for (size_t i = 0; i < inputs.size(); ++i) {
            context_->setTensorAddress(input_names_[i].c_str(), 
                                       inputs[i].data_ptr());
        }
        
        // 출력 바인딩
        for (size_t i = 0; i < outputs.size(); ++i) {
            context_->setTensorAddress(output_names_[i].c_str(),
                                       outputs[i].data_ptr());
        }
        
        // 실행
        context_->enqueueV3(stream_);
    }
    
private:
    void build_engine(const std::string& onnx_path, const SessionConfig& config) {
        // TensorRT builder 생성
        auto builder = nvinfer1::createInferBuilder(logger_);
        auto network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        auto parser = nvonnxparser::createParser(*network, logger_);
        
        // ONNX 파싱
        parser->parseFromFile(onnx_path.c_str(), 
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
        
        // 빌더 설정
        auto builder_config = builder->createBuilderConfig();
        builder_config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE, config.workspace_size);
        
        // FP16 설정
        if (config.fp16 && builder->platformHasFastFp16()) {
            builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        // INT8 설정 (calibration 필요)
        if (config.int8 && builder->platformHasFastInt8()) {
            builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
            // calibrator 설정 필요
        }
        
        // 엔진 빌드
        engine_.reset(builder->buildEngineWithConfig(*network, *builder_config));
        context_.reset(engine_->createExecutionContext());
    }
    
    std::string get_cache_path(const std::string& model_path, 
                               const SessionConfig& config) {
        // 모델 경로 + 설정을 해시하여 캐시 경로 생성
        std::hash<std::string> hasher;
        std::string key = model_path + "_" + 
                         std::to_string(config.fp16) + "_" +
                         std::to_string(config.int8);
        size_t hash = hasher(key);
        
        return config.cache_dir + "/" + std::to_string(hash) + ".engine";
    }
    
    bool load_from_cache(const std::string& cache_path) {
        std::ifstream file(cache_path, std::ios::binary);
        if (!file) return false;
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> data(size);
        file.read(data.data(), size);
        
        auto runtime = nvinfer1::createInferRuntime(logger_);
        engine_.reset(runtime->deserializeCudaEngine(data.data(), size));
        context_.reset(engine_->createExecutionContext());
        
        return engine_ != nullptr;
    }
    
    void save_to_cache(const std::string& cache_path) {
        auto serialized = engine_->serialize();
        std::ofstream file(cache_path, std::ios::binary);
        file.write(static_cast<const char*>(serialized->data()), 
                   serialized->size());
    }
    
    nvinfer1::ILogger logger_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

}  // namespace xpuruntime
```

---

## 4. ONNX Runtime Engine

### 4.1 설계 포인트

- CUDA Execution Provider 사용
- Graph Optimization Level 설정
- TensorRT EP와 혼합 사용 가능

### 4.2 구현

```cpp
namespace xpuruntime {

class OnnxRuntimeEngine : public IInferenceEngine {
public:
    std::string name() const override { return "onnxrt"; }
    
    void load(const std::string& model_path,
              const SessionConfig& config) override {
        // 세션 옵션 설정
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // CUDA Provider 추가
        if (config.device.find("cuda") != std::string::npos) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = parse_device_id(config.device);
            cuda_options.arena_extend_strategy = 1;  // kSameAsRequested
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
        
        // TensorRT Provider (옵션)
        if (config.engine == "onnxrt+trt") {
            OrtTensorRTProviderOptions trt_options;
            trt_options.device_id = parse_device_id(config.device);
            trt_options.trt_fp16_enable = config.fp16;
            trt_options.trt_int8_enable = config.int8;
            
            session_options.AppendExecutionProvider_TensorRT(trt_options);
        }
        
        // 세션 생성
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), 
                                                   session_options);
        
        // 입출력 정보 캐시
        cache_io_info();
    }
    
    void run(const std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override {
        // Ort::Value 변환
        std::vector<Ort::Value> ort_inputs;
        for (const auto& input : inputs) {
            ort_inputs.push_back(create_ort_value(input));
        }
        
        // 실행
        auto ort_outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_ptrs_.data(), ort_inputs.data(), ort_inputs.size(),
            output_names_ptrs_.data(), output_names_ptrs_.size());
        
        // 출력 변환
        outputs.clear();
        for (auto& ort_output : ort_outputs) {
            outputs.push_back(create_tensor(ort_output));
        }
    }
    
private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "xpuruntime"};
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptrs_;
    std::vector<const char*> output_names_ptrs_;
    
    void cache_io_info() {
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 입력 정보
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.get());
            input_names_ptrs_.push_back(input_names_.back().c_str());
        }
        
        // 출력 정보
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
            output_names_ptrs_.push_back(output_names_.back().c_str());
        }
    }
};

}  // namespace xpuruntime
```

---

## 5. Python InferenceSession

### 5.1 사용자 API

```python
# src/python/xpuruntime/inference/session.py
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
from .. import _core
from ..policies import KernelPolicy, ExecutionPolicy

class InferenceSession:
    """통합 추론 세션"""
    
    def __init__(
        self,
        model: str | Path,
        device: str = "cuda:0",
        engine: str = "tensorrt",
        kernel_policy: Optional[KernelPolicy] = None,
        exec_policy: Optional[ExecutionPolicy] = None,
        fp16: bool = False,
        int8: bool = False,
        cache_dir: Optional[str] = None,
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30,
    ):
        """
        Args:
            model: ONNX 모델 경로
            device: 실행 디바이스 ("cuda:0", "npu", "cpu")
            engine: 추론 엔진 ("tensorrt", "onnxrt", "openvino", "qnn")
            kernel_policy: 커널 선택 정책
            exec_policy: 실행 정책 (NPU용)
            fp16: FP16 모드 활성화
            int8: INT8 양자화 활성화
            cache_dir: 엔진 캐시 디렉토리
            max_batch_size: 최대 배치 크기
            workspace_size: TensorRT workspace 크기 (바이트)
        """
        self._model_path = str(model)
        
        # 설정 구성
        config = _core.SessionConfig()
        config.device = device
        config.engine = engine
        config.fp16 = fp16
        config.int8 = int8
        config.cache_dir = cache_dir or ""
        config.max_batch_size = max_batch_size
        config.workspace_size = workspace_size
        
        if kernel_policy:
            config.kernel_policy = kernel_policy._get_internal()
        if exec_policy:
            config.exec_policy = exec_policy._get_internal()
        
        # 세션 생성
        self._session = _core.InferenceSession(self._model_path, config)
        self._config = config
    
    @property
    def input_info(self) -> List[Dict]:
        """입력 텐서 정보"""
        infos = self._session.get_input_info()
        return [{"name": i.name, "shape": list(i.shape), "dtype": i.dtype} 
                for i in infos]
    
    @property
    def output_info(self) -> List[Dict]:
        """출력 텐서 정보"""
        infos = self._session.get_output_info()
        return [{"name": i.name, "shape": list(i.shape), "dtype": i.dtype}
                for i in infos]
    
    @property
    def engine_name(self) -> str:
        """사용 중인 엔진 이름"""
        return self._session.engine_name()
    
    def run(
        self,
        inputs: Dict[str, np.ndarray] | List[np.ndarray],
    ) -> Dict[str, np.ndarray] | List[np.ndarray]:
        """
        추론 실행
        
        Args:
            inputs: 입력 데이터 (딕셔너리 또는 리스트)
        
        Returns:
            출력 데이터 (입력과 동일한 형식)
        """
        if isinstance(inputs, dict):
            # 딕셔너리 입력
            input_tensors = {k: _core.Tensor.from_numpy(v) 
                           for k, v in inputs.items()}
            output_tensors = self._session.run(input_tensors)
            return {k: v.to_numpy() for k, v in output_tensors.items()}
        else:
            # 리스트 입력
            input_tensors = [_core.Tensor.from_numpy(v) for v in inputs]
            output_tensors = self._session.run(input_tensors)
            return [v.to_numpy() for v in output_tensors]
    
    def __repr__(self) -> str:
        return (f"<InferenceSession model={self._model_path} "
                f"engine={self.engine_name} device={self._config.device}>")
```

### 5.2 사용 예시

```python
import xpuruntime as xrt
import numpy as np

# 기본 사용
sess = xrt.InferenceSession(
    "model.onnx",
    device="cuda:0",
    engine="tensorrt",
    fp16=True,
)

# 입출력 정보 확인
print(sess.input_info)
# [{'name': 'input', 'shape': [1, 3, 224, 224], 'dtype': 'float32'}]

# 추론 실행
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = sess.run({"input": input_data})
print(output["output"].shape)

# 커널 정책 적용
policy = xrt.KernelPolicy(
    matmul="cublasLt",
    attention="flash_v2",
)

sess = xrt.InferenceSession(
    "model.onnx",
    device="cuda:0",
    engine="onnxrt",
    kernel_policy=policy,
)
```

---

## 6. IOBinding (고급)

Zero-copy I/O를 위한 고급 인터페이스.

### 6.1 설계

```python
class IOBinding:
    """Zero-copy I/O 바인딩"""
    
    def __init__(self, session: InferenceSession):
        self._binding = session._session.create_io_binding()
    
    def bind_input(
        self,
        name: str,
        tensor: np.ndarray | "torch.Tensor",
        device: str = "cuda",
    ):
        """입력 바인딩 (GPU 메모리 직접 사용)"""
        if hasattr(tensor, 'data_ptr'):  # PyTorch Tensor
            ptr = tensor.data_ptr()
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
        else:  # NumPy array
            ptr = tensor.ctypes.data
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
        
        self._binding.bind_input(name, ptr, shape, dtype, device)
    
    def bind_output(
        self,
        name: str,
        tensor: np.ndarray | "torch.Tensor",
        device: str = "cuda",
    ):
        """출력 바인딩"""
        if hasattr(tensor, 'data_ptr'):
            ptr = tensor.data_ptr()
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
        else:
            ptr = tensor.ctypes.data
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
        
        self._binding.bind_output(name, ptr, shape, dtype, device)


# 사용 예시
import torch

sess = xrt.InferenceSession("model.onnx", engine="tensorrt")

# GPU 텐서 생성
input_tensor = torch.randn(1, 3, 224, 224, device="cuda")
output_tensor = torch.empty(1, 1000, device="cuda")

# IOBinding 사용 (zero-copy)
binding = sess.create_io_binding()
binding.bind_input("input", input_tensor)
binding.bind_output("output", output_tensor)

sess.run_with_io_binding(binding)

print(output_tensor)  # 결과가 이미 GPU 텐서에 있음
```

---

## 7. Engine Registry

### 7.1 동적 엔진 등록

```cpp
namespace xpuruntime {

class EngineRegistry {
public:
    static EngineRegistry& instance();
    
    // 엔진 등록
    void register_engine(const std::string& name,
                         std::function<std::unique_ptr<IInferenceEngine>()> factory);
    
    // 엔진 생성
    std::unique_ptr<IInferenceEngine> create_engine(const std::string& name);
    
    // 등록된 엔진 목록
    std::vector<std::string> get_available_engines() const;
    
private:
    std::map<std::string, 
             std::function<std::unique_ptr<IInferenceEngine>()>> factories_;
};

// 엔진 자동 등록 매크로
#define REGISTER_ENGINE(name, engine_class) \
    static bool _reg_engine_##engine_class = []() { \
        EngineRegistry::instance().register_engine( \
            name, []() { return std::make_unique<engine_class>(); }); \
        return true; \
    }()

// 사용
REGISTER_ENGINE("tensorrt", TensorRTEngine);
REGISTER_ENGINE("onnxrt", OnnxRuntimeEngine);

}  // namespace xpuruntime
```

---

## 8. 캐싱 전략

### 8.1 엔진 캐시

```python
# 캐시 디렉토리 구조
~/.cache/xpuruntime/
├── tensorrt/
│   ├── <model_hash>_fp16.engine
│   ├── <model_hash>_fp32.engine
│   └── <model_hash>_int8.engine
└── openvino/
    └── <model_hash>/
        ├── model.xml
        └── model.bin
```

### 8.2 캐시 키 생성

```cpp
std::string generate_cache_key(const std::string& model_path,
                               const SessionConfig& config) {
    // 다음 요소를 해시에 포함:
    // - 모델 파일 내용 해시 (또는 수정 시간)
    // - fp16/int8 설정
    // - GPU compute capability
    // - TensorRT/ORT 버전
    // - workspace_size, max_batch_size
    
    std::string key = "";
    key += get_file_hash(model_path);
    key += "_fp16=" + std::to_string(config.fp16);
    key += "_int8=" + std::to_string(config.int8);
    key += "_sm=" + std::to_string(get_sm_version());
    key += "_trt=" + get_tensorrt_version();
    
    return sha256(key);
}
```

---

## 9. 에러 처리

### 9.1 모델 로드 실패

```python
try:
    sess = xrt.InferenceSession("invalid.onnx")
except FileNotFoundError:
    print("모델 파일을 찾을 수 없습니다")
except xrt.XpuRuntimeError as e:
    print(f"모델 로드 실패: {e}")
```

### 9.2 추론 실패

```python
try:
    output = sess.run({"input": wrong_shape_input})
except xrt.XpuRuntimeError as e:
    print(f"추론 실패: {e}")
except xrt.OutOfMemoryError:
    print("GPU 메모리 부족")
```

---

## 10. 관련 문서

- [02_cpp_core_runtime.md](./02_cpp_core_runtime.md) - C++ 코어 런타임
- [03_python_binding.md](./03_python_binding.md) - Python 바인딩
- [06_kernel_policy.md](./06_kernel_policy.md) - 커널 정책
