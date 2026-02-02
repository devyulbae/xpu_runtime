cmake_minimum_required(VERSION 3.24)
project(xpuruntime VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

option(XRT_BUILD_TESTS "Build tests" ON)
option(XRT_BUILD_PYTHON "Build Python bindings" ON)
option(XRT_USE_TENSORRT "Enable TensorRT backend" OFF)
option(XRT_USE_ONNXRUNTIME "Enable ONNX Runtime backend" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED COMPONENTS cublas cudnn)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

if(XRT_USE_TENSORRT)
  find_package(TensorRT QUIET)
endif()
if(XRT_USE_ONNXRUNTIME)
  find_package(OnnxRuntime QUIET)
endif()

if(XRT_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
  endif()
endif()

add_subdirectory(src/cpp)

if(XRT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
