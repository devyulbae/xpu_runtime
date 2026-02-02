# FindTensorRT.cmake - Optional TensorRT (stub for scaffold)
# TASK_009 will provide full implementation.
if(DEFINED ENV{TENSORRT_ROOT})
  find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS $ENV{TENSORRT_ROOT}
    PATH_SUFFIXES include
  )
  find_library(TensorRT_LIBRARY nvinfer
    HINTS $ENV{TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64
  )
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
)
if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
  set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})
endif()
