# FindOnnxRuntime.cmake - Optional ONNX Runtime (stub for scaffold)
# TASK_008 will provide full implementation.
if(DEFINED ENV{ONNXRUNTIME_ROOT})
  find_path(OnnxRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS $ENV{ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include
  )
  find_library(OnnxRuntime_LIBRARY onnxruntime
    HINTS $ENV{ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64
  )
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
  REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR
)
if(OnnxRuntime_FOUND)
  set(OnnxRuntime_INCLUDE_DIRS ${OnnxRuntime_INCLUDE_DIR})
  set(OnnxRuntime_LIBRARIES ${OnnxRuntime_LIBRARY})
endif()
