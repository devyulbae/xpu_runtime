# xpuruntime C++ core (TASK_002: skeleton + Python binding)

add_library(xpuruntime_core SHARED
  core/version.cpp
  core/exceptions.cpp
  core/device_manager.cpp
  core/memory_manager.cpp
  core/stream_manager.cpp
  core/kernel_registry.cpp
  core/dispatcher.cpp
  core/profiler.cpp
  core/backend_registry.cpp
)

target_include_directories(xpuruntime_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(xpuruntime_core PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cudnn
)

if(XRT_BUILD_PYTHON)
  pybind11_add_module(_core bindings/module.cpp)
  target_link_libraries(_core PRIVATE xpuruntime_core)
  target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  install(TARGETS _core LIBRARY DESTINATION xpuruntime RUNTIME DESTINATION xpuruntime ARCHIVE DESTINATION xpuruntime)
endif()

install(TARGETS xpuruntime_core
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/xpuruntime DESTINATION include)
