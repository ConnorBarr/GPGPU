cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab3 LANGUAGES CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 62 75 86)
  # See https://developer.nvidia.com/cuda-gpus for GPU compute capabilities (CC)
  # Compilation speed will increase if you only list the CC of your own card
endif()

set(n vec-add)
#set(NAME2 stencil)
add_executable(${n} ${n}.cu)
add_executable(sync sync.cu)

#target_compile_options(${n} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_30,code=sm_30>)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${n})
