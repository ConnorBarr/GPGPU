cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab2 LANGUAGES CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cuda_project1 main.cpp)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT cuda_project1)