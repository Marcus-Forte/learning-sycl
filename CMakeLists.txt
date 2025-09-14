cmake_minimum_required(VERSION 3.22)
project(sycl-app)

find_package(oneMath REQUIRED)
find_package(Eigen3 REQUIRED)

set(SYCL_OPTIONS -fsycl -fsycl-targets=nvptx64-nvidia-cuda,native_cpu)
add_compile_options(${SYCL_OPTIONS})
add_link_options(${SYCL_OPTIONS})
add_executable(primes src/primes.cc)

add_library(NBody
src/n_body/NBodyCPU.cc
src/n_body/NBodyGPU.cc)
add_executable(n_body_sim src/n_body_sim.cc)
target_link_libraries(n_body_sim NBody)

add_executable(gemm src/gemm.cc)

# Option to select BLAS backend for `gemm`. Note oneMath does not allow to build both backends at the same time.
option (USE_CUBLAS_BACKEND "Use cuBLAS backend for oneMath" OFF)
if(USE_CUBLAS_BACKEND)
    message(STATUS "Using cuBLAS backend for oneMath")
    target_compile_definitions(gemm PRIVATE USE_CUBLAS_BACKEND)
    set(BLAS_BACKEND onemath_blas_cublas)
else()
    message(STATUS "Using generic backend for oneMath")
    set(BLAS_BACKEND onemath_blas_generic)
endif()
target_link_libraries(gemm ONEMATH::onemath ${BLAS_BACKEND} Eigen3::Eigen)

# Build an app to use GEMM directly from CUDA/cuBLAS
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found")
    add_executable(gemm_cublas src/gemm_cublas.cc)
    target_link_libraries(gemm_cublas CUDA::cublas CUDA::cudart Eigen3::Eigen)
    set_property(TARGET gemm_cublas PROPERTY COMPILE_OPTIONS "") # Remove SYCL options
else()
    message(STATUS "CUDAToolkit not found, skipping gemm_cublas target")
endif()