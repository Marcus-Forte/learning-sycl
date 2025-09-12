cmake_minimum_required(VERSION 3.22)
project(sycl-app)

find_package(oneMath REQUIRED)
find_package(Eigen3 REQUIRED)
add_executable(primes src/primes.cc)

add_library(NBody
src/n_body/NBodyCPU.cc
src/n_body/NBodyGPU.cc)
add_executable(n_body_sim src/n_body_sim.cc)
target_link_libraries(n_body_sim NBody)


add_executable(gemm src/gemm.cc)
target_link_libraries(gemm ONEMATH::onemath onemath_blas_generic Eigen3::Eigen)

# Test agains direct cuda calls
set(CUDAToolkit_ROOT CUDAToolkit_ROOT=/usr/local/cuda-12.6)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    find_package(cuBLAS REQUIRED)
    message(STATUS "CUDAToolkit found")
    add_executable(gemm_cublas src/gemm_cublas.cc)
    target_link_libraries(gemm_cublas CUDA::cublas CUDA::cudart Eigen3::Eigen)
    set_property(TARGET gemm_cublas PROPERTY COMPILE_OPTIONS "") # Remove SYCL options
else()
    message(STATUS "CUDAToolkit not found, skipping gemm_cublas target")
endif()