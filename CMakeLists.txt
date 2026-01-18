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

add_executable(onemath src/onemath.cc)

# Option to select BLAS backend for `onemath`. app Note oneMath does not allow to build both backends at the same time.
option (USE_CUBLAS_BACKEND "Use cuBLAS backend for oneMath" OFF)
option (USE_LAPACK_BACKEND "Use lapack backend for onemath" ON)
option (USE_GENERIC_BACKEND "Use generic backend for oneMath" OFF)
# TODO can we get this from oneMath somehow?
if(USE_CUBLAS_BACKEND)
    message(STATUS "Using cuBLAS backend for oneMath")
    target_compile_definitions(onemath PRIVATE USE_CUBLAS_BACKEND)
    set(BLAS_BACKEND ONEMATH::onemath_blas_cublas cuda)
elseif(USE_LAPACK_BACKEND)
    message(STATUS "Using lapack backend for oneMath")
    target_compile_definitions(onemath PRIVATE USE_LAPACK_BACKEND)
    set(BLAS_BACKEND ONEMATH::onemath_blas_netlib gfortran) # lapack
elseif(USE_GENERIC_BACKEND)
    target_compile_definitions(onemath PRIVATE USE_GENERIC_BACKEND)
    message(STATUS "Using generic backend for oneMath")
    set(BLAS_BACKEND ONEMATH::onemath_blas_generic) # generic
else()
    message(FATAL_ERROR "Please select a BLAS backend for oneMath by setting one of USE_CUBLAS_BACKEND, USE_LAPACK_BACKEND or USE_GENERIC_BACKEND options.")
endif()
target_link_libraries(onemath ONEMATH::onemath ${BLAS_BACKEND} Eigen3::Eigen)

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