cmake_minimum_required(VERSION 3.25)
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

# find_package(CUDAToolkit REQUIRED)
# link_libraries(CUDA::cublas CUDA::cudart)
# to test GEMM

add_executable(gemm src/gemm.cc)
target_link_libraries(gemm ONEMATH::onemath onemath_blas_generic Eigen3::Eigen )

