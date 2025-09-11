cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=native_cpu")
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