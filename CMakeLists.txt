cmake_minimum_required(VERSION 3.20)
project(sycl-app)

find_package(Eigen3 REQUIRED)

# NVIDIA contaienrs may only work with JIT if they inherit nvidia/cuda:12.8.0-devel*.
# It is possible to use nvidia/cuda:12.8.0-runtime*, but only with AoT compilation.
# To build with AoT for CUDA (e.g for --acpp-targets=cuda:sm_70), make sure to link with Cuda and Boost.
# Uncomment below for CUDA AOT.
# find_package(CUDAToolkit REQUIRED)
# find_package(Boost REQUIRED COMPONENTS fiber)
# link_libraries(CUDA::cudart Boost::fiber)

link_libraries(Eigen3::Eigen)
add_compile_options(--acpp-targets=generic -O3)

add_executable(main src/main.cc)
add_executable(point3_diff src/point3_diff.cc)

