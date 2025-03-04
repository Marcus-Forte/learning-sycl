cmake_minimum_required(VERSION 3.20)
project(sycl-app)

find_package(Eigen3 REQUIRED)
# To build with AoT for CUDA (e.g for --acpp-targets=cuda:sm_70), make sure to link with Cuda and Boost.
link_libraries(Eigen3::Eigen)
add_compile_options(--acpp-targets=generic -O3)

add_executable(main src/main.cc)
add_executable(point3_diff src/point3_diff.cc)

