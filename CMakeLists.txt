cmake_minimum_required(VERSION 3.20)
set(CMAKE_TOOLCHAIN_FILE cmake/sycl_toolchain.cmake)
project(sycl-app)

find_package(Eigen3 REQUIRED)
link_libraries(Eigen3::Eigen)
add_compile_options(--acpp-targets=generic -O3)


add_executable(main src/main.cc)
add_executable(point3_diff src/point3_diff.cc)

