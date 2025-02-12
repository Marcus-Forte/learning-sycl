cmake_minimum_required(VERSION 3.20)
project(sycl-app)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

add_compile_options(-fsycl -fsycl-targets=spir64)
add_link_options(-fsycl -lsycl)

add_executable(main src/main.cc)