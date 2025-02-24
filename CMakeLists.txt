cmake_minimum_required(VERSION 3.20)
set(CMAKE_TOOLCHAIN_FILE cmake/sycl_toolchain.cmake)
project(sycl-app)
add_compile_options(--acpp-targets=generic)
add_executable(main src/main.cc)

# target_link_libraries(main OpenCL)