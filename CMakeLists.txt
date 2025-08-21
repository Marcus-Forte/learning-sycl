cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda,native_cpu,spir64")
project(sycl-app)

# find_package(Eigen3 REQUIRED)
# link_libraries(Eigen3::Eigen)
# add_executable(point3_diff src/point3_diff.cc)

add_executable(primes src/primes.cc)

add_library(NBody
src/n_body/NBodyCPU.cc
src/n_body/NBodyGPU.cc)
add_executable(n_body_sim src/n_body_sim.cc)
target_link_libraries(n_body_sim NBody)

include(GNUInstallDirs)

