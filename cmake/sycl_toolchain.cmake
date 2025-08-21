# Set the target architecture
set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross-compilation tools (clang as the compiler)
set(CMAKE_C_COMPILER /opt/sycl/llvm/build/bin/clang)
set(CMAKE_CXX_COMPILER /opt/sycl/llvm/build/bin/clang++)

# Ensure that CMake will search in the cross-compilation environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
