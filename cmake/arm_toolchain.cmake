# toolchain-arm64.cmake

# Set the target architecture
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross-compilation tools (clang as the compiler)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# set(CMAKE_C_COMPILER clang)
# set(CMAKE_CXX_COMPILER clang++)
# set(CMAKE_C_FLAGS "--target=aarch64-linux-gnu -march=armv8-a")
# set(CMAKE_CXX_FLAGS "--target=aarch64-linux-gnu -march=armv8-a")

# Specify the path to the cross-compilation libraries and headers
# set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# Ensure that CMake will search in the cross-compilation environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
