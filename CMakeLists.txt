cmake_minimum_required(VERSION 3.20)
project(sycl-app)

add_compile_options(-fsycl)
add_link_options(-fsycl)

add_compile_options(--target=aarch64-linux-gnu)
# add_link_options(--target=aarch64-linux-gnu)


add_executable(main src/main.cc)

# /opt/sycl/llvm/build/bin/clang++ --target=aarch64-linux-gnu -march=armv8-a -g CMakeFiles/main.dir/src/main.cc.o -o main -fsycl -L /opt/sycl/llvm/build-arm/lib