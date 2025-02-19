FROM --platform=linux/arm64 ubuntu:plucky AS host_arch

ENV SYCL_HOME="/opt/sycl"

WORKDIR $SYCL_HOME

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y software-properties-common clang clangd libclang-dev libclang-cpp-dev \
libboost-dev libboost-context-dev libboost-fiber-dev build-essential \ 
git python3 cmake ocl-icd-dev ocl-icd-opencl-dev

RUN git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git -b v19.1.4 \ 
&& mkdir SPIRV-LLVM-Translator/build && cd SPIRV-LLVM-Translator/build \
&& cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SYCL_HOME .. \
&& make -j6 install

RUN git clone https://github.com/pocl/pocl.git && \
  mkdir pocl/build && cd pocl/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SYCL_HOME -DLLC_HOST_CPU=apple-m3 .. && \
  make -j6 install

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git -b v24.10.0 && \
  mkdir AdaptiveCpp/build && cd AdaptiveCpp/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENCL_BACKEND=ON -DOpenCL_LIBRARY=/opt/sycl/lib/libOpenCL.so --DCMAKE_INSTALL_PREFIX=$SYCL_HOME .. && \
  make -j6 install

ENV PATH=$PATH:$SYCL_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SYCL_HOME/lib

# Cross-compile / Build SYCL runtime for aarch64.
# FROM host_arch as aarch64
# COPY cmake/arm_toolchain.cmake arm_toolchain.cmake
# RUN CMAKE_TOOLCHAIN_FILE=/opt/sycl/arm_toolchain.cmake python3 $DPCPP_HOME/llvm/buildbot/configure.py -t Release --native_cpu --cmake-gen "Unix Makefiles" -o build-arm && \
#     python3 $DPCPP_HOME/llvm/buildbot/compile.py -o build-arm -j20 -t sycl

