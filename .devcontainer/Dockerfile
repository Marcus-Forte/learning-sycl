FROM ubuntu:latest AS host_arch

ENV SYCL_HOME="/opt/sycl"

WORKDIR $SYCL_HOME

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y software-properties-common clang libclang-dev libboost-dev libboost-context-dev libboost-fiber-dev  build-essential git python3 cmake g++-aarch64-linux-gnu

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git && \
  mkdir AdaptiveCpp/build && cd AdaptiveCpp/build && \
  cmake -DCMAKE_INSTALL_PREFIX=$SYCL_HOME .. && \
  make -j6 install

ENV PATH=$PATH:$SYCL_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SYCL_HOME/lib

# Cross-compile / Build SYCL runtime for aarch64.
# FROM host_arch as aarch64
# COPY cmake/arm_toolchain.cmake arm_toolchain.cmake
# RUN CMAKE_TOOLCHAIN_FILE=/opt/sycl/arm_toolchain.cmake python3 $DPCPP_HOME/llvm/buildbot/configure.py -t Release --native_cpu --cmake-gen "Unix Makefiles" -o build-arm && \
#     python3 $DPCPP_HOME/llvm/buildbot/compile.py -o build-arm -j20 -t sycl

