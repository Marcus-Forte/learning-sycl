FROM ubuntu:latest as x64

ENV DPCPP_HOME="/opt/sycl"

WORKDIR $DPCPP_HOME

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y build-essential git hwloc python3 cmake software-properties-common g++-aarch64-linux-gnu

RUN git clone https://github.com/intel/llvm -b sycl

RUN python3 $DPCPP_HOME/llvm/buildbot/configure.py -t Release --host-target "AArch64;ARM;X86" --native_cpu --cmake-gen "Unix Makefiles" && \
    python3 $DPCPP_HOME/llvm/buildbot/compile.py -j20 -t install

ENV PATH=$PATH:$DPCPP_HOME/llvm/build/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_HOME/llvm/build/lib

# Cross-compile / Build SYCL runtime for aarch64.
FROM x64 as aarch64
COPY cmake/arm_toolchain.cmake arm_toolchain.cmake
RUN CMAKE_TOOLCHAIN_FILE=/opt/sycl/arm_toolchain.cmake python3 $DPCPP_HOME/llvm/buildbot/configure.py -t Release --native_cpu --cmake-gen "Unix Makefiles" -o build-arm && \
    python3 $DPCPP_HOME/llvm/buildbot/compile.py -o build-arm -j20 -t sycl

