# Using ubuntu 22 as it matches `nvcr.io/nvidia/l4t-cuda:12.2.12-devel` for Jetson Orin.

ARG BASE_IMG=nvidia/cuda:12.6.3-devel-ubuntu22.04

FROM ${BASE_IMG} AS develop

ENV DEBIAN_FRONTEND=noninteractive
ENV SYCL_HOME="/opt/sycl"
ENV PATH=$PATH:${SYCL_HOME}/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SYCL_HOME}/lib

RUN apt update && apt install -y git python3 cmake build-essential ninja-build clangd libeigen3-dev && \
  apt clean && apt autoremove && \
  rm -rf /var/lib/apt/lists/* 

RUN cd /tmp && git clone https://github.com/intel/llvm -b sycl && \ 
  python3 /tmp/llvm/buildbot/configure.py --cuda --native_cpu -t Release --cmake-opt="-DCMAKE_INSTALL_PREFIX=${SYCL_HOME}" && \
  python3 /tmp/llvm/buildbot/compile.py -t install -j8 && \
  rm -rf /tmp/llvm

# Install oneMath for SYCL + Generic BLAS. Use use CMAKE_CXX_FLAGS to compile for a number of devices already.
# https://uxlfoundation.github.io/oneMath/building_the_project_with_dpcpp.html#build-additional-options-dpcpp
# This means we override: https://github.com/uxlfoundation/oneMath/blob/develop/src/blas/backends/generic/CMakeLists.txt

# Generic BLAS use Sycl purely - as opposed to specifc GEMM implementations like cuBLAS from nvidia.
RUN cd /tmp && git clone https://github.com/uxlfoundation/oneMath.git && \
    mkdir oneMath/build && cd oneMath/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=native_cpu,nvptx64-nvidia-cuda,spir64" -DENABLE_GENERIC_BLAS_BACKEND=ON -DENABLE_MKLCPU_BACKEND=FALSE -DENABLE_MKLGPU_BACKEND=False .. && \
    cmake --build . -j4 && \
    cmake --install . --prefix ${SYCL_HOME}

COPY ./cmake/cmake-kits.json ${SYCL_HOME}/cmake/
COPY ./cmake/sycl_toolchain.cmake ${SYCL_HOME}/cmake

## For SYCL runtime, use 
#
# ENV SYCL_HOME="/opt/sycl"
# ENV PATH=$PATH:${SYCL_HOME}/bin
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SYCL_HOME}/lib

# COPY --from=mdnf1992/sycl-dev ${SYCL_HOME}/ ${SYCL_HOME}/
#


