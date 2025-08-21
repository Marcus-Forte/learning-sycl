# Base Image to use. 
# Options: 
# - nvidia/cuda:12.6.3-devel-ubuntu24.04
# - nvcr.io/nvidia/12.6.11-devel:12.6.11-devel-aarch64-ubuntu22.04

ARG BASE_IMG=nvidia/cuda:12.6.3-devel-ubuntu24.04

FROM ${BASE_IMG} AS develop
ENV DEBIAN_FRONTEND=noninteractive
# Re-use args
ARG DEBIAN_FRONTEND
ENV SYCL_HOME="/opt/sycl"

WORKDIR ${SYCL_HOME}

RUN apt update && apt install -y git python3 cmake build-essential ninja-build clangd && \
  apt clean && apt autoremove && \
  rm -rf /var/lib/apt/lists/* 

RUN git clone https://github.com/intel/llvm -b sycl 

RUN python3 ${SYCL_HOME}/llvm/buildbot/configure.py --cuda --native_cpu -t Release && \
  python3 ${SYCL_HOME}/llvm/buildbot/compile.py

COPY ./cmake/cmake-kits.json ${SYCL_HOME}/cmake/
COPY ./cmake/sycl_toolchain.cmake ${SYCL_HOME}/cmake

ENV PATH=$PATH:${SYCL_HOME}/llvm/build/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SYCL_HOME}/llvm/build/lib
