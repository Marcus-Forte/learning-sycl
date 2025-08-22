##

ARG BASE_IMG=nvidia/cuda:12.6.3-devel-ubuntu24.04

FROM ${BASE_IMG} AS develop

ENV DEBIAN_FRONTEND=noninteractive
ENV SYCL_HOME="/opt/sycl"
ENV PATH=$PATH:${SYCL_HOME}/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SYCL_HOME}/lib

RUN apt update && apt install -y git python3 cmake build-essential ninja-build clangd && \
  apt clean && apt autoremove && \
  rm -rf /var/lib/apt/lists/* 

RUN cd /tmp && git clone https://github.com/intel/llvm -b sycl 
RUN cd /tmp && python3 /tmp/llvm/buildbot/configure.py --cuda --native_cpu -t Release
RUN cd /tmp && python3 /tmp/llvm/buildbot/compile.py -t install -j8

RUN cp -r /tmp/llvm/build/install ${SYCL_HOME}/ && rm -rf /tmp/llvm

COPY ./cmake/cmake-kits.json ${SYCL_HOME}/cmake/
COPY ./cmake/sycl_toolchain.cmake ${SYCL_HOME}/cmake

## 

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS runtime

ENV SYCL_HOME="/opt/sycl"
ENV PATH=$PATH:${SYCL_HOME}/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SYCL_HOME}/lib

COPY --from=develop ${SYCL_HOME}/ ${SYCL_HOME}/