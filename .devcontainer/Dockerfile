# Default Args
ARG SYCL_HOME="/opt/sycl"
ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_VER=19
ARG WITH_OPENCL=OFF

# Base Image to use. 
ARG BASE_IMG=nvcr.io/nvidia/12.6.11-devel:12.6.11-devel-aarch64-ubuntu22.04

FROM ${BASE_IMG} AS develop
# Re-use args
ARG SYCL_HOME
ARG DEBIAN_FRONTEND
ARG LLVM_VER
ARG WITH_OPENCL

ARG TARGETARCH

WORKDIR /tmp

#  llvm-spirv-${LLVM_VER}
RUN apt update && apt install -y hwloc git pkg-config lsb-release wget software-properties-common gnupg \
  clang clangd libclang-dev libclang-cpp-dev \
  libboost-dev libboost-context-dev libboost-fiber-dev build-essential \
  python3 cmake ocl-icd-opencl-dev \
  libeigen3-dev libtbb-dev && \
  apt clean && apt autoremove && \
  rm -rf /var/lib/apt/lists/* 

# Install LLVM
RUN wget https://apt.llvm.org/llvm.sh && \ 
  chmod +x llvm.sh && \
  ./llvm.sh ${LLVM_VER} && \
  apt-get install -y libclang-${LLVM_VER}-dev

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git -b v25.02.0 && \
  mkdir AdaptiveCpp/build && cd AdaptiveCpp/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENCL_BACKEND=${WITH_OPENCL} -DWITH_CUDA_BACKEND=ON -DCMAKE_INSTALL_PREFIX=${SYCL_HOME} .. && \
  make -j4 install && \
  rm -rf /tmp/AdaptiveCpp

COPY ./cmake/cmake-kits.json ${SYCL_HOME}/cmake/
COPY ./cmake/sycl_toolchain.cmake ${SYCL_HOME}/cmake

ENV PATH=$PATH:${SYCL_HOME}/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SYCL_HOME}/lib
# ENV OCL_ICD_VENDORS=/etc/${SYCL_HOME}/OpenCL/vendors/

FROM develop AS build

COPY . /tmp

# Sadly cmake cannot install libraries opened at runtime by acpp, so using RUNTIME_DEPENDENCY_SET is not enough.
RUN mkdir -p /tmp/build && cd /tmp/build && \
  cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_TOOLCHAIN_FILE=$SYCL_HOME/cmake/sycl_toolchain.cmake && \
  make -j4 && \
  make install

# Runtime container does not have bitcode library to use generic LLVMToPtx.
FROM ${BASE_IMG} AS runtime
# Re-use args
ARG SYCL_HOME
ARG DEBIAN_FRONTEND
ARG LLVM_VER

RUN apt-get update && apt-get install -y wget lsb-release \
  software-properties-common gnupg

RUN wget https://apt.llvm.org/llvm.sh && \ 
  chmod +x llvm.sh && \
  ./llvm.sh ${LLVM_VER}

COPY --from=build /usr/local/bin/* /usr/local/bin/
COPY --from=build ${SYCL_HOME}/ ${SYCL_HOME}/