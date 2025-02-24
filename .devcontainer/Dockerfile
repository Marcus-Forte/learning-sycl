FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV SYCL_HOME="/opt/sycl"

WORKDIR /tmp

ARG DEBIAN_FRONTEND=noninteractive

ARG LLVM_VER=18

RUN apt update && apt install -y git pkg-config software-properties-common \
clang clangd libclang-dev libclang-cpp-dev \
libboost-dev libboost-context-dev libboost-fiber-dev build-essential llvm-spirv-${LLVM_VER} \
python3 cmake ocl-icd-opencl-dev

#  libnvidia-compute-535? for icd

RUN git clone https://github.com/pocl/pocl.git && \
  mkdir pocl/build && cd pocl/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DENABLE_ICD=ON -DCMAKE_INSTALL_PREFIX=$SYCL_HOME -DLLC_HOST_CPU=apple-m3 .. && \
  make -j6 install

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git -b v24.10.0 && \
  mkdir AdaptiveCpp/build && cd AdaptiveCpp/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENCL_BACKEND=ON -DCMAKE_INSTALL_PREFIX=$SYCL_HOME .. && \
  make -j6 install

ENV PATH=$PATH:$SYCL_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SYCL_HOME/lib