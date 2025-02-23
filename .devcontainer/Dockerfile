FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV SYCL_HOME="/opt/sycl"

WORKDIR $SYCL_HOME

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y software-properties-common clang clangd libclang-dev libclang-cpp-dev \
libboost-dev libboost-context-dev libboost-fiber-dev build-essential \ 
git python3 cmake ocl-icd-dev ocl-icd-opencl-dev

#  libnvidia-compute-535? for icd

RUN git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git -b v18.1.8 \
&& mkdir SPIRV-LLVM-Translator/build && cd SPIRV-LLVM-Translator/build \
&& cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SYCL_HOME .. \
&& make -j6 install

RUN git clone https://github.com/pocl/pocl.git && \
  mkdir pocl/build && cd pocl/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ICD=ON -DCMAKE_INSTALL_PREFIX=$SYCL_HOME .. && \
  make -j6 install

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git -b v24.10.0 && \
  mkdir AdaptiveCpp/build && cd AdaptiveCpp/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENCL_BACKEND=ON -DCMAKE_INSTALL_PREFIX=$SYCL_HOME .. && \
  make -j6 install

ENV PATH=$PATH:$SYCL_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SYCL_HOME/lib