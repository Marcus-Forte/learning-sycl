# Jetson containers: https://developer.nvidia.com/embedded/jetson-cloud-native
# Nvidia Containers: https://hub.docker.com/r/nvidia/cuda

ARG BASE_IMG=nvidia/cuda:12.6.1-devel-ubuntu24.04

FROM ${BASE_IMG} AS sycl

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

FROM sycl AS onemath_generic
# Install oneMath + Generic BLAS (Pure Sycl). Use CMAKE_CXX_FLAGS to compile for a number of devices already.
# https://uxlfoundation.github.io/oneMath/building_the_project_with_dpcpp.html#building-for-onemath-generic-sycl-blas
# This means we override: https://github.com/uxlfoundation/oneMath/blob/develop/src/blas/backends/generic/CMakeLists.txt#L53
RUN cd /tmp && git clone https://github.com/uxlfoundation/oneMath.git && \
    mkdir oneMath/build && cd oneMath/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=native_cpu,nvptx64-nvidia-cuda,spir64" -DENABLE_GENERIC_BLAS_BACKEND=ON -DENABLE_MKLCPU_BACKEND=FALSE -DENABLE_MKLGPU_BACKEND=False .. && \
    cmake --build . -j4 && \
    cmake --install . --prefix ${SYCL_HOME} && \
    rm -rf /tmp/oneMath


FROM sycl AS onemath_cublas
# Install oneMath + CUDA BLAS + CUDA Solver. `DCUDA_CUDA_LIBRARY` is set to help onemath find cudart from container.

RUN cd /tmp && git clone https://github.com/uxlfoundation/oneMath.git && \
    mkdir oneMath/build && cd oneMath/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUSOLVER_BACKEND=False -DENABLE_CUBLAS_BACKEND=TRUE -DENABLE_MKLCPU_BACKEND=FALSE -DENABLE_MKLGPU_BACKEND=False .. \
    -DCUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/libcudart.so && \
    cmake --build . -j4 && \
    cmake --install . --prefix ${SYCL_HOME} && \
    rm -rf /tmp/oneMath


