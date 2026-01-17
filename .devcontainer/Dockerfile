# Jetson containers: https://developer.nvidia.com/embedded/jetson-cloud-native
# Nvidia Containers: https://hub.docker.com/r/nvidia/cuda

ARG BASE_IMG=nvidia/cuda:12.6.1-devel-ubuntu24.04

FROM ${BASE_IMG} AS sycl

ENV DEBIAN_FRONTEND=noninteractive
ENV SYCL_HOME="/opt/sycl"
ENV PATH=$PATH:${SYCL_HOME}/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SYCL_HOME}/lib

RUN apt-get update && apt-get install -y git libhwloc-dev python3 cmake build-essential ninja-build clangd libeigen3-dev && \
  apt-get clean && apt-get autoremove && \
  rm -rf /var/lib/apt/lists/* 

RUN cd /tmp && git clone https://github.com/intel/llvm -b sycl && \ 
  python3 /tmp/llvm/buildbot/configure.py --cuda --native_cpu -t Release --cmake-opt="-DCMAKE_INSTALL_PREFIX=${SYCL_HOME}" && \
  python3 /tmp/llvm/buildbot/compile.py -t install -j8 && \
  rm -rf /tmp/llvm

FROM sycl AS lapack
# Install lapack
RUN apt-get update && apt install -y wget gfortran && \
  apt-get clean && apt autoremove && \
  rm -rf /var/lib/apt/lists/*

RUN cd /tmp && wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz -O lapack.tar.gz && \
    tar -xvf lapack.tar.gz && \
    mkdir lapack-3.12.1/build && cd lapack-3.12.1/build && \
    cmake -DBUILD_TESTING=OFF -DTEST_FORTRAN_COMPILER=OFF -DCBLAS=ON .. && \
    cmake --build . -j4 && \
    cmake --install . && \
    rm -rf /tmp/lapack*

FROM lapack AS onemath_lapack

RUN cd /tmp && git clone https://github.com/uxlfoundation/oneMath.git && \
    mkdir oneMath/build && cd oneMath/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_FUNCTIONAL_TESTS=False \
    -DENABLE_NETLIB_BACKEND=ON \
    -DENABLE_MKLCPU_BACKEND=OFF \
    -DENABLE_MKLGPU_BACKEND=OFF .. && \
    cmake --build . -j4 && \
    cmake --install . --prefix ${SYCL_HOME} && \
    rm -rf /tmp/oneMath

FROM sycl AS onemath_generic
# Install oneMath + Generic BLAS (Pure Sycl). Use CMAKE_CXX_FLAGS to compile for a number of devices already.
# https://uxlfoundation.github.io/oneMath/building_the_project_with_dpcpp.html#building-for-onemath-generic-sycl-blas
# This means we override: https://github.com/uxlfoundation/oneMath/blob/develop/src/blas/backends/generic/CMakeLists.txt#L53
RUN cd /tmp && git clone https://github.com/uxlfoundation/oneMath.git && \
    mkdir oneMath/build && cd oneMath/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_FUNCTIONAL_TESTS=False \
    -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=native_cpu,nvptx64-nvidia-cuda,spir64"\
    -DENABLE_GENERIC_BLAS_BACKEND=ON \
    -DENABLE_MKLCPU_BACKEND=OFF \
    -DENABLE_MKLGPU_BACKEND=OFF .. && \
    cmake --build . -j4 && \
    cmake --install . --prefix ${SYCL_HOME} && \
    rm -rf /tmp/oneMath


FROM lapack AS onemath_cublas
# Install oneMath + CUDA BLAS + CUDA Solver. `DCUDA_CUDA_LIBRARY` is set to help onemath find cudart from container.
# Make sure gcc 13 is used.

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update

# Install GCC 13 and G++ 13
RUN apt-get install -y gcc-13 g++-13 && \
    apt-get clean && apt autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && git clone https://github.com/uxlfoundation/oneMath.git && \
    mkdir oneMath/build && cd oneMath/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_FUNCTIONAL_TESTS=OFF \
    -DENABLE_CUSOLVER_BACKEND=OFF \
    -DENABLE_NETLIB_BACKEND=ON \
    -DENABLE_CUBLAS_BACKEND=ON \
    -DENABLE_MKLCPU_BACKEND=OFF \
    -DENABLE_MKLGPU_BACKEND=OFF .. \
    -DCUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/libcudart.so && \
    cmake --build . -j4 && \
    cmake --install . --prefix ${SYCL_HOME} && \
    rm -rf /tmp/oneMath


