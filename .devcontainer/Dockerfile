FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS build

ENV SYCL_HOME="/opt/sycl"
ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_VER=18

ARG TARGETARCH

WORKDIR /tmp

RUN apt update && apt install -y hwloc git pkg-config software-properties-common \
  clang clangd libclang-dev libclang-cpp-dev \
  libboost-dev libboost-context-dev libboost-fiber-dev build-essential llvm-spirv-${LLVM_VER} \
  python3 cmake ocl-icd-opencl-dev \
  libeigen3-dev && \
  apt clean && apt autoremove && \
  rm -rf /var/lib/apt/lists/* 

# RUN git clone https://github.com/pocl/pocl.git && \
#   mkdir pocl/build && cd pocl/build && \
#   if [ "$TARGETARCH" = "arm64" ]; then \
#     cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ICD=ON -DCMAKE_INSTALL_PREFIX=$SYCL_HOME -DLLC_HOST_CPU=apple-m3 ..; \
#   else \
#     cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ICD=ON -DCMAKE_INSTALL_PREFIX=$SYCL_HOME ..; \
#   fi && \
#   make -j6 install

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git -b v24.10.0 && \
  mkdir AdaptiveCpp/build && cd AdaptiveCpp/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENCL_BACKEND=ON -DCMAKE_INSTALL_PREFIX=$SYCL_HOME .. && \
  make -j1 install && \
  rm -rf /tmp/AdaptiveCpp

COPY ./cmake/cmake-kits.json $SYCL_HOME/cmake/
COPY ./cmake/sycl_toolchain.cmake $SYCL_HOME/cmake

ENV PATH=$PATH:$SYCL_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SYCL_HOME/lib

# Runtime docker image with non-dev-libraries only. We use cuda devel to enable acpp JIT.

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS runtime

ENV SYCL_HOME="/opt/sycl"
ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_VER=18

RUN apt-get update && apt-get install -y hwloc libgomp1 libclang-cpp18 ocl-icd-libopencl1 llvm-spirv-${LLVM_VER} clang-${LLVM_VER} && \
  apt-get clean && apt-get autoremove && \
  rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:$SYCL_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SYCL_HOME/lib

COPY --from=build $SYCL_HOME $SYCL_HOME