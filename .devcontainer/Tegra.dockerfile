# https://developer.nvidia.com/embedded/jetson-cloud-native
# https://catalog.ngc.nvidia.com/containers?filters=&orderBy=scoreDESC&query=l4t&page=&pageSize=

# The base image has have matching CUDA runtime on the jetson device.
FROM nvcr.io/nvidia/l4t-cuda:12.2.12-devel

ENV SYCL_HOME="/opt/sycl"
ENV PATH=$PATH:${SYCL_HOME}/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SYCL_HOME}/lib

COPY --from=mdnf1992/sycl-dev ${SYCL_HOME}/ ${SYCL_HOME}/

RUN apt-get update && apt-get install -y git cmake libeigen3-dev && \
    apt clean && apt autoremove && \
    rm -rf /var/lib/apt/lists/* 

# Using CUBLAS & CUSOLVER backend from NVIDIA. Apparently only works if built on a Jetson device with --runtime nvidia.
# RUN cd /tmp && git clone https://github.com/uxlfoundation/oneMath.git && \
#     mkdir oneMath/build && cd oneMath/build && \
#     cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUSOLVER_BACKEND=TRUE -DENABLE_CUBLAS_BACKEND=TRUE -DENABLE_MKLCPU_BACKEND=FALSE -DENABLE_MKLGPU_BACKEND=False .. && \
#     cmake --build . -j4 && \
#     cmake --install . --prefix ${SYCL_HOME}