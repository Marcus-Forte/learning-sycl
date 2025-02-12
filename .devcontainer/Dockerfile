FROM ubuntu:latest

ENV DPCPP_HOME="/opt/sycl"

WORKDIR $DPCPP_HOME

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y libpocl-dev pocl-opencl-icd build-essential git hwloc python3 cmake

RUN git clone https://github.com/intel/llvm -b sycl

RUN python3 $DPCPP_HOME/llvm/buildbot/configure.py --host-target "AArch64;ARM;X86" --cmake-gen "Unix Makefiles" && \
    python3 $DPCPP_HOME/llvm/buildbot/compile.py -j6

ENV PATH=$PATH:$DPCPP_HOME/llvm/build/install/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_HOME/llvm/build/install/lib