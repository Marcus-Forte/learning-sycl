FROM mdnf1992/cpp-dev

ENV DPCPP_HOME="/opt/sycl_workspace"

WORKDIR $DPCPP_HOME

RUN apt update && apt install -y libpocl-dev pocl-opencl-icd:

RUN git clone https://github.com/intel/llvm -b sycl && \
    python3 $DPCPP_HOME/llvm/buildbot/configure.py --host-target "AArch64;ARM;X86" --cmake-gen "Unix Makefiles" && \
    python3 $DPCPP_HOME/llvm/buildbot/compile.py
