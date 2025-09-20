#!/bin/bash -e

# Build SYCL Docker images

# Default (Generic)
docker build -f .devcontainer/Dockerfile --target onemath_generic -t mdnf1992/sycl-dev . && \
docker push mdnf1992/sycl-dev

# Default (Lapack)
docker build -f .devcontainer/Dockerfile --target onemath_lapack -t mdnf1992/sycl-dev:lapack . && \
docker push mdnf1992/sycl-dev:lapack

# Tegra (Cuda)
docker build -f .devcontainer/Dockerfile --build-arg BASE_IMG=nvcr.io/nvidia/l4t-cuda:12.2.12-devel --target onemath_cublas -t mdnf1992/sycl-dev:tegra . && \
docker push mdnf1992/sycl-dev:tegra