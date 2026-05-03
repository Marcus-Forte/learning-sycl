#!/bin/bash -e

# Build SYCL Docker images

# Default (Lapack)
docker build --push -f .devcontainer/Dockerfile --build-arg BASE_IMG=mdnf1992/cpp-dev --target onemath_lapack -t mdnf1992/sycl-dev .

# Default (Lapack) + Cuda
docker build --push -f .devcontainer/Dockerfile --build-arg SYCL_CUDA_FLAG="--cuda" --build-arg BASE_IMG=mdnf1992/cpp-dev --target onemath_lapack -t mdnf1992/sycl-dev .

# Default (Generic)
docker build --push -f .devcontainer/Dockerfile --build-arg BASE_IMG=mdnf1992/cpp-dev --target onemath_generic -t mdnf1992/sycl-dev:generic .

# Tegra (Cuda)
docker build --push -f .devcontainer/Dockerfile --build-arg BASE_IMG=nvcr.io/nvidia/l4t-cuda:12.2.12-devel --build-arg SYCL_CUDA_FLAG="--cuda" --target onemath_cublas -t mdnf1992/sycl-dev:tegra .