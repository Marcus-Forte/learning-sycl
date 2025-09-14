#!/bin/bash -e

# Build SYCL Docker images

# Default
docker build -f .devcontainer/Dockerfile -t mdnf1992/sycl-dev .

# Tegra
docker build -f .devcontainer/Dockerfile --build-arg BASE_IMG=nvcr.io/nvidia/l4t-cuda:12.2.12-devel -t mdnf1992/sycl-dev:tegra .