#!/bin/bash -e

# Build SYCL Docker images

docker build -f .devcontainer/Dockerfile --target develop -t mdnf1992/sycl-dev .