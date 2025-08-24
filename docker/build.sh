#!/bin/bash -e

# Build SYCL Docker images

docker build -f .devcontainer/Dockerfile --target develop -t mdnf1992/sycl-dev .

docker build -f .devcontainer/Dockerfile --target runtime -t mdnf1992/sycl-run .
