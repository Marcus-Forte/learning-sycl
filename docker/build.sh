#!/bin/bash -e

# Build SYCL development docker image

docker build -f .devcontainer/Dockerfile --target develop -t mdnf1992/sycl-dev .
