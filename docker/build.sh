#!/bin/bash -e

# Build SYCL development docker image

docker build -f .devcontainer/Dockerfile -t mdnf1992/sycl-dev .
