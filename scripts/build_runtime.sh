#!/bin/bash
# Build a runtime docker to be pulled by the target.


RUNTIME_IMAGE_TAG="sycl-runtime"

docker build -f .devcontainer/Dockerfile --target runtime -t $RUNTIME_IMAGE_TAG .

echo "Runtime sycl image: \"$RUNTIME_IMAGE_TAG\""