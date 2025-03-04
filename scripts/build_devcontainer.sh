#!/bin/bash
# Build the devcontainer docker image

DEVCONTAINTER_IMAGE_TAG="mdnf1992/sycl-dev"

docker buildx build -f .devcontainer/Dockerfile --target build -t $DEVCONTAINTER_IMAGE_TAG .

echo "Devcontainer sycl image: \"$DEVCONTAINTER_IMAGE_TAG\""