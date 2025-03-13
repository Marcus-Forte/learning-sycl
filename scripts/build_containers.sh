#!/bin/bash -e

# Build SYCL docker images

usage() {
    echo "Usage: $0 [-a] [-r] [-p] [-h]"
    echo "  -a  Build for all platforms (linux/amd64, linux/arm64)"
    echo "  -r  Build the runtime container"
    echo "  -p  Push the containers"
    echo "  -h  Show help"
    exit 1
}

RUNTIME=false
PUSH=false
ALL_PLATFORMS=false

while getopts "arph" opt; do
    case ${opt} in
        a)
            ALL_PLATFORMS=true
            ;;
        r)
            RUNTIME=true
            ;;
        p) 
            PUSH=true
            ;;
        h)
            usage
            ;;
        \?)
            usage
            ;;
    esac
done

if $ALL_PLATFORMS; then
    PLATFORMS="--platform linux/amd64,linux/arm64"
else
    PLATFORMS=""
fi

if $RUNTIME; then
    TAG="mdnf1992/sycl-runtime"
    TARGET="runtime"
else # devcontainer
    TAG="mdnf1992/sycl-dev"
    TARGET="build"
fi

docker buildx build $PLATFORMS -f .devcontainer/Dockerfile --target $TARGET -t $TAG .

if $PUSH; then
    docker push $TAG
fi
