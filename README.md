# SYCL made easy

## Docker

- Build the dev / runtime image by calling `./docker/build.sh`
 
## Resources

* https://intel.github.io/llvm/index.html
* https://github.khronos.org/SYCL_Reference/
* https://sycl.tech/getting-started#academy

## SYCL Book

* https://library.oapen.org/handle/20.500.12657/76704#:~:text=This%20open%20access%20book%20enables%20C%2B%2B%20programmers%20to,explanations%2C%20and%20code%20examples%20to%20illustrate%20key%20topics.


## OpenCL 
- ICD loader loads ICD clients. Vendor should supply ICD clients. Only one loader needed.
https://stackoverflow.com/questions/36951200/enable-opencl-over-multiple-platforms-in-linux-how-to-proceed-with-icd-files
https://linuxhandbook.com/setup-opencl-linux-docker/
- Linking against ICD loader means you can multiplex CL devices! The `ocl-icd-opencl-dev` brings one with it.

# TODO
- Figure out efficient way of cross compiling host side (e.g from x86_64 -> arm64 ).
- Create new, stripped away container with libraries only.
