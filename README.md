# SYCL made easy

1. --
 
## Resources

* https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md
* https://sycl.tech/getting-started/academy/1-what-is-sycl/lesson
* https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
* https://portablecl.org/docs/html/sycl_with_pocl_arm.html
* https://portablecl.org/docs/html/install.html
* https://github.com/pocl/pocl
* https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md
* https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/sycl-ecosystem.md

## SYCL Book

* https://library.oapen.org/handle/20.500.12657/76704#:~:text=This%20open%20access%20book%20enables%20C%2B%2B%20programmers%20to,explanations%2C%20and%20code%20examples%20to%20illustrate%20key%20topics.

## Troubleshooting / Env Vars

- ocl: `OCL_ICD_VENDORS`
- pocl: `POCL_DEBUG=all`
- acpp: `ACPP_VISIBILITY_MASK`
- acpp: `ACPP_DEBUG_LEVEL=4`

- ICD loader loads ICD clients. Vendor should supply ICD clients. Only one loader needed.
https://stackoverflow.com/questions/36951200/enable-opencl-over-multiple-platforms-in-linux-how-to-proceed-with-icd-files
https://linuxhandbook.com/setup-opencl-linux-docker/

# TODO
- Figure out efficient way of cross compiling host side (e.g from x86_64 -> arm64 ).
- Create new, stripped away container with libraries only.

# General Info:
- Linking against ICD loader means you can multiplex CL devices! The `ocl-icd-opencl-dev` brings one with it.
