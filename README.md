# SYCL + OneMath

This repository contains small SYCL experiments and OneMath BLAS examples.
It focuses on comparing CPU and accelerator backends and validating results
against CPU implementations.

## Test Applications

Build targets in this repo provide the test applications used for validation
and benchmarking:

- `primes`: prime counting with SYCL kernels.
- `n_body_sim`: N-body simulation (CPU/GPU backends).
- `onemath`: GEMM/SYRK via oneMath BLAS.
- `gemm_cublas`: CUDA cuBLAS GEMM example.

## Docker

- Build the dev image by calling `./docker/build.sh`

### Docker images for different backends

Use separate images to target specific backends. Common patterns:

- SYCL + oneMath w/ generic SYCL backend
- SYCL + oneMath w/ Netlib/LAPACK backend
- SYCL + oneMath w/ cuBLAS backend (CUDA)

Select the image that matches the backend you want to test.
 
## Resources

* https://intel.github.io/llvm/index.html
* https://github.khronos.org/SYCL_Reference/
* https://sycl.tech/getting-started#academy

## SYCL Book

* https://library.oapen.org/handle/20.500.12657/76704#:~:text=This%20open%20access%20book%20enables%20C%2B%2B%20programmers%20to,explanations%2C%20and%20code%20examples%20to%20illustrate%20key%20topics.

## OneMath

OneMath exposes BLAS functions via SYCL. A number of options and backends can be used. 

- https://uxlfoundation.github.io/oneMath/introduction.html
- https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemath/source/
