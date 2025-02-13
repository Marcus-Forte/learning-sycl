# SYCL for Raspberry Pi

1. Install LLVM. `https://apt.llvm.org/` 
2. build oneAPI LLVM+SYCL.
3. Build POCL. Careful attention to arch flags.
  - `cmake -DENABLE_SPIRV=ON .. -DENABLE_ICD=ON -DCMAKE_INSTALL_PREFIX=install ..`
  - `DENABLE_CUDA=ON` => fix issue where libcuda is searched in wrong lib folder.
4. Install OCL loaders `https://launchpad.net/~ocl-icd/+archive/ubuntu/ppa`
 


## Resources

* https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md
* https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
* https://portablecl.org/docs/html/sycl_with_pocl_arm.html
* https://portablecl.org/docs/html/install.html
* https://github.com/pocl/pocl
* https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md

## SYCL 

* https://library.oapen.org/handle/20.500.12657/76704#:~:text=This%20open%20access%20book%20enables%20C%2B%2B%20programmers%20to,explanations%2C%20and%20code%20examples%20to%20illustrate%20key%20topics.

## Troubleshooting

- `SYCL_UR_TRACE=-1 POCL_DEBUG=all`
- `OCL_ICD_VENDORS`

- SParc plug?


- ICD loader loads ICD clients. Vendor should supply ICD clients. Only one loader needed.
https://stackoverflow.com/questions/36951200/enable-opencl-over-multiple-platforms-in-linux-how-to-proceed-with-icd-files
https://linuxhandbook.com/setup-opencl-linux-docker/