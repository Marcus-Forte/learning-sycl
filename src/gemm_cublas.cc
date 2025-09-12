#include <iostream>

#include <Eigen/Dense>
#include <chrono>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Macro for robust CUDA error checking
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Macro for robust CUBLAS error checking
#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "CUBLAS Error %d at %s:%d\n", status, __FILE__,          \
              __LINE__);                                                       \
      /* Note: CUBLAS does not provide a function to get error strings. */     \
      /* You can look up the error enum `cublasStatus_t` in the documentation. \
       */                                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "Usage: gemm_cublas <dim> \n";
    return -1;
  }

  // create dim * dim matrices
  const int dim = atoi(argv[1]);

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(dim, dim);
  Eigen::MatrixXd C(dim, dim);

  // cuBLAS GEMM (CUDA 12.6)

  double *mA, *mB, *mC;
  CHECK_CUDA(cudaMalloc((void **)&mA, dim * dim * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void **)&mB, dim * dim * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void **)&mC, dim * dim * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(mA, A.data(), dim * dim * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(mA, A.data(), dim * dim * sizeof(double), cudaMemcpyHostToDevice));;

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  const double alpha = 1.0;
  const double beta = 0.0;

  auto start_cublas = std::chrono::high_resolution_clock::now();
  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                           &alpha, mA, dim, mB, dim, &beta, mC, dim));
  cudaDeviceSynchronize();
  auto delta_cublas_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_cublas)
          .count();
  std::cout << "cuBLAS Gemm took:" << delta_cublas_us << " us\n";

  cublasDestroy(handle);
  // free all

  cudaFree(mA);
  cudaFree(mB);
  cudaFree(mC);
}