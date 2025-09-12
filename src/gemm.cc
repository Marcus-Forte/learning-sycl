#include <oneapi/math.hpp>

#include <future>
#include <iostream>

#define EIGEN_ARM64_USE_SVE
#include <Eigen/Dense>

#include "common.hh"

int main(int argc, char **argv) {

  if (argc < 3) {
    std::cout << "Usage: gemm <dim> <device id> \n";
    return -1;
  }

  // create dim * dim matrices
  const int dim = atoi(argv[1]);
  const int device_idx = atoi(argv[2]);

  if (device_idx >= sycl::device::get_devices().size()) {
    std::cout << "Not valid device index\n";
    exit(0);
  }

  for (const auto &device : sycl::device::get_devices()) {
    std::cout << "Found device: " << device.get_info<sycl::info::device::name>()
              << std::endl;
  }

  sycl::queue queue(sycl::device::get_devices()[device_idx]);
  printDeviceInfo(queue);

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(dim, dim);
  Eigen::MatrixXd C(dim, dim);

  const oneapi::math::backend_selector<oneapi::math::backend::generic> backend(
      queue);

  auto *mA = sycl::malloc_device<double>(dim * dim, queue);
  auto *mB = sycl::malloc_device<double>(dim * dim, queue);
  auto *mC = sycl::malloc_device<double>(dim * dim, queue);

  auto copy_A_event = queue.copy<double>(A.data(), mA, A.size());
  auto copy_B_event = queue.copy<double>(B.data(), mB, B.size());

  auto start = std::chrono::high_resolution_clock::now();
  sycl::event::wait({copy_A_event, copy_B_event});
  auto delta_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now() - start)
                      .count();
  std::cout << "Copy to device took:" << delta_us << " us\n";

  // GPU GEMM
  auto cpu_gemm_event = std::async(std::launch::async, [&]() {
    auto start = std::chrono::high_resolution_clock::now();
    C = A * B;
    const auto delta_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start)
            .count();
    std::cout << "CPU Gemm took:" << delta_cpu_us << " us\n";
  });

  // dispatch
  start = std::chrono::high_resolution_clock::now();
  auto res = oneapi::math::blas::column_major::gemm(
      backend,
      oneapi::math::transpose::nontrans, // op(a)
      oneapi::math::transpose::nontrans, // op(b)
      dim,                               // m
      dim,                               // n
      dim,                               // k
      1.0,                               // alpha
      mA,                                // A*
      dim,                               // lda
      mB,                                // B*
      dim,                               // ldb
      0.0,                               // beta
      mC,                                // C*
      dim,                               // ldc
      {copy_A_event, copy_B_event});

  res.wait();
  delta_us = std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::high_resolution_clock::now() - start)
                 .count();

  std::cout << "SYCL Gemm took:" << delta_us << " us\n";
  queue.copy<double>(mC, C.data(), C.size()).wait();
  cpu_gemm_event.get();

  // // cuBLAS GEMM (CUDA 12.6)

  // cublasHandle_t handle;
  // cublasCreate(&handle);

  // const double alpha = 1.0;
  // const double beta = 0.0;

  // auto start_cublas = std::chrono::high_resolution_clock::now();
  // cublasDgemm(handle,
  //       CUBLAS_OP_N, CUBLAS_OP_N,
  //       dim, dim, dim,
  //       &alpha,
  //       mA, dim,
  //       mB, dim,
  //       &beta,
  //       mC, dim);
  // cudaDeviceSynchronize();
  // auto delta_cublas_us = std::chrono::duration_cast<std::chrono::microseconds>(
  //   std::chrono::high_resolution_clock::now() - start_cublas).count();
  // std::cout << "cuBLAS Gemm took:" << delta_cublas_us << " us\n";

  // cublasDestroy(handle);
  // free all
  sycl::free(mA, queue);
  sycl::free(mB, queue);
  sycl::free(mC, queue);
}