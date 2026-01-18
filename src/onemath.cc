#include <oneapi/math.hpp>

#include <future>
#include <iostream>

#define EIGEN_ARM64_USE_SVE
#include <Eigen/Dense>

#include "common.hh"

// Performs the operations: 
// Gemm := C = A * B
// Syrk := C = A^T * A

int main(int argc, char **argv) {

  if (argc != 5) {
    std::cout << "Execute: Gemm := C = A(m x n) * B (n x k) and Syrk := C = A^T * A\n";
    std::cout << "Usage: gemm <dim m> <dim n> <dim k> <device id> \n";
    std::cout << "e.g: gemm 500 500 500 0 \n";
    return -1;
  }

  // create dim * dim matrices
  const int dim_m = atoi(argv[1]);
  const int dim_n = atoi(argv[2]);
  const int dim_k = atoi(argv[3]);
  const int device_idx = atoi(argv[4]);

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

  Eigen::MatrixXf A = Eigen::MatrixXf::Random(dim_m, dim_n);
  Eigen::MatrixXf B = Eigen::MatrixXf::Random(dim_n, dim_k);
  Eigen::MatrixXf C(dim_m, dim_k);
  Eigen::MatrixXf C_syrk(dim_n, dim_n); // C_syrk = A^T * T := (n x n)

  auto *mA = sycl::malloc_device<float>(dim_m * dim_n, queue);
  auto *mB = sycl::malloc_device<float>(dim_n * dim_k, queue);
  auto *mC = sycl::malloc_device<float>(dim_m * dim_k, queue);
  auto *mC_syrk = sycl::malloc_device<float>(dim_n * dim_n, queue);

  auto copy_A_event = queue.copy<float>(A.data(), mA, A.size());
  auto copy_B_event = queue.copy<float>(B.data(), mB, B.size());

  auto start = std::chrono::high_resolution_clock::now();
  sycl::event::wait({copy_A_event, copy_B_event});
  auto delta_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now() - start)
                      .count();
  std::cout << "Copy to device took:" << delta_us << " us\n";

  // GPU EIGEN GEMM
  auto cpu_syrk_event = std::async(std::launch::async, [&]() {
    auto start = std::chrono::high_resolution_clock::now();
    C_syrk = A.transpose() * A;
    const auto delta_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start)
            .count();
    std::cout << "CPU Eigen Syrk took:" << delta_cpu_us << " us\n";
  });

  // CPU EIGEN SYRK
  auto cpu_gemm_event = std::async(std::launch::async, [&]() {
    auto start = std::chrono::high_resolution_clock::now();
    C = A * B;
    const auto delta_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start)
            .count();
    std::cout << "CPU Eigen Gemm took:" << delta_cpu_us << " us\n";
  });

  // dispatch
  #ifdef USE_CUBLAS_BACKEND
   const oneapi::math::backend_selector<oneapi::math::backend::cublas> backend(
      queue);
  #elif defined(USE_LAPACK_BACKEND)
   const oneapi::math::backend_selector<oneapi::math::backend::netlib> backend(
      queue);
  #elif defined(USE_GENERIC_BACKEND)
  const oneapi::math::backend_selector<oneapi::math::backend::generic> backend(
      queue);
      #else
      #error "No BLAS backend selected for oneMath"
  #endif
  
  start = std::chrono::high_resolution_clock::now();
  auto res = oneapi::math::blas::column_major::gemm(
      backend,
      oneapi::math::transpose::nontrans, // op(a)
      oneapi::math::transpose::nontrans, // op(b)
      dim_m,                               // m (C is m x k)
      dim_k,                               // n (C is m x k)
      dim_n,                               // k (A is m x n, B is n x k)
      1.0,                                 // alpha
      mA,                                  // A*
      dim_m,                               // lda
      mB,                                  // B*
      dim_n,                               // ldb
      0.0,                                 // beta
      mC,                                  // C*
      dim_m                                // ldc
      );

  res.wait();
  delta_us = std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::high_resolution_clock::now() - start)
                 .count();

  std::cout << "Onemath gemm took:" << delta_us << " us\n";


  start = std::chrono::high_resolution_clock::now();
    auto res_syrk = oneapi::math::blas::column_major::syrk(
      backend,
      oneapi::math::uplo::upper,        // upper | lower
      oneapi::math::transpose::trans,   // op(a)
      dim_n,                            // n (C is n x n)
      dim_m,                            // k (A is m x n)
      1.0,                              // alpha
      mA,                               // A*
      dim_m,                            // lda
      0.0,                              // beta
      mC_syrk,                          // C*
      dim_n                             // ldc
      );
  res_syrk.wait();
  delta_us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::high_resolution_clock::now() - start)
                  .count();

  std::cout << "Onemath syrk took:" << delta_us << " us\n";

  cpu_gemm_event.get();
  cpu_syrk_event.get();
  
  sycl::free(mA, queue);
  sycl::free(mB, queue);
  sycl::free(mC, queue);
  sycl::free(mC_syrk, queue);
}