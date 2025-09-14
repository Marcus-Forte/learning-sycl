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

  Eigen::MatrixXf A = Eigen::MatrixXf::Random(dim, dim);
  Eigen::MatrixXf B = Eigen::MatrixXf::Random(dim, dim);
  Eigen::MatrixXf C(dim, dim);

  auto *mA = sycl::malloc_device<float>(dim * dim, queue);
  auto *mB = sycl::malloc_device<float>(dim * dim, queue);
  auto *mC = sycl::malloc_device<float>(dim * dim, queue);

  auto copy_A_event = queue.copy<float>(A.data(), mA, A.size());
  auto copy_B_event = queue.copy<float>(B.data(), mB, B.size());

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
  #ifdef USE_CUBLAS_BACKEND
   const oneapi::math::backend_selector<oneapi::math::backend::cublas> backend(
      queue);
  #else
  const oneapi::math::backend_selector<oneapi::math::backend::generic> backend(
      queue);
  #endif
  
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
      dim                               // ldc
      );

  res.wait();
  delta_us = std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::high_resolution_clock::now() - start)
                 .count();

  std::cout << "SYCL Gemm took:" << delta_us << " us\n";
  queue.copy<float>(mC, C.data(), C.size()).wait();
  cpu_gemm_event.get();
  
  sycl::free(mA, queue);
  sycl::free(mB, queue);
  sycl::free(mC, queue);
}