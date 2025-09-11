#include <sycl/sycl.hpp>

#include "common.hh"
#include <cmath>

static void printUsage() {
  std::cout << "Usage: primes <number> <device>\n";
  std::cout << "Computes all prime numbers up to the specified number.\n";
}

static unsigned int is_prime(unsigned int n) {
  if (n < 2)
    return 0;
  for (unsigned int i = 2, sq = std::sqrt(n); i <= sq; ++i)
    if (n % i == 0)
      return 0;
  return 1;
}

int main(int argc, char **argv) {

  if (argc != 3) {
    printUsage();
    return 1;
  }

  int number = std::atoi(argv[1]);
  int device_idx = atoi(argv[2]);

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

  auto start = std::chrono::high_resolution_clock::now();

  auto *gpu_reduction_result = sycl::malloc_device<unsigned int>(1, queue);

  queue.submit([&](sycl::handler &cgh) {
    // auto out = sycl::stream(1024, 768, cgh);
    auto total_nr_primes = sycl::reduction<unsigned int>(
        gpu_reduction_result, 0, sycl::plus<unsigned int>{},
        sycl::property::reduction::initialize_to_identity{});

    cgh.parallel_for(
        sycl::range<1>(number), total_nr_primes,
        [=](sycl::id<1> idx, auto &reduction) { reduction += is_prime(idx); });
  }).wait();

  unsigned int total_nr_primes_result = 0;
  queue.copy(gpu_reduction_result, &total_nr_primes_result, 1).wait();

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Kernel execution time: " << elapsed.count()
            << " milliseconds\n";

  std::cout << "Total nr primes: " << total_nr_primes_result << std::endl;
}