#include <sycl/sycl.hpp>

#include <chrono>

#include "common.hh"
#include "n_body/NBodyCPU.hh"
#include "n_body/NBodyGPU.hh"

int main(int argc, char **argv) {

  if (argc < 3) {
    std::cout << "too few args..\n";
    exit(0);
  }

  const size_t num_elements = std::atoi(argv[1]);
  const int num_iterations = std::atoi(argv[2]);

  int device_idx = 0;
  if (argc == 4) {
    device_idx = atoi(argv[3]);

    if (device_idx >= sycl::device::get_num_devices()) {
      std::cout << "Not valid device index\n";
      exit(0);
    }
  }

  for (const auto &device : sycl::device::get_devices()) {
    std::cout << "Found device: " << device.get_info<sycl::info::device::name>()
              << std::endl;
  }

  sycl::queue queue(sycl::device::get_devices()[device_idx]);
  printDeviceInfo(queue);

  NBodyGPU nbody_gpu(queue);
  NBodyCPU nbody_cpu;
  nbody_gpu.reserve(num_elements);
  nbody_cpu.reserve(num_elements);

  for (size_t i = 0; i < num_elements; i++) {
    float x = std::rand() % 100;
    float y = std::rand() % 100;
    float mass = (std::rand() % 100) + 1.0;
    nbody_cpu.addBody({x, y, mass});
    nbody_gpu.addBody({x, y, mass});
  }

  // Iterating CPU
  std::cout << "Iterating " << num_iterations << " times" << std::endl;
  auto now = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; i++) {
    nbody_cpu.update();
  }
  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - now)
                   .count();

  nbody_cpu.printAll();
  std::cout << "Done. CPU time: " << delta << " ms" << std::endl;

  // Iterating GPU
  std::cout << "Iterating " << num_iterations << " times" << std::endl;
  now = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; i++) {
    nbody_gpu.update();
  }

  delta = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - now)
              .count();

  nbody_gpu.printAll();
  std::cout << "Done. GPU time: " << delta << " ms" << std::endl;

  return 0;
}