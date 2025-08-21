#include <sycl/sycl.hpp>

#include <chrono>

#include "common.hh"
#include "n_body/NBodyCPU.hh"
#include "n_body/NBodyGPU.hh"

void printUsage() {
  std::cout << "Usage: n_body_sim <num_elements> <num_iterations> [device_idx] "
               "[compare] \n";
}

int main(int argc, char **argv) {

  if (argc < 3) {
    printUsage();
    exit(0);
  }

  const size_t num_elements = std::atoi(argv[1]);
  const int num_iterations = std::atoi(argv[2]);

  int device_idx = 0;
  if (argc == 4) {
    device_idx = atoi(argv[3]);

    if (device_idx >= sycl::device::get_devices().size()) {
      std::cout << "Not valid device index\n";
      exit(0);
    }
  }

  bool compare = false;
  if (argc == 5) {
    std::cout << "Compare results enabled.\n";
    compare = true;
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

  // Iterating GPU
  auto now = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; i++) {
    std::cout << "GPU Iteration " << i << std::endl;
    nbody_gpu.update();
  }

  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - now)
                   .count();

  std::cout << "Done. GPU time: " << delta << " ms" << std::endl;

  // Iterating CPU
  now = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; i++) {
    std::cout << "CPU Iteration " << i << std::endl;
    nbody_cpu.update();
  }
  delta = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - now)
              .count();

  std::cout << "Done. CPU time: " << delta << " ms" << std::endl;

  // For debugging
  if (num_elements < 10) {
    std::cout << "CPU Results:\n";
    const auto cpu_bodies = nbody_cpu.getBodies();
    for (const auto &body : cpu_bodies) {
      std::cout << "GPU Body: " << body.x << " " << body.y << " " << std::endl;
    }
    std::cout << "GPU Results:\n";
    const auto gpu_bodies = nbody_gpu.getBodies();
    for (const auto &body : gpu_bodies) {
      std::cout << "GPU Body: " << body.x << " " << body.y << " " << std::endl;
    }
  }

  // Note that CPU and GPU numeric precision can deviate significantly if
  // algorithm is not numerically stable.

  if (compare) {
    const auto &cpu_bodies = nbody_cpu.getBodies();
    const auto &gpu_bodies = nbody_gpu.getBodies();
    bool match = true;
    for (int i = 0; i < num_elements; ++i) {
      auto dx = cpu_bodies[i].x - gpu_bodies[i].x;
      auto dy = cpu_bodies[i].y - gpu_bodies[i].y;
      const auto diff = dx * dx + dy * dy;
      if (diff > 1e-5) {
        std::cout << "Mismatch body: " << i << "(diff: " << diff << "): \n";
        std::cout << "\tCPU: " << cpu_bodies[i].x << " " << cpu_bodies[i].y
                  << std::endl;
        std::cout << "\tGPU: " << gpu_bodies[i].x << " " << gpu_bodies[i].y
                  << std::endl;
        match = false;
      }
    }
    if (match) {
      std::cout << "Results match.\n";
    }
  }

  return 0;
}
