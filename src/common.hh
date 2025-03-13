#pragma once

#include <iostream>
#include <sycl/sycl.hpp>

inline void printDeviceInfo(const sycl::queue &queue) {
  std::cout << "Running on: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  std::cout
      << "Device Memory: "
      << queue.get_device().get_info<sycl::info::device::global_mem_size>() /
             (1024 * 1024)
      << " MB\n";
  std::cout
      << "Device Shared Memory: "
      << queue.get_device().get_info<sycl::info::device::local_mem_size>() /
             1024
      << " KB\n";

  std::cout
      << "Max Work Groups: "
      << queue.get_device().get_info<sycl::info::device::max_work_group_size>()
      << "\n";

  std::cout
      << "Max Compute units "
      << queue.get_device().get_info<sycl::info::device::max_compute_units>()
      << "\n";
}