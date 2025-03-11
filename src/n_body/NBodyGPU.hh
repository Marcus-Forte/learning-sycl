#pragma once

#include "INBody.hh"
#include <sycl/sycl.hpp>

using usm_device_allocator =
    sycl::usm_allocator<Body2, sycl::usm::alloc::shared>;

class NBodyGPU : public INBody {
public:
  NBodyGPU(sycl::queue &queue)
      : queue_(queue), allocator_(queue), bodies_(allocator_) {}
  void reserve(size_t num_elements) override;
  void addBody(Body2 &&body) override;
  void update() override;
  void printAll(); // for debugging

private:
  sycl::queue queue_;
  usm_device_allocator allocator_;
  std::vector<Body2, usm_device_allocator> bodies_;
};