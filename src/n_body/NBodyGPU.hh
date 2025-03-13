#pragma once

#include "INBody.hh"
#include <sycl/sycl.hpp>

class NBodyGPU : public INBody {
public:
  NBodyGPU(sycl::queue &queue);
  void reserve(size_t num_elements) override;
  void addBody(Body2 &&body) override;
  void update() override;
  std::vector<Body2> getBodies();

private:
  sycl::queue &queue_;
  Body2 *bodies_;
  size_t num_bodies_;
  size_t added_bodies_;
};