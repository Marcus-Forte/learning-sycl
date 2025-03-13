#include "NBodyGPU.hh"

NBodyGPU::NBodyGPU(sycl::queue &queue) : queue_(queue) {}

void NBodyGPU::reserve(std::size_t num_elements) {
  // https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html
  // https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/
  bodies_ = sycl::malloc_device<Body2>(num_elements, queue_);
  num_bodies_ = num_elements;
  added_bodies_ = 0;
}

// Make sure memory is allocated.
void NBodyGPU::addBody(Body2 &&body) {
  queue_.copy<Body2>(&body, &bodies_[added_bodies_], 1).wait();
  added_bodies_++;
}

void NBodyGPU::update() {

  // Note that because we must call kernels with [=] capture,
  // we must excpliciy capture the class members as follow to access them in
  // the kernel. Directly using those variables will not work.

  auto *bodies_caputure = bodies_;
  auto num_bodies_capture = num_bodies_;

  auto compute_forces =
      queue_.parallel_for(sycl::range<1>(num_bodies_), [=](sycl::id<1> idx) {
        // Compute influence from other bodies
        for (int j = 0; j < num_bodies_capture; j++) {
          if (idx[0] != j) {
            bodies_caputure[idx].update(bodies_caputure[j]);
          }
        }
      });

  // compute motion
  queue_
      .submit([&](sycl::handler &cgh) {
        cgh.depends_on(compute_forces);
        cgh.parallel_for(sycl::range<1>(num_bodies_),
                         [=](sycl::id<1> idx) { bodies_caputure[idx].step(); });
      })
      .wait();
}

std::vector<Body2> NBodyGPU::getBodies() {
  std::vector<Body2> bodies(num_bodies_);

  queue_.copy<Body2>(bodies_, bodies.data(), num_bodies_).wait();
  return bodies;
}
