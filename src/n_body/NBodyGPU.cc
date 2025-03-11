#include "NBodyGPU.hh"

void NBodyGPU::reserve(std::size_t num_elements) {
  bodies_.reserve(num_elements);
}

void NBodyGPU::addBody(Body2 &&body) {
  bodies_.emplace_back(body.x, body.y, body.mass_);
}

void NBodyGPU::update() {

  auto compute_forces = queue_.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(bodies_.size()), [=](sycl::id<1> idx) {
      for (int j = 0; j < bodies_.size(); j++) {
        if (idx[0] != j) {
          // Compute influence of all other bodies
          bodies_[idx].update(bodies_[j]);
        }
      }
    });
  });

  auto compute_motion = queue_.submit([&](sycl::handler &cgh) {
    cgh.depends_on(compute_forces);
    cgh.parallel_for(sycl::range<1>(bodies_.size()),
                     [=](sycl::id<1> idx) { bodies_[idx].step(); });
  });

  compute_motion.wait();
}

void NBodyGPU::printAll() {
    // Only works for shared usm
    for (const auto &body : bodies_) {
        std::cout << "GPU Body: " << body.x << " " << body.y << " " << body.mass_
                  << std::endl;
      }
}