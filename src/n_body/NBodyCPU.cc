#include "NBodyCPU.hh"

void NBodyCPU::reserve(size_t num_elements) { bodies_.reserve(num_elements); }

void NBodyCPU::addBody(Body2 &&body) {
  bodies_.emplace_back(body.x, body.y, body.mass_);
}

void NBodyCPU::update() {
  for (size_t i = 0; i < bodies_.size(); i++) {
    for (size_t j = 0; j < bodies_.size(); j++) {
      if (i != j) {

        // Compute influence of all other bodies
        bodies_[j].update(bodies_[i]);
      }
    }
  }

  // Apply motion
  for (auto &body : bodies_) {
    body.step();
  }
}
