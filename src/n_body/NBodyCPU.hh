#pragma once

#include "INBody.hh"
#include <vector>

class NBodyCPU : public INBody {
public:
  void reserve(size_t num_elements) override;
  void addBody(Body2 &&body) override;
  void update() override;
  inline std::vector<Body2> &getBodies() { return bodies_; }

private:
  std::vector<Body2> bodies_;
};