#pragma once

#include "INBody.hh"
#include <vector>

class NBodyCPU : public INBody {
public:
   void reserve(size_t num_elements) override;
  void addBody(Body2&& body) override;
  void update() override;
  void printAll();

private:
  std::vector<Body2> bodies_;
};