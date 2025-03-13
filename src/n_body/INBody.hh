#pragma once

#include <cmath>
#include <cstddef>
#include <sycl/sycl.hpp>

constexpr float G = 1.0f;
constexpr float dt = 0.1f; // time factor

// Representation of a body in 2D space.
struct Body2 {
  Body2() : x(0), y(0), mass_(0), vx_(0), vy_(0), ax_(0), ay_(0) {}
  Body2(float x, float y, float mass)
      : x(x), y(y), mass_(mass), vx_(0), vy_(0), ax_(0), ay_(0) {}
  float x;
  float y;
  float mass_;

  // Motion attributes
  float vx_;
  float vy_;

  float ax_;
  float ay_;

  // Compute the force exerted by other on this.
  // >> This same function can be called in a SYCL kernel! <<
  inline void update(const Body2 &other) {

    float dx = other.x - x;
    float dy = other.y - y;
    float distSquared = dx * dx + dy * dy;
    if (distSquared == 0)
      return; // just skip anomalies
    float dist = sycl::sqrt(distSquared);

    float force = G * mass_ * other.mass_ / distSquared;
    float fx = force * dx / dist;
    float fy = force * dy / dist;

    // Update acceleration
    ax_ += fx / mass_;
    ay_ += fy / mass_;
  }

  // Apply motion
  inline void step() {
    vx_ += ax_ * dt;
    vy_ += ay_ * dt;
    x += vx_ * dt;
    y += vy_ * dt;
  }
};

class INBody {
public:
  // Reserve memory
  virtual void reserve(std::size_t num_elements) = 0;

  // Add body to the system
  virtual void addBody(Body2 &&body) = 0;

  // Iterate over all bodies and compute their next motion
  virtual void update() = 0;
};