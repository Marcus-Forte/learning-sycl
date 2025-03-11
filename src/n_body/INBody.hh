#pragma once

#include <cmath>
#include <cstddef>

// Always start with zero motion.
struct Body2 {
  Body2() = default;
  Body2(float x, float y, float mass) : x(x), y(y), mass_(mass), vx_(0), vy_(0), ax_(0), ay_(0) {}
  float x;
  float y;
  float mass_;

  // motion
  float vx_;
  float vy_;

  float ax_;
  float ay_;

   // Compute the force exerted by other on this
  inline void update(const Body2& other) {
   
    float dx = other.x - x;
    float dy = other.y - y;
    float distSquared = dx * dx + dy * dy;
    if(distSquared == 0) return; // just skip anomalies
    float dist = std::sqrt(distSquared);
    // Assume G = 1
    float force = mass_ * other.mass_ / distSquared;
    float fx = force * dx / dist;
    float fy = force * dy / dist;

    // Update acceleration
    ax_ += fx / mass_;
    ay_ += fy / mass_;
  }

  // Apply motion
  inline void step() {
    vx_ += ax_;
    vy_ += ay_;
    x += vx_;
    y += vy_;
  }

};

class INBody {
public:
  // Reserve memory
  virtual void reserve(std::size_t num_elements) = 0;
  // Add body to the system
  virtual void addBody(Body2&& body) = 0;

  // Iterate over all bodies and compute their next motion
  virtual void update() = 0;

  
};