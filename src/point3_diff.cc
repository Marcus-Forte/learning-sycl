#include "common.hh"
#include <Eigen/Dense>
#include <sycl/sycl.hpp>

using PointT = Eigen::Matrix<float, 3, 1>;
using PointCloudT = std::vector<PointT>;

const PointT query(0.0, 0.0, 0.0);
constexpr float max_distance = 15.0;

// Kernels : Runs in both CPU or GPU!!
static inline float Pt2SquaredDistance(const PointT &p1, const PointT &p2) {
  float dx = p1.x() - p2.x();
  float dy = p1.y() - p2.y();
  float dz = p1.z() - p2.z();
  return dx * dx + dy * dy + dz * dz;
}

static inline PointT transformPoint(const PointT &point,
                                    Eigen::Affine3f &transform) {
  return transform * point;
}

void printUsage() {
  std::cout << "Usage: point3_diff <num_points> [device_idx] \n";
}

int main(int argc, char **argv) {

  if (argc < 2) {
    printUsage();
    exit(0);
  }

  int device_idx = 0;
  if (argc == 3) {
    device_idx = atoi(argv[2]);

    if (device_idx >= sycl::device::get_devices().size()) {
    std::cout << "Not valid device index\n";
    exit(0);
  }
  }

  for (const auto &device : sycl::device::get_devices()) {
    std::cout << "Found device: " << device.get_info<sycl::info::device::name>()
              << std::endl;
  }

  const size_t num_points = std::atoi(argv[1]);
  sycl::queue queue(sycl::device::get_devices()[device_idx]);
  printDeviceInfo(queue);

  PointCloudT input(num_points, PointT::Zero());
  PointCloudT output(num_points, PointT::Zero());

  // fill input with random numbers
  for (auto &point : input) {
    point = PointT::Random();
  }

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translate(Eigen::Vector3f(1.0, 2.0, 3.0));
  transform.rotate(Eigen::AngleAxisf(M_PI / 4, Eigen::Vector3f::UnitZ()));

  auto now = std::chrono::high_resolution_clock::now();
  // Transform
  std::transform(input.begin(), input.end(), output.begin(),
                 [&transform](const PointT &point) {
                   auto output = transformPoint(point, transform);
                   return output;
                 });

  // Reduction
  float cpu_reduction_result = 0;
  for (int i = 0; i < num_points; i++) {
    cpu_reduction_result += Pt2SquaredDistance(input[i], output[i]);
  }

  // Query closest
  int closest_points = 0;
  for (int i = 0; i < num_points; i++) {
    if (Pt2SquaredDistance(query, output[i]) < max_distance) {
      ++closest_points;
    }
  }

  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - now)
                   .count();
  std::cout << "CPU time: " << delta << " ms" << std::endl;

  std::cout << "Needed GPU memory: "
            << (sizeof(PointT) * num_points * 2 + sizeof(Eigen::Affine3f)) /
                   (1024 * 1024)
            << " MB \n";

  std::cout << "Copying CPU -> GPU data..." << std::endl;
  auto *input_gpu = sycl::malloc_device<PointT>(num_points, queue);
  auto *output_gpu = sycl::malloc_device<PointT>(num_points, queue);
  auto *matrix_gpu = sycl::malloc_device<Eigen::Affine3f>(1, queue);

  queue.copy(input.data(), input_gpu, num_points).wait();
  queue.copy(&transform, matrix_gpu, 1).wait();

  auto *gpu_reduction_result = sycl::malloc_device<float>(1, queue);
  auto *gpu_closest_points = sycl::malloc_device<int>(1, queue);

  now = std::chrono::high_resolution_clock::now();

  queue
      .submit([&](sycl::handler &cgh) {
        auto allDistances =
            sycl::reduction(gpu_reduction_result, 0.0f, sycl::plus<float>{});

        auto closestPoints =
            sycl::reduction(gpu_closest_points, 0, sycl::plus<int>{});

        cgh.parallel_for(
            sycl::range<1>(num_points), allDistances, closestPoints,
            [=](sycl::id<1> idx, auto &reduction, auto &nr_closest_points) {
              // Transform
              output_gpu[idx] = transformPoint(input_gpu[idx], *matrix_gpu);
              // Reduction
              reduction += Pt2SquaredDistance(input_gpu[idx], output_gpu[idx]);
              PointT query_(0.0, 0.0, 0.0);
              // Query closest
              if (Pt2SquaredDistance(query_, output_gpu[idx]) < max_distance) {
                ++nr_closest_points;
              }
            });
      })
      .wait();

  delta = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - now)
              .count();
  std::cout << "GPU time: " << delta << " ms" << std::endl;

  int gpu_closest_points_result = 0;
  queue.copy(gpu_closest_points, &gpu_closest_points_result, 1).wait();

  float gpu_reduction_result_result = 0;
  queue.copy(gpu_reduction_result, &gpu_reduction_result_result, 1).wait();

  std::cout << "CPU closest points: " << closest_points << std::endl;
  std::cout << "GPU closest points: " << gpu_closest_points_result << std::endl;

  std::cout << "GPU Reduction: " << gpu_reduction_result_result << std::endl;
  std::cout << "CPU Reduction: " << cpu_reduction_result << std::endl;

  PointT output_last_point;
  // copy last point
  queue.copy(&output_gpu[num_points - 1], &output_last_point, 1).wait();

  /// CPU vs GPU results
  std::cout << "CPU: " << input[num_points - 1].transpose() << " -> "
            << output[num_points - 1].transpose() << std::endl;
  std::cout << "GPU: " << input[num_points - 1].transpose() << " -> "
            << output_last_point.transpose() << std::endl;

  return 0;
}