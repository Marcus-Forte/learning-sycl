#include <Eigen/Dense>
#include <sycl/sycl.hpp>

using PointT = Eigen::Matrix<float, 2, 1>;
using PointCloudT = std::vector<PointT>;

float Pt2Distance(const PointT &p1, const PointT &p2) {
  float dx = p1.x() - p2.x();
  float dy = p1.y() - p2.y();
  return tan(cos(cos(sin(cos(dx * dx + dy * dy)))));
}
// Kernel : Runs in CPU and GPU
PointT transformPoint(const PointT &point, Eigen::Affine2f &transform) {
  auto cx = std::cos(point[0]);
  auto cy = std::sin(point[0]);

  auto cxcy = std::sin(std::cos(cx) + cy);
  return transform * point;
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "too few args..\n";
    exit(0);
  }

  int device_idx = 0;
  if (argc == 3){
    device_idx = atoi(argv[2]);

    if (device_idx >= sycl::device::get_num_devices()) {
      std::cout << "Not valid device index\n";
      exit(0);
    }
  }
  
  for (const auto & device : sycl::device::get_devices()) {
    std::cout << "Found device: " << device.get_info<sycl::info::device::name>() << std::endl;
  }

  const size_t num_points = std::atoi(argv[1]);
  sycl::queue queue (sycl::device::get_devices()[device_idx]);
  std::cout << "Running on: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  PointCloudT input(num_points, PointT::Zero());
  PointCloudT output(num_points, PointT::Zero());

  // fill input with random numbers
  for (auto &point : input) {
    point = PointT::Random();
  }

  Eigen::Affine2f transform = Eigen::Affine2f::Identity();
  transform.translate(Eigen::Vector2f(1.0, 1.0));
  transform.rotate(0.5);

  auto now = std::chrono::high_resolution_clock::now();
  // std::transform(input.begin(), input.end(), output.begin(),
  //                [&transform](const PointT &point) {
  //                  return transformPoint(point, transform);
  //                });

  float cpu_reduction_result = 0;

  for (int i = 0; i < num_points; i++) {
    cpu_reduction_result += Pt2Distance(input[i], output[i]);
  }

  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - now)
                   .count();
  std::cout << "CPU time: " << delta << " ms" << std::endl;

  auto *input_gpu =
      sycl::malloc_device<PointT>(sizeof(PointT) * num_points, queue);
  auto *output_gpu =
      sycl::malloc_device<PointT>(sizeof(PointT) * num_points, queue);
  auto *matrix =
      sycl::malloc_device<Eigen::Affine2f>(sizeof(Eigen::Affine2f), queue);

  queue.memcpy(input_gpu, input.data(), sizeof(PointT) * num_points).wait();
  queue.memcpy(matrix, &transform, sizeof(Eigen::Affine2f)).wait();

  auto *gpu_reduction_result = sycl::malloc_device<float>(sizeof(float), queue);
  // *gpu_reduction_result = 0.0f;

  now = std::chrono::high_resolution_clock::now();
  queue
      .submit([&](sycl::handler &cgh) {
        auto allDistances =
            sycl::reduction(gpu_reduction_result, 0.0f, sycl::plus<float>{});

        cgh.parallel_for(
            sycl::range<1>(num_points), allDistances,
            [=](sycl::id<1> idx, auto& reduction) {

              // output_gpu[idx] = transformPoint(input_gpu[idx], *matrix);
              // output_gpu[idx1].x() = (*matrix)(0, 0) * input_gpu[idx1][0] + (*matrix)(0, 1) * input_gpu[idx1][1] + (*matrix)(0,2);
              // output_gpu[idx].y() = (*matrix)(1, 0) * input_gpu[idx].x() + (*matrix)(1, 1) * input_gpu[idx].y() + (*matrix)(1,2);
              

              reduction += Pt2Distance(input_gpu[idx], output_gpu[idx]);
            });
      })
      .wait();
  delta = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - now)
              .count();
  std::cout << "GPU time: " << delta << " ms" << std::endl;

  std::cout << "GPU Reduction: " << *gpu_reduction_result << std::endl;
  std::cout << "CPU Reduction: " << cpu_reduction_result << std::endl;

  auto *output_cast = reinterpret_cast<PointT *>(output_gpu);

  /// CPU vs GPU results
  for (int i = 0; i < 1; i++) {
    std::cout << input[num_points - i - 1].transpose() << " -> "
              << output[num_points - i - 1].transpose() << std::endl;
    std::cout << input[num_points - i - 1].transpose() << " -> "
              << output_cast[num_points - i - 1].transpose() << std::endl;
  }

  return 0;
}