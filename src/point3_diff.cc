#include <Eigen/Dense>
#include <sycl/sycl.hpp>

using PointT = Eigen::Matrix<float, 2, 1>;
using PointCloudT = std::vector<PointT>;

float squaredDist(const PointT &a, const PointT &b) {
  return (a - b).squaredNorm();
}

// Kernel : Runs in CPU or GPU
PointT transformPoint(const PointT &point, Eigen::Affine2f &transform) {
  return transform * point;
}

struct ErrorModel {

  ErrorModel() : transform_(Eigen::Affine2f::Identity()) {}
  ErrorModel(const Eigen::Affine2f &transform) : transform_(transform) {}

  PointT operator()(const PointT &src, PointT &tgt) {
    return tgt - transform_ * src;
  }

  Eigen::Affine2f transform_;
};

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "too few args..\n";
    exit(0);
  }

  const size_t num_points = std::atoi(argv[1]);
  sycl::queue queue;
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
  std::transform(input.begin(), input.end(), output.begin(),
                 [&transform](const PointT &point) {
                   return transformPoint(point, transform);
                 });

  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - now)
                   .count();
  std::cout << "CPU time: " << delta << " ms" << std::endl;
  for (int i = 0; i < 1; i++) {
    std::cout << input[num_points - i - 1].transpose() << " -> "
              << output[num_points - i - 1].transpose() << std::endl;
  }

  float total_dist = 0;
  for (const auto &el : output) {
    total_dist += el.squaredNorm();
  }

  std::cout << "CPU Total dist: " << total_dist << std::endl;

  // GPU //

  auto *input_gpu =
      sycl::malloc_shared<PointT>(sizeof(PointT) * num_points, queue);
  auto *output_gpu =
      sycl::malloc_shared<PointT>(sizeof(PointT) * num_points, queue);
  auto *observations_gpu =
      sycl::malloc_shared<PointT>(sizeof(PointT) * num_points, queue);
  auto *matrix =
      sycl::malloc_shared<Eigen::Affine2f>(sizeof(Eigen::Affine2f), queue);

  queue.memcpy(input_gpu, input.data(), sizeof(PointT) * num_points).wait();
  queue.memcpy(matrix, transform.data(), sizeof(Eigen::Affine2f)).wait();

  ErrorModel model(*matrix);

  now = std::chrono::high_resolution_clock::now();

  auto *reduction_store = sycl::malloc_shared<float>(sizeof(float), queue);
  *reduction_store = 0.0;
  // Compute error
  queue
      .submit([&](sycl::handler &cgh) {
        auto squared_point =
            sycl::reduction(reduction_store, std::plus<float>());

        cgh.parallel_for(
            sycl::range<1>(num_points), squared_point,
            [&](sycl::id<1> idx, auto &reduction) {
              // model(input_gpu[idx], output_gpu[idx]);
              output_gpu[idx] = model(input_gpu[idx], observations_gpu[idx]);
              reduction.combine(squaredDist(output_gpu[idx], output_gpu[idx]));
            });
      })
      .wait();
  delta = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - now)
              .count();
  std::cout << "GPU time: " << delta << " ms" << std::endl;
  std::cout << "GPU total dist: " << *reduction_store << std::endl;

  auto *output_cast = reinterpret_cast<PointT *>(output_gpu);

  for (int i = 0; i < 1; i++) {
    std::cout << input[num_points - i - 1].transpose() << " -> "
              << output_cast[num_points - i - 1].transpose() << std::endl;
  }

  return 0;
}