#include <iostream>
#include <sycl/sycl.hpp>

template <class T> void print_few(const std::vector<T> &input) {
  constexpr auto num = 5;
  for (int i = 0; i < num; i++) {
    std::cout << input[i] << " ";
  }

  std::cout << "\n...\n";
  for (int i = input.size(); i > input.size() - num; i--) {
    std::cout << input[i - 1] << " ";
  }
  std::cout << std::endl;
}

template <class T>
void cpu_square(const std::vector<T> &input, std::vector<T> &out) {
  for (int i = 0; i < input.size(); i++) {
    out[i] = input[i] * input[i];
  }
}

template <class T> T cpu_reduction(const std::vector<T> &input) {
  return std::reduce(input.begin(), input.end(), 0.0, std::plus<T>());
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "too few args..\n";
    exit(0);
  }
  sycl::queue queue;
  std::cout << "Running on: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  std::cout << "USM: "
            << queue.get_device().has(sycl::aspect::usm_device_allocations)
            << std::endl;

  uint64_t total_time = 0;
  const size_t iterations = std::atoi(argv[1]);
  const size_t N = std::atoi(argv[2]);

  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "Elements: " << N << " size: " << sizeof(int) * N / 1024 << " KB"
            << std::endl;

  std::vector<int64_t> input(N, 1);
  std::iota(input.begin(), input.end(), 0);
  print_few(input);

  std::cout << "CPU Start reduction\n";
  auto now = std::chrono::high_resolution_clock::now();
  const auto cpu_reduction_result = cpu_reduction(input);
  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - now)
                   .count();
  std::cout << "CPU Reduction result: " << cpu_reduction_result
            << " time: " << delta << " ms" << std::endl;

  std::cout << "CPU Start Square vectors\n";
  std::vector<int64_t> output(N, 1);
  now = std::chrono::high_resolution_clock::now();
  cpu_square(input, output);
  delta = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - now)
              .count();
  std::cout << "CPU Square Vector result: " << cpu_reduction_result
            << " time: " << delta << " ms" << std::endl;

  print_few(output);

  auto *d_input = sycl::malloc_device<int64_t>(N, queue);
  queue.memcpy(d_input, input.data(), sizeof(int64_t) * N).wait();

  auto *d_out_host = sycl::malloc_shared<int64_t>(N, queue);
  std::cout << "GPU Start Square vectors\n";
  now = std::chrono::high_resolution_clock::now();
  queue
      .submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
          d_out_host[idx] = d_input[idx] * d_input[idx];
        });
      })
      .wait();

  delta = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - now)
              .count();
  std::cout << "GPU Square result time: " << delta << " ms" << std::endl;

  std::cout << d_out_host[0] << " " << d_out_host[N - 1] << std::endl;

  std::cout << "GPU Start reduction\n";
  int64_t *gpu_reduction_result = sycl::malloc_device<int64_t>(1, queue);

  now = std::chrono::high_resolution_clock::now();

  queue
      .submit([&](sycl::handler &h) {
        auto reduction =
            sycl::reduction(gpu_reduction_result, sycl::plus<int64_t>());

        h.parallel_for(sycl::range<1>(N), reduction,
                       [=](sycl::id<1> idx, auto &reduction) {
                         reduction += d_input[idx];
                       });
      })
      .wait();

  int64_t gpu_reduction_result_;
  queue.memcpy(&gpu_reduction_result_, gpu_reduction_result, sizeof(int64_t))
      .wait();

  delta = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - now)
              .count();
  std::cout << "GPU Reduction result: " << gpu_reduction_result_
            << " time: " << delta << " ms" << std::endl;

  return 0;
}