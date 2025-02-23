#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
  sycl::queue q;

  if (argc < 2 ) {
    std::cout << "too few args..\n";
    exit(0);
  }
  std::cout << "Running on: "
  << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  uint64_t total_time = 0;
  const size_t iterations = std::atoi(argv[1]);
  const size_t N = std::atoi(argv[2]);

  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "Elements: " << N << std::endl;

  for(int it=0; it < iterations; ++it) {
  sycl::event ex;
  int* d_buf = sycl::malloc_device<int>(N, q   );
  int* h_buf = sycl::malloc_host<int>(N, q );

  for(int i = 0; i < N; i ++){
        h_buf[i] = i*i;
  }
  auto time_now = std::chrono::high_resolution_clock::now();
  q.memcpy(d_buf, h_buf, N*sizeof(int)).wait();

  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> it){
    const int i = it[0];
    d_buf[i] += i;
  }).wait();
  q.memcpy(h_buf, d_buf, N*sizeof(int)).wait();
  int correct = 1;
  for(int i = 0; i < N; i ++){
    if(h_buf[i] != i*i + i){
        std::cerr << "ERROR: h_buf[" << i << "]=" << h_buf[i] << " and shuold be " << i*i + i << std::endl;
        correct =0;
    }
  }
  if(correct){
    std::cout << "Results are correct!!\n";
  }
  auto time_delta = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time_now).count();
  total_time += time_delta;
  }
  std::cout << "Avg Time: " << total_time /  iterations << " us\n";

  

  return 0;
}