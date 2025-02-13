// t.cpp
#include <sycl/sycl.hpp>
#include <iostream>


#define N 10

int main() {
  sycl::queue q;

  sycl::event ex;
  int* d_buf = sycl::malloc_device<int>(N, q   );
  int* h_buf = sycl::malloc_host<int>(N, q );

  for(int i = 0; i < N; i ++){
        h_buf[i] = i*i;
  }
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

  //# Print the device name
  std::cout << "Device 1: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "He";
  return 0;
}