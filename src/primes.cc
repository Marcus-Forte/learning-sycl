#include "common.hh"
#include <sycl/sycl.hpp>
#include <cmath> // <-- This is the line you need to add

struct Result {
  unsigned int first;
  unsigned int last;
   int num_primes;
};

struct IntRange {
    unsigned int first;
    unsigned int last;
};

static void printUsage() {
  std::cout << "Usage: primes <number> <threads> <device>\n";
  std::cout << "Computes all prime numbers up to the specified number.\n";
}

static bool is_prime(int n) {
  if (n < 2)
    return false;
  for (int i = 2, sq = std::sqrt(n); i <= sq; ++i)
    if (n % i == 0)
      return false;
  return true;
}

static Result compute_primes(unsigned int first, unsigned int last) {
   int num_primes = 0;
  for (unsigned int i = first; i <= last; ++i) {
    if (is_prime(i)) {
      ++num_primes;
    }
  }
  return {first, last, num_primes};
}

int main(int argc, char **argv) {

  if (argc != 4) {
    printUsage();
    return 1;
  }

  int number = std::atoi(argv[1]);
  int workers = std::atoi(argv[2]);
  int device_idx = atoi(argv[3]);

//   if (device_idx >= sycl::device::get_num_devices()) {
//     std::cout << "Not valid device index\n";
//     exit(0);
//   }

  for (const auto &device : sycl::device::get_devices()) {
    std::cout << "Found device: " << device.get_info<sycl::info::device::name>()
              << std::endl;
  }

  sycl::queue queue(sycl::device::get_devices()[device_idx]);
  printDeviceInfo(queue);


  // compute ranges for each worker
  unsigned int range = number / workers;
  std::vector<IntRange> range_pairs;
  for (unsigned int i = 0; i < workers; ++i) {
    unsigned int first = i * range;
    unsigned int last = (i == workers - 1) ? number : (i + 1) * range - 1;
    range_pairs.push_back( {first,last});
  }

  std::cout << "Nr. Ranges: " << range_pairs.size() << std::endl;

  auto *gpu_ranges = sycl::malloc_device<IntRange>(range_pairs.size(), queue);
  auto *gpu_reduction_result = sycl::malloc_device< int>(1, queue);
  queue.memset(gpu_reduction_result, 0, sizeof( int));

  queue.copy(range_pairs.data(), gpu_ranges, range_pairs.size()).wait();

  queue.submit([&](sycl::handler &cgh) {

    auto out = sycl::stream(1024, 768, cgh);

    auto total_nr_primes =
            sycl::reduction(gpu_reduction_result, 0, sycl::plus< int>{});

            cgh.parallel_for(
            sycl::range<1>(workers), total_nr_primes, [=] (sycl::id<1> idx, auto &reduction) {
                auto result = compute_primes(gpu_ranges[idx].first, gpu_ranges[idx].last);
                reduction += result.num_primes;
                out << gpu_ranges[idx].first << "  " << gpu_ranges[idx].last << " --> " << result.num_primes << "\n";
            });

  }).wait();

   int total_nr_primes_result = 0;
  queue.copy(gpu_reduction_result, &total_nr_primes_result, 1).wait();


   std::cout << "Total nr primes: " << total_nr_primes_result << std::endl;
}