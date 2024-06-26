// %%writefile lab/sub_group_lab.cpp
// #include <sycl/sycl.hpp>
// using namespace sycl;
// static constexpr size_t N = 1024; // global size
// static constexpr size_t B = 256;  // work-group size
// static constexpr size_t S = 32;  // sub-group size
// int main() {
//     queue q;
//     std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
//     //# allocate USM shared allocation for input data array and sg_data array
//     int *data = malloc_shared<int>(N, q);
//     int *sg_data = malloc_shared<int>(N/S, q);
//     //# initialize input data array
//     for (int i = 0; i < N; i++) data[i] = i;
//     for (int i = 0; i < N; i++) std::cout << data[i] << " ";
//     std::cout << "\n\n";
//     //# Kernel task to compute sub-group sum and save to sg_data array
//     //# STEP 1 : set fixed sub_group size of value S in the kernel below
//     q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) 
//     [[sycl::reqd_sub_group_size(S)]] {
//     auto sg = item.get_sub_group();
//     auto i = item.get_global_id(0);
//     //# STEP 2: Add all elements in sub_group using sub_group reduce
//     int sg_sum = reduce_over_group(sg, data[i], std::plus<>());
//     //# STEP 3 : save each sub-group sum to sg_data array
//     if (sg.leader()) {
//         sg_data[item.get_group(0)] = sg_sum;
//     }
//     }).wait();
//     //# print sg_data array
//     for (int i = 0; i < N/S; i++) std::cout << sg_data[i] << " ";
//     std::cout << "\n";
//     //# STEP 4: compute sum of all elements in sg_data array
//     int sum = 0;
//     for (int i = 0; i < N/S; i++) sum += sg_data[i];
//     std::cout << "\nSum = " << sum << "\n";
//     //# free USM allocations
//     free(data, q);
//     free(sg_data, q);
//     return 0;
// }
%%writefile lab/sub_group_lab.cpp
#include <sycl/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 1024; // global size
static constexpr size_t B = 256;  // work-group size
static constexpr size_t S = 32;   // sub-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  // Allocate USM shared allocation for input data array and sg_data array
  int *data = malloc_shared<int>(N, q);
  int *sg_data = malloc_shared<int>(N / S, q);

  // Initialize input data array
  for (int i = 0; i < N; i++) data[i] = i;
  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n\n";

  // Kernel task to compute sub-group sum and save to sg_data array
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) [[sycl::reqd_sub_group_size(S)]] {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    // Add all elements in sub_group using sub_group reduce
    int sg_sum = reduce_over_group(sg, data[i], std::plus<>());

    // Save each sub-group sum to sg_data array
    if (sg.leader()) {
      int group_id = item.get_group(0);
      int sg_id = sg.get_group_id();
      int sg_index = group_id * (B / S) + sg_id;
      sg_data[sg_index] = sg_sum;
    }
  }).wait();

  // Print sg_data array
  for (int i = 0; i < N / S; i++) std::cout << sg_data[i] << " ";
  std::cout << "\n";

  // Compute sum of all elements in sg_data array
  int sum = 0;
  for (int i = 0; i < N / S; i++) sum += sg_data[i];

  std::cout << "\nSum = " << sum << "\n";

  // Free USM allocations
  free(data, q);
  free(sg_data, q);

  return 0;
}
