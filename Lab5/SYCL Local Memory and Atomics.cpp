%%writefile lab/atomics_lab.cpp
#include <sycl/sycl.hpp>
#include <limits>
using namespace sycl;
static constexpr size_t N = 1024; // global size
int main() {
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    // Initialize data
    auto data = malloc_shared<int>(N, q);
    for (int i = 0; i < N; i++) data[i] = i;
    // Initialize min and max
    auto min = malloc_shared<int>(1, q);
    auto max = malloc_shared<int>(1, q);
    min[0] = std::numeric_limits<int>::max();
    max[0] = std::numeric_limits<int>::min();
    // Reduction Kernel using atomics
    q.parallel_for(N, [=](auto i) {
    //# STEP 1: create atomic reference for min and max
    auto atomic_min = sycl::atomic_ref<int, 
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device, 
                        access::address_space::global_space>(min[0]);
    auto atomic_max = sycl::atomic_ref<int, 
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device, 
                        access::address_space::global_space>(max[0]);
    //# STEP 2: add atomic operation for min and max computation  
    atomic_min.fetch_min(data[i]);
    atomic_max.fetch_max(data[i]);
    }).wait();
    // Compute mid-range using the min and max
    auto mid = (min[0] + max[0]) / 2.0;
    std::cout << "Minimum   = " << min[0] << "\n";
    std::cout << "Maximum   = " << max[0] << "\n";
    std::cout << "Mid-Range = " << mid << "\n";
    // Free allocated memory
    free(data, q);
    free(min, q);
    free(max, q);
    return 0;
}