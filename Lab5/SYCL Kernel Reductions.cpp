%%writefile lab/reduction_lab.cpp
#include <sycl/sycl.hpp>
using namespace sycl;
static constexpr size_t N = 1024; // global size
static constexpr size_t B = 128; // work-group size
int main() {
    //# setup queue with default selector
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    //# initialize data array using usm
    auto data = malloc_shared<int>(N, q);
    for (int i = 0; i < N; i++) data[i] = i;
    //# implicit USM for writing min and max value
    int* min = malloc_shared<int>(1, q);
    int* max = malloc_shared<int>(1, q);
    *min = 0;
    *max = 0;
    //# STEP 1 : Create reduction objects for computing min and max
    //# YOUR CODE GOES HERE
    auto min_reduction = sycl::reduction(min, sycl::minimum<>());
    auto max_reduction = sycl::reduction(max, sycl::maximum<>());
    //# Reduction Kernel get min and max
    q.submit([&](handler& h) {
    //# STEP 2 : add parallel_for with reduction objects for min and max
    //# YOUR CODE GOES HERE
        h.parallel_for(nd_range<1>{N, B}, min_reduction, max_reduction, [=]
        (nd_item<1> it, auto& min_temp, auto& max_temp) {
        int i = it.get_global_id(0);
        min_temp.combine(data[i]);
        max_temp.combine(data[i]);
        });
    }).wait();
    //# STEP 3 : Compute mid_range from min and max
    int mid_range = (*min+*max)/2;
    //# YOUR CODE GOES HERE
    std::cout << "Mid-Range = " << mid_range << "\n";
    free(data, q);
    free(min, q);
    free(max, q);
    return 0;
}