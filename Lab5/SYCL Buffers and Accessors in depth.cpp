%%writefile lab/lab_buffers.cpp
#include <sycl/sycl.hpp>
using namespace sycl;
int main() {
    const int N = 256;
    int data[N];
    for (int i = 0; i < N; i++) data[i] = i;
    std::cout << "\nInput Values: ";    
    for (int i = 0; i < N; i++) std::cout << data[i] << " "; 
    std::cout << "\n";
    buffer<int, 1> buf_data(data, range<1>(N));
    //# STEP 1 : Create 3 sub-buffers for buf_data with length 64, 128 and 64. 
    buffer<int, 1> sub_buf1(buf_data, id<1>(0), range<1>(64));
    buffer<int, 1> sub_buf2(buf_data, id<1>(64), range<1>(128));
    buffer<int, 1> sub_buf3(buf_data, id<1>(192), range<1>(64));
    //# STEP 2 : Submit task to Multiply the elements in first sub buffer by 2 
    queue q1;
    q1.submit([&](handler& h) {
        auto acc = sub_buf1.get_access<access::mode::read_write>(h);
        h.parallel_for(range<1>(64), [=](id<1> idx) {
            acc[idx] *= 2;
        });
    });
    //# STEP 3 : Submit task to Multiply the elements in second sub buffer by 3    
    queue q2;
    q2.submit([&](handler& h) {
        auto acc = sub_buf2.get_access<access::mode::read_write>(h);
        h.parallel_for(range<1>(128), [=](id<1> idx) {
            acc[idx] *= 3;
        });
    });    
    //# STEP 4 : Submit task to Multiply the elements in third sub buffer by 2 
    queue q3;
    q3.submit([&](handler& h) {
        auto acc = sub_buf3.get_access<access::mode::read_write>(h);
        h.parallel_for(range<1>(64), [=](id<1> idx) {
            acc[idx] *= 2;
        });
    });  
    //# STEP 5 : Create Host accessors to get the results back to the host from the device
    {
        auto host_acc = buf_data.get_access<access::mode::read>();
        for (int i = 0; i < N; i++) {
            data[i] = host_acc[i];
        }
    }
    std::cout << "\nOutput Values: ";
    for (int i = 0; i < N; i++) std::cout << data[i] << " ";
    std::cout << "\n";
    return 0;
}