#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include <thread>

using namespace std;
using namespace std::chrono;
// CSR matrix class
class matCsr {
public:
    vector<double> values;
    vector<int> col_indices;
    vector<int> row_ptr;
    int n, m, nnz;

    matCsr() : n(0), m(0), nnz(0) {}
    matCsr(int n0, int m0) : n(n0), m(m0), nnz(0) {
        row_ptr.resize(n0 + 1, 0);
    }

    void append(int n0, int m0, double val) {
        if (abs(val) < 1e-9) return;
        values.push_back(val);
        col_indices.push_back(m0);
        row_ptr[n0 + 1]++;
        nnz++;
    }

    void finalize() {
        for (int i = 0; i < n; ++i) {
            row_ptr[i + 1] += row_ptr[i];
        }
    }

    void createMat(int n0, int m0) {
        n = n0;
        m = m0;
        values.clear();
        col_indices.clear();
        row_ptr.resize(n0 + 1, 0);
        nnz = 0;
    }
};

// Dense matrix class
class mat {
public:
    int n, m;
    vector<double> v; // Change to a single vector to ensure contiguous memory allocation

    mat() {
        m = 1;
        n = 1;
        createMat(1, 1);
    }

    mat(int n0, int m0) {
        m = m0;
        n = n0;
        createMat(n0, m0);
    }

    void createMat(int n0, int m0) {
        n = n0;
        m = m0;
        v.resize(n * m, 0.0); // Ensure contiguous memory allocation
    }

    double& at(int i, int j) {
        return v[i * m + j];
    }

    const double& at(int i, int j) const {
        return v[i * m + j];
    }

    // other member functions ...
};

__global__ void csrMatMultKernel(int n, int m, int k, double* d_values, int* d_col_indices, int* d_row_ptr, double* d_B, double* d_C, int startRow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + startRow;
    if (row < n) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int idx = d_row_ptr[row]; idx < d_row_ptr[row + 1]; ++idx) {
                int col = d_col_indices[idx];
                if (col < m) { // Ensure column index is within bounds
                    sum += d_values[idx] * d_B[col * k + j];
                }
            }
            d_C[(row - startRow) * k + j] = sum; // Adjust row index for the offset
        }
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(1); \
        } \
    } while (0)

void matMultiplyCPU(int n_threads,const matCsr& csr, const mat& B, mat& C, int startRow, int endRow) {
    int k = B.m;
    omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(dynamic)
    for (int row = startRow; row < endRow; row++) {
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int idx = csr.row_ptr[row]; idx < csr.row_ptr[row + 1]; ++idx) {
                int col = csr.col_indices[idx];
                if (col < csr.m) { // Ensure column index is within bounds
                    sum += csr.values[idx] * B.at(col, j);
                }
            }
            C.at(row, j) = sum;
        }
    }
}

void matMultiply(const matCsr& csr, const mat& B, mat& C, int k,int n_threads=8) {
    int n = csr.n;
    int m = csr.m;
    int numRowsCPU = n / k;
    int numRowsGPU = n - numRowsCPU;

    C.createMat(n, B.m);

    // CPU part
    thread cpuThread(matMultiplyCPU, n_threads, ref(csr), ref(B), ref(C), 0, numRowsCPU);

    // GPU part
    double* d_values, * d_B, * d_C;
    int* d_col_indices, * d_row_ptr;

    CUDA_CHECK(cudaMalloc(&d_values, csr.nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, csr.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, m * B.m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, numRowsGPU * B.m * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_values, csr.values.data(), csr.nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, csr.col_indices.data(), csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.v.data(), m * B.m * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (numRowsGPU + blockSize - 1) / blockSize;

    csrMatMultKernel << <gridSize, blockSize >> > (n, m, B.m, d_values, d_col_indices, d_row_ptr, d_B, d_C, numRowsCPU);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C.v.data() + numRowsCPU * B.m, d_C, numRowsGPU * B.m * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Wait for the CPU thread to finish
    cpuThread.join();
}

void generateRandomSparseMatrix(matCsr& csr, int n, int m, int nnz) {
    csr.createMat(n, m);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis_row(0, n - 1);
    uniform_int_distribution<> dis_col(0, m - 1);
    uniform_real_distribution<> dis_val(0, 100);

    for (int i = 0; i < nnz; ++i) {
        int row = dis_row(gen);
        int col = dis_col(gen);
        double val = dis_val(gen);
        csr.append(row, col, val);
    }
    csr.finalize();
}

void generateRandomMatrix(mat& dense, int n, int m) {
    dense.createMat(n, m);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 100);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            dense.at(i, j) = dis(gen);
        }
    }
}

int main() {
    int n = 4096;
    // int m = 4096;
    // int nnz = 1000000; // Number of non-zero elements in CSR matrix
    int splitFactor = 128; // k value for CPU/GPU split

    // matCsr csr;
    // mat B(m, n), C;
    for(int n=1024;n>=16;n>>=1)
    {
        matCsr csr;
        mat B(4096, 4096), C;
        // generateRandomSparseMatrix(csr, n, m, nnz);
        // generateRandomMatrix(B, m, n);
        generateRandomSparseMatrix(csr, 4096, 4096, 1ll*4096*4096/n);
        generateRandomMatrix(B, 4096, 4096);
        for (int spl = 2; spl <= 10; spl++)
        {
            auto start = high_resolution_clock::now();
            for (int i = 1; i < 10; i++)  matMultiply(csr, B, C, spl, 8);
            // for (int i = 1; i < 10; i++)  matMultiply(csr, B, C, thr);
            auto end = high_resolution_clock::now();
            auto duration1 = duration_cast<milliseconds>(end - start).count();
            //cout << "Time Elapsed : " << duration1 << " ms with factor = " <<thr << endl;
            cout << duration1 << " " << n << " "<< spl << endl;
        }
    }
    // for (int thr = 2; thr <= 10; thr++)
    // {
    //     auto start = high_resolution_clock::now();
    //     //for (int i = 1; i < 10; i++)  matMultiply(csr, B, C, splitFactor);
    //     for (int i = 1; i < 10; i++)  matMultiply(csr, B, C, thr);
    //     auto end = high_resolution_clock::now();
    //     auto duration1 = duration_cast<milliseconds>(end - start).count();
    //     //cout << "Time Elapsed : " << duration1 << " ms with factor = " <<thr << endl;
    //     cout << duration1 << " " << thr << endl;
    // }
    // for (int i = 0; i < min(n, 5); ++i) {
    //     for (int j = 0; j < min(n, 5); ++j) {
    //         cout << C.at(i, j) << " ";
    //     }
    //     cout << endl;
    // }

    return 0;
}