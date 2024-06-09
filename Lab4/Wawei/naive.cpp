////g++ -fopenmp -mavx2 -lpthread -o mat1.exe mat1.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

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

class mat {
public:
	int n, m;
	vector<vector<double>> v;

	mat() : n(0), m(0) {}
	mat(int n0, int m0) : n(n0), m(m0) {
		createMat(n0, m0);
	}

	void createMat(int n0, int m0) {
		n = n0;
		m = m0;
		v.clear();
		v.resize(n, vector<double>(m, 0));
	}

	void matTimes(double c) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				v[i][j] *= c;
	}

	double findDiff(const mat& other) {
		double curr = 0.0;
		if (n != other.n || m != other.m) return -1;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) {
				curr += abs(v[i][j] - other.v[i][j]);
			}
		return curr;
	}

	mat& operator=(const mat& other) {
		if (this != &other) {
			createMat(other.n, other.m);
			for (int i = 0; i < n; i++)
				for (int j = 0; j < m; j++)
					v[i][j] = other.v[i][j];
		}
		return *this;
	}
};

//void matMultiply_avx_omp(matCsr& x1, mat& x2, mat& res, int nThreads = 4) {
//	omp_set_num_threads(nThreads);
//	int n = x1.n, m = x2.m;
//	res.createMat(n, m);
//#pragma omp parallel for schedule(dynamic)
//	for (int i = 0; i < n; ++i) {
//		for (int j = x1.row_ptr[i]; j < x1.row_ptr[i + 1]; ++j) {
//			int col = x1.col_indices[j];
//			double val = x1.values[j];
//			for (int k = 0; k + 4 < m; k += 4) {
//				__m256d sum = _mm256_loadu_pd(&res.v[i][k]);
//				__m256d x2_v = _mm256_loadu_pd(&x2.v[col][k]);
//				__m256d x1_v = _mm256_set1_pd(val);
//				sum = _mm256_add_pd(sum, _mm256_mul_pd(x1_v, x2_v));
//				_mm256_storeu_pd(&res.v[i][k], sum);
//			}
//			for (int k = m - m % 4; k < m; k++) {
//				res.v[i][k] += val * x2.v[col][k];
//			}
//		}
//	}
//	 //for(int i=0;i<10;i++) cout<<res.v[i][0]<<" ";
//	 //cout<<endl;
//}

void matMultiply_mpi1(matCsr& x1, mat& x2, mat& res,int threadNum=4) {
	int n = x1.n, m = x2.m;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int chunk = n / size;
	int start_row = rank * chunk;
	int end_row = (rank == size - 1) ? n : start_row + chunk;

	res.createMat(n, m);

    omp_set_num_threads(threadNum);
#pragma omp parallel for schedule(dynamic)
	for (int i = start_row; i < end_row; ++i) {
		for (int j = x1.row_ptr[i]; j < x1.row_ptr[i + 1]; ++j) {
			int col = x1.col_indices[j];
			double val = x1.values[j];
			for (int k = 0; k < m; ++k) {
				res.v[i][k] += val * x2.v[col][k];
			}
		}
	}

	vector<double> row_buffer(m);
	for (int i = 0; i < n; ++i) {
		int owner = i / chunk;
		if (rank == owner) {
			row_buffer.assign(res.v[i].begin(), res.v[i].end());
		}
		MPI_Bcast(&row_buffer[0], m, MPI_DOUBLE, owner, MPI_COMM_WORLD);
		if (rank != owner) {
			res.v[i].assign(row_buffer.begin(), row_buffer.end());
		}
	}
	//if (rank == 0) {
	//	cout << "First 10 elements of the first row: ";
	//	for (int i = 0; i < 10; i++) {
	//		cout << res.v[i][0] << " ";
	//	}
	//	cout << endl;
	//}
}

void matMultiply_mpi2(matCsr& x1, mat& x2, mat& res,int threadNum=4) {
	int n = x1.n, m = x2.m;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int chunk = n / size;
	int start_row = rank * chunk;
	int end_row = (rank == size - 1) ? n : start_row + chunk;

	res.createMat(n, m);

omp_set_num_threads(threadNum);
#pragma omp parallel for schedule(dynamic)
	for (int i = start_row; i < end_row; ++i) {
		for (int j = x1.row_ptr[i]; j < x1.row_ptr[i + 1]; ++j) {
			int col = x1.col_indices[j];
			double val = x1.values[j];
			for (int k = 0; k < m; ++k) {
				res.v[i][k] += val * x2.v[col][k];
			}
		}
	}

	if (rank != 0) {
		for (int i = start_row; i < end_row; ++i) {
			MPI_Send(&res.v[i][0], m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
	}
	else {
		for (int r = 1; r < size; ++r) {
			int r_start_row = r * chunk;
			int r_end_row = (r == size - 1) ? n : r_start_row + chunk;
			for (int i = r_start_row; i < r_end_row; ++i) {
				MPI_Recv(&res.v[i][0], m, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	}
	//if (rank == 0) {
	//	cout << "First 10 elements of the first row: ";
	//	for (int i = 0; i < 10; i++) {
	//		cout << res.v[i][0] << " ";
	//	}
	//	cout << endl;
	//}
}

void matMultiply_cyclic(matCsr& x1, mat& x2, mat& res, int threadNum, int rank, int size) {
	int n = x1.n, m = x2.m;

	res.createMat(n, m);

omp_set_num_threads(threadNum);
#pragma omp parallel for schedule(dynamic)
	for (int i = rank; i < n; i += size) {
		for (int j = x1.row_ptr[i]; j < x1.row_ptr[i + 1]; ++j) {
			int col = x1.col_indices[j];
			double val = x1.values[j];
			for (int k = 0; k < m; ++k) {
				res.v[i][k] += val * x2.v[col][k];
			}
		}
	}

	if (rank != 0) {
		for (int i = rank; i < n; i += size) {
			MPI_Send(&res.v[i][0], m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
	}
	else {
		for (int r = 1; r < size; ++r) {
			for (int i = r; i < n; i += size) {
				MPI_Recv(&res.v[i][0], m, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	}
}
void matMultiply(matCsr& x1, mat& x2, mat& res)
{
   int n = x1.n, m = x2.m;
   res.createMat(n, m);
   for (int i = 0; i < n; i++)
   {
       for (int j = x1.row_ptr[i]; j < x1.row_ptr[i + 1]; j++)
       {
           int col = x1.col_indices[j];
           double val = x1.values[j];
           for (int k = 0; k < m; k++)
           {
               res.v[i][k] += val * x2.v[col][k];
           }
       }
   }
   // for(int i=0;i<10;i++)  cout<<res.v[i][0]<<" ";
   // cout<<endl;
}

void generateRandomSparseMatrix(matCsr& csr, int n, int m, int nnz) {
	csr.createMat(n, m);
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> dis(0, n - 1);
	uniform_real_distribution<> dis_val(0, 100);

	for (int i = 0; i < nnz; ++i) {
		int row = dis(gen);
		int col = dis(gen);
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
			dense.v[i][j] = dis(gen);
		}
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
    for(int tnum=1;tnum<=1;tnum++)
    {
        for (int n0 = 8192; n0 <= 8192; n0 <<= 1)
        {
            for (int t0 = 128; t0 <= 128; t0 <<= 1)
            {
                matCsr csr;
                mat dense, result;
                int rank,size;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &size);
                generateRandomSparseMatrix(csr, n0, n0, n0*t0/1024*n0);
                generateRandomMatrix(dense, n0, n0);

                auto start = high_resolution_clock::now();
                for (int i = 1; i <= 10; ++i) matMultiply(csr, dense, result);
                auto end = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(end - start).count();

                if (rank == 0) {
                    cout << 0 <<" "<<tnum <<" " << n0 << " " << t0 / 1024.0 << " " << duration << endl;
                    //matMultiply_avx_omp(csr, dense, result);
                }

            }
        }
    }
	
    // ------------------------------
    // for(int tnum=1;tnum<=1;tnum++)
    // {
    //     for (int n0 = 128; n0 <= 2048; n0 <<= 1)
    //     {
    //         for (int t0 = n0 * n0 / 100; t0 <= n0 * n0 / 100; t0 <<= 1)
    //         {
    //             matCsr csr;
    //             mat dense, result;
    //             int rank,size;
    //             MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //             MPI_Comm_size(MPI_COMM_WORLD, &size);
    //             generateRandomSparseMatrix(csr, n0, n0, t0);
    //             generateRandomMatrix(dense, n0, n0);

    //             auto start = high_resolution_clock::now();
    //             for (int i = 1; i <= 10; ++i) matMultiply(csr, dense, result);
    //             auto end = high_resolution_clock::now();
    //             auto duration = duration_cast<milliseconds>(end - start).count();

    //             if (rank == 0) {
    //                 cout << 0 <<" "<<tnum <<" " << n0 << " " << t0 << " " << duration << endl;
    //                 //matMultiply_avx_omp(csr, dense, result);
    //             }

    //         }
    //     }
    // }
	MPI_Finalize();

	return 0;
}