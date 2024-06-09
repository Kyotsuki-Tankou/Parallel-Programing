//#include <iostream>
//#include <vector>
//#include <cstdlib>
//#include <ctime>
//#include <cmath>
//#include <algorithm>
//#include <fstream>
//#include <random>
//#include <unordered_map>
//#include <xmmintrin.h>
//#include <windows.h>
//#include <atomic>
//#include <mpi.h>
//#include <omp.h>
//#include <immintrin.h>
//#include <queue>
//#include <mutex>
//#include <condition_variable>
//
//using namespace std;
//
//class matCsr {
//public:
//    vector<double> values;
//    vector<int> col_indices;
//    vector<int> row_ptr;
//    int n, m, nnz;
//
//    matCsr() : n(0), m(0), nnz(0) {}
//    matCsr(int n0, int m0) : n(n0), m(m0), nnz(0) {
//        row_ptr.resize(n0 + 1, 0);
//    }
//
//    void append(int n0, int m0, double val) {
//        if (abs(val) < 1e-9) return;
//        values.push_back(val);
//        col_indices.push_back(m0);
//        row_ptr[n0 + 1]++;
//        nnz++;
//    }
//
//    void finalize() {
//        for (int i = 0; i < n; ++i) {
//            row_ptr[i + 1] += row_ptr[i];
//        }
//    }
//
//    void createMat(int n0, int m0) {
//        n = n0;
//        m = m0;
//        values.clear();
//        col_indices.clear();
//        row_ptr.resize(n0 + 1, 0);
//        nnz = 0;
//    }
//};
//
//class mat {
//public:
//    int n, m;
//    vector<vector<double>> v;
//
//    mat() : n(0), m(0) {}
//    mat(int n0, int m0) : n(n0), m(m0) {
//        createMat(n0, m0);
//    }
//
//    void createMat(int n0, int m0) {
//        n = n0;
//        m = m0;
//        v.clear();
//        v.resize(n, vector<double>(m, 0));
//    }
//
//    void matTimes(double c) {
//        for (int i = 0; i < n; i++)
//            for (int j = 0; j < m; j++)
//                v[i][j] *= c;
//    }
//
//    double findDiff(const mat& other) {
//        double curr = 0.0;
//        if (n != other.n || m != other.m) return -1;
//        for (int i = 0; i < n; i++)
//            for (int j = 0; j < m; j++) {
//                curr += abs(v[i][j] - other.v[i][j]);
//            }
//        return curr;
//    }
//
//    mat& operator=(const mat& other) {
//        if (this != &other) {
//            createMat(other.n, other.m);
//            for (int i = 0; i < n; i++)
//                for (int j = 0; j < m; j++)
//                    v[i][j] = other.v[i][j];
//        }
//        return *this;
//    }
//};
//
//void matMultiply_CSR_MPI(matCsr& X, mat& Y, mat& res, int world_rank, int world_size) {
//    int n = X.n;
//    int m = Y.m;
//    int rows_per_proc = n / world_size;
//
//    mat local_res;
//    local_res.createMat(rows_per_proc, m);
//
//    for (int i = world_rank * rows_per_proc; i < (world_rank + 1) * rows_per_proc; ++i) {
//        for (int j = X.row_ptr[i]; j < X.row_ptr[i + 1]; ++j) {
//            int col = X.col_indices[j];
//            double val = X.values[j];
//            for (int k = 0; k < m; ++k) {
//                local_res.v[i % rows_per_proc][k] += val * Y.v[col][k];
//            }
//        }
//    }
//
//    if (world_rank == 0) {
//        res.createMat(n, m);
//    }
//
//    for (int i = 0; i < rows_per_proc; ++i) {
//        MPI_Gather(local_res.v[i].data(), m, MPI_DOUBLE,
//            res.v[world_rank * rows_per_proc + i].data(), m, MPI_DOUBLE,
//            0, MPI_COMM_WORLD);
//    }
//}
//
//void dataProcess(mat& y_old, mat& y_new, double preserved = 0.8, double changed = 0.1, double masked = 0.1) {
//    random_device rd;
//    default_random_engine eng(rd());
//    uniform_real_distribution<double> distr(0, 1);
//    uniform_int_distribution<int> rdr(-1, y_old.m + 1);
//    int n0 = y_old.n, m0 = y_old.m;
//    y_new.createMat(n0, m0);
//    y_new.v.assign(n0, vector<double>(m0, -1.0));
//
//    for (int i = 0; i < y_old.n; i++) {
//        if (y_old.v[i][0] != -1) {
//            double r = distr(eng);
//            if (r < preserved) {
//                for (int j = 0; j < m0; j++) {
//                    y_new.v[i][j] = y_old.v[i][j];
//                }
//            }
//            else if (r > preserved + changed) {
//                for (int j = 0; j < m0; j++) {
//                    y_new.v[i][j] = -1;
//                }
//            }
//            else {
//                int sd = rdr(eng);
//                while (true) {
//                    if (sd < 0 || sd >= y_old.m) {
//                        sd = rdr(eng);
//                        continue;
//                    }
//                    if (y_old.v[i][sd] == 1) {
//                        sd = rdr(eng);
//                        continue;
//                    }
//
//                    for (int j = 0; j < m0; j++) {
//                        y_new.v[i][j] = 0;
//                    }
//                    y_new.v[i][sd] = 1;
//                    break;
//                }
//            }
//        }
//    }
//}
//
//void labelPropagation_MPI(matCsr& X, mat& y_label, mat& y_pred, mat& y_res, int world_rank, int world_size, int mode = 0, double alpha = 0.5, int max_iter = 1000) {
//    int n_samples = X.n;
//    int n_classes = y_label.m;
//    mat Y = y_label;
//
//    matCsr W;
//    W.createMat(n_samples, n_samples);
//
//    for (int i = 0; i < X.nnz; ++i) {
//        int row = X.col_indices[i];
//        int col = X.col_indices[i];
//        double val = X.values[i];
//        double dist = val * val;
//        double similarity = exp(-alpha * dist);
//        W.append(row, col, similarity);
//    }
//    W.finalize();
//
//    mat Y_old;
//    mat Y_new;
//    for (int iter = 0; iter < max_iter; ++iter) {
//        Y_old = Y_new;
//
//        matMultiply_CSR_MPI(W, Y, Y_new, world_rank, world_size);
//
//        if (world_rank == 0) {
//            for (int i = 0; i < n_samples; ++i) {
//                double row_sum = 0.0;
//                for (int j = 0; j < n_classes; ++j) {
//                    row_sum += Y_new.v[i][j];
//                }
//                for (int j = 0; j < n_classes; ++j) {
//                    Y_new.v[i][j] /= row_sum;
//                }
//                bool has_prior = false;
//                for (int j = 0; j < n_classes; ++j) {
//                    if (y_label.v[i][j] != -1) {
//                        Y_new.v[i][j] = y_label.v[i][j];
//                        has_prior = true;
//                        break;
//                    }
//                }
//                if (!has_prior) {
//                    for (int j = 0; j < n_classes; ++j) {
//                        Y.v[i][j] = Y_new.v[i][j];
//                    }
//                }
//            }
//
//            double diff = Y_old.findDiff(Y_new);
//            diff = diff / n_samples;
//            if (iter == 0) continue;
//            if (diff < 1e-5) {
//                break;
//            }
//        }
//
//        MPI_Bcast(&Y.v[0][0], Y.n * Y.m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    }
//
//    if (world_rank == 0) {
//        y_pred.createMat(n_samples, 1);
//        for (int i = 0; i < n_samples; ++i) {
//            int max_index = 0;
//            double max_value = Y.v[i][0];
//            for (int j = 1; j < n_classes; ++j) {
//                if (Y.v[i][j] > max_value) {
//                    max_value = Y.v[i][j];
//                    max_index = j;
//                }
//            }
//            y_pred.v[i][0] = max_index;
//        }
//
//        ofstream out1("output1.txt");
//        ofstream out2("output2.txt");
//
//        for (int i = 0; i < y_pred.n; ++i) {
//            out2 << y_pred.v[i][0] << endl;
//        }
//        for (int i = 0; i < y_res.n; ++i) {
//            out1 << y_res.v[i][0] << endl;
//        }
//    }
//}
//
//int main(int argc, char* argv[]) {
//    MPI_Init(&argc, &argv);
//
//    int world_size;
//    int world_rank;
//
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//
//    // 读取输入文件并初始化数据
//    matCsr X;
//    mat y_label;
//    // 读取并初始化X和y_label
//
//    mat y_pred;
//    mat y_res;
//    labelPropagation_MPI(X, y_label, y_pred, y_res, world_rank, world_size);
//
//    MPI_Finalize();
//    return 0;
//}
