// g++ -fopenmp -mavx2 -lpthread -o LPA.exe LPA.cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <random>
#include <unordered_map>
#include <xmmintrin.h>
#include <fstream>
#include <windows.h>
#include <pthread.h>
#include <atomic>
#include <omp.h>
#include <immintrin.h>
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace std;

class matCsr {
public:
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int n, m;

    matCsr() : n(0), m(0) {}

    void createMat(int n0, int m0) {
        n = n0;
        m = m0;
        values.clear();
        col_indices.clear();
        row_ptr.assign(n + 1, 0);
    }

    void append(int n0, int m0, double val) {
        if (abs(val) < 1e-9) return;
        values.push_back(val);
        col_indices.push_back(m0);
        row_ptr[n0 + 1]++;
    }

    void finalize() {
        for (int i = 0; i < n; ++i) {
            row_ptr[i + 1] += row_ptr[i];
        }
    }

    void matTimes(double c) {
        if (c == 0) {
            values.clear();
            col_indices.clear();
            row_ptr.assign(n + 1, 0);
            return;
        }
        for (auto& val : values) {
            val *= c;
        }
    }
};

class mat {
public:
    int n, m;
    std::vector<std::vector<double>> v;
    void createMat(int n0, int m0);
    void matTimes(double c);
    double findDiff(mat);
    mat()
    {
        m = 1, n = 1;
        createMat(1, 1);
    }
    mat(int n0, int m0)
    {
        m = m0, n = n0;
        createMat(n0, m0);
    }
    mat& operator=(const mat& other);
    mat(const mat& other) {
        n = other.n;
        m = other.m;
        createMat(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                v[i][j] = other.v[i][j];
            }
        }
    }
    double getval(int x, int y) { return v[x][y]; }
    void editval(int x, int y, double val) { v[x][y] = val; }
    void setneg() { v.clear(); v.resize(n, std::vector<double>(m, -1.0)); }
    void editval2(int x, int y) { for (int i = 0; i < m; i++)v[x][i] = 0.0; v[x][y] = 1.0; }
};

bool isZero(double a)
{
    return abs(a) < 0.00000001 ? true : false;
}

void mat::createMat(int n0, int m0)
{
    n = n0, m = m0;
    v.clear();
    v.resize(n, std::vector<double>(m, 0));
    return;
}

void mat::matTimes(double c)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            v[i][j] *= c;
    return;
}

double mat::findDiff(mat other)
{
    double curr = 0.0;
    if (n != other.n || m != other.m)  return -1;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            curr += abs(v[i][j] - other.v[i][j]);
        }
    return curr;
}

mat& mat::operator=(const mat& other)
{
    if (this != &other)
    {
        createMat(other.n, other.m);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                v[i][j] = other.v[i][j];
    }

    return *this;
}

void matMultiply_CSR(matCsr& x1, mat& x2, mat& res) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = x1.row_ptr[i]; j < x1.row_ptr[i + 1]; ++j) {
            int col = x1.col_indices[j];
            double val = x1.values[j];
            for (int k = 0; k < m; ++k) {
                res.v[i][k] += val * x2.v[col][k];
            }
        }
    }
}

void matMultiply_AVX_CSR(matCsr& x1, mat& x2, mat& res) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = x1.row_ptr[i]; j < x1.row_ptr[i + 1]; ++j) {
            int col = x1.col_indices[j];
            double val = x1.values[j];
            for (int k = 0; k + 4 < m; k += 4) {
                __m256d sum = _mm256_loadu_pd(&res.v[i][k]);
                __m256d x2_v = _mm256_loadu_pd(&x2.v[col][k]);
                __m256d x1_v = _mm256_set1_pd(val);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(x1_v, x2_v));
                _mm256_storeu_pd(&res.v[i][k], sum);
            }
            for (int k = m - m % 4; k < m; k++) {
                res.v[i][k] += val * x2.v[col][k];
            }
        }
    }
}

void matMultiply(mat& x1, mat& x2, mat& res)
{
    int n = x1.n, m = x2.m, k = x1.m;

    res.createMat(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            for (int t = 0; t < k; t++)
                res.v[i][j] += x1.v[i][t] * x2.v[t][j];
}

void dataProcess(mat& y_old, mat& y_new, double preserved = 0.8, double changed = 0.1, double masked = 0.1)
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(0, 1);
    std::uniform_int_distribution<int> rdr(-1, y_old.m + 1);
    int n0 = y_old.n, m0 = y_old.m;
    y_new.createMat(n0, m0);
    y_new.setneg();

    for (int i = 0; i < y_old.n; i++)
    {
        if (y_old.v[i][0] != -1)
        {
            double r = distr(eng);
            if (r < preserved)  for (int j = 0; j < m0; j++)  y_new.v[i][j] = y_old.v[i][j];
            else if (r > preserved + changed) for (int j = 0; j < m0; j++)  y_new.v[i][j] = -1;
            else
            {
                int sd = rdr(eng);
                while (true)
                {
                    if (sd < 0 || sd >= y_old.m) { sd = rdr(eng); continue; }
                    if (y_old.v[i][sd] == 1) { sd = rdr(eng); continue; }

                    for (int j = 0; j < m0; j++)  y_new.v[i][j] = 0;
                    y_new.v[i][sd] = 1;
                    break;
                }
            }
        }
    }
}

void labelPropagation(matCsr& X, mat& y_label, mat& y_pred, mat& y_res, int mode = 0, double alpha = 0.5, int max_iter = 1000, int sort_thread = 8, int sort_mode = 1)
{
    int n_samples = X.n;
    int n_classes = y_label.m;
    double diff2 = 0, diff1 = 0, diff = 0;
    mat Y = y_label;
    matCsr W;
    W.createMat(n_samples, n_samples);

    for (int i = 0; i < X.row_ptr[n_samples]; ++i) {
        int row = X.col_indices[i];
        int col = X.col_indices[i];
        double val = X.values[i];
        double dist = val * val;
        double similarity = exp(-alpha * dist);
        W.append(row, col, similarity);
    }
    W.finalize();

    mat Y_old;
    mat Y_new;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        Y_old = Y_new;
        switch (mode)
        {
            case 1:matMultiply_AVX_CSR(W,Y,Y_new);
            default:
                matMultiply_CSR(W, Y, Y_new);
        }

        for (int i = 0; i < n_samples; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < n_classes; ++j) {
                row_sum += Y_new.v[i][j];
            }
            for (int j = 0; j < n_classes; ++j) {
                Y_new.v[i][j] /= row_sum;
            }
            bool has_prior = false;
            for (int j = 0; j < n_classes; ++j) {
                if (y_label.v[i][j] != -1) {
                    Y_new.v[i][j] = y_label.v[i][j];
                    has_prior = true;
                    break;
                }
            }
            if (!has_prior) {
                for (int j = 0; j < n_classes; ++j) {
                    Y.v[i][j] = Y_new.v[i][j];
                }
            }
        }

        double diff = Y_old.findDiff(Y_new);
        diff = diff / n_samples;
        if (iter == 0) continue;
        if (diff < 1e-5) {
            break;
        }
    }

    y_pred.createMat(n_samples, 1);
    for (int i = 0; i < n_samples; ++i) {
        int max_index = 0;
        double max_value = Y.v[i][0];
        for (int j = 1; j < n_classes; ++j) {
            if (Y.v[i][j] > max_value) {
                max_value = Y.v[i][j];
                max_index = j;
            }
        }
        y_pred.v[i][0] = max_index;
    }
    ofstream out1("res_o.txt");
    for (int i = 0; i < y_pred.n; ++i) {
        out1 << y_pred.v[i][0] << endl;
    }
}

int main()
{
    srand(time(0));
    ifstream in1("out.txt");
    ifstream in2("out2.txt");
    ofstream out1("res.txt");
    ofstream outTime("Time.txt");

    int n0, m0, m1;
    in1 >> n0;
    in2 >> m0 >> m1;

    matCsr X;
    X.createMat(m0, m0);

    mat y_label(m0, m1);

    int x, y;
    double z;
    int temp = 0;
    ios::sync_with_stdio(false);

    for (int i = 0; i < n0; i++) {
        in1 >> x >> y >> z;
        X.append(x, y, z);
    }
    X.finalize();

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int> rdr(0, 10);

    for (int i = 0; i < m0; i++) {
        y = rdr(eng);
        in2 >> x;
        if (y == 1) {
            temp++;
            for (int r = 0; r < m1; r++) y_label.v[i][r] = (r == x) ? 1 : 0;
        } else {
            for (int r = 0; r < m1; r++) y_label.v[i][r] = -1;
        }
    }

    std::cout << m0 << " " << temp << std::endl << std::endl;

    mat res;
    mat y_pred;
    mat y_res;
    mat y_label_new(y_label.n, y_label.m);
    y_label_new = y_label;

    dataProcess(y_label, y_label_new);

    for (int t = 16; t <= 16;) {
        cout << "sorT = " << t << endl;
        for (int s = 1; s <= 1; s++) {
            for (int i = 1; i <= 2; i++) {
                LARGE_INTEGER head, tail, freq;
                QueryPerformanceFrequency(&freq);
                QueryPerformanceCounter(&head);
                for (int j = 0; j < 5; j++) labelPropagation(X, y_label_new, y_pred, y_res, i, 0.5, 1000, t, s);
                QueryPerformanceCounter(&tail);
                outTime << (1 << (i % 10 - 1)) << "," << i / 10 + 1 << "," << (tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart << endl;
                cout << i << ": Time Elapsed: " << (tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart << "ms" << endl;
                if (i % 10 == 0) cout << endl;
            }
        }
        t = t ? t << 1 : 1;
    }
}
