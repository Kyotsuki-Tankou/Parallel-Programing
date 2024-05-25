//g++ -fopenmp -mavx2 -lpthread -o LPA.exe LPA.cpp
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

// #include "pybind11/pybind11.h"
// #include "pybind11/stl.h"
// namespace py = pybind11;
// using namespace py;
using namespace std;
class matCoo {
public:
    struct elems {
        double v;
        int row, col;
        bool operator<(const elems& other) const {
            return this->row < other.row || this->row == other.row && this->col < other.col;
        }
        bool operator<(int i) const {
            return row < i;
        }
    };
    std::vector<elems>elem;
    int n, m;
    int totalElements;
    void createMat(int n0, int m0);
    void matTimes(double c);
    void append(int n0, int m0, double val);
    matCoo& operator=(const matCoo& other);
    matCoo()
    {
        elem.clear();
        n = 0, m = 0;
        totalElements = 0;
    }
    matCoo(int n0, int m0)
    {
        elem.clear();
        n = n0, m = n0;
        totalElements = 0;
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
    return abs(a) < 0.0000000001 ? true : false;
}

struct ThreadArgSort {
    std::vector<matCoo::elems>::iterator start;
    std::vector<matCoo::elems>::iterator end;
};
void* thread_sort(void* arg) {
    ThreadArgSort* threadArg = (ThreadArgSort*)arg;
    std::sort(threadArg->start, threadArg->end);
    return NULL;
}
void multi_thread_sort(std::vector<matCoo::elems>& vec, int n_threads) {
    std::vector<pthread_t> threads(n_threads);
    std::vector<ThreadArgSort> args(n_threads);
    int len = vec.size() / n_threads;
    int remainder = vec.size() % n_threads;
    for (int i = 0; i < n_threads; i++) {
        args[i].start = vec.begin() + i * len + std::min(i, remainder);
        if (i < remainder) {
            args[i].end = args[i].start + len + 1;
        } else {
            args[i].end = args[i].start + len;
        }
        pthread_create(&threads[i], NULL, thread_sort, &args[i]);
    }
    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    for (int i = 1; i < n_threads; i++) {
        std::inplace_merge(vec.begin(), args[i].start, args[i].end);
    }
}

void openmp_sort(std::vector<matCoo::elems>&vec,int n_threads)
{
    omp_set_num_threads(n_threads);
    std::vector<matCoo::elems>::iterator start;
    std::vector<matCoo::elems>::iterator end;
    int len=vec.size()/n_threads;
    int reminder=vec.size()%n_threads;
#pragma omp parallel for schedule(dynamic)
    for(int i=0;i<n_threads;i++)
    {
        start=vec.begin()+i*len+min(i,reminder);
        if(i<reminder)  end=start+len+1;
        else  end=start+len;
        sort(start,end);
    }

    for(int i=1;i<n_threads;i++)
    {
        start=vec.begin()+i*len+min(i,reminder);
        if(i<reminder)  end=start+len+1;
        else  end=start+len;
        inplace_merge(vec.begin(), start, end);
    }
}
void matCoo::createMat(int n0, int m0)
{
    n = n0;
    m = m0;
    totalElements = 0;
    elem.clear();
    return;
}
void matCoo::matTimes(double c)
{
    long long num = elem.size();
    if (c == 0)
    {
        elem.clear();
        n = 0, m = 0;
        totalElements = 0;
        return;
    }
    for (int i = 0; i < num; i++)
    {
        elem[i].v *= c;
    }
    return;
}
void matCoo::append(int n0, int m0, double val)
{
    if (isZero(val))  return;
    elems newElem;
    newElem.row = n0;
    newElem.col = m0;
    newElem.v = val;
    elem.push_back(newElem);
    totalElements++;
    return;
}
matCoo& matCoo::operator=(const matCoo& other)
{
    this->elem = other.elem;
    this->m = other.m;
    this->n = other.n;
    this->totalElements = other.totalElements;
    return *this;
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
void matMultiply(matCoo& x1, mat& x2, mat& res) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);
    for (int i = 0; i < n; i++) {
        long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            int q = p;
            while (q < x1.elem.size() && x1.elem[q].row == i) {
                sum += x1.elem[q].v * x2.v[x1.elem[q].col][j];
                q++;
            }
            res.v[i][j] = sum;
        }
    }
    return;
}
void matMultiplyBlock(matCoo& x1, mat& x2, mat& res, int blockSize = 64) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < m; jj += blockSize) {
            for (int i = ii; i < min(ii + blockSize, n); i++) {
                long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
                for (int j = jj; j < min(jj + blockSize, m); j++) {
                    double sum = 0.0;
                    int q = p;
                    while (q < x1.elem.size() && x1.elem[q].row == i) {
                        sum += x1.elem[q].v * x2.v[x1.elem[q].col][j];
                        q++;
                    }
                    res.v[i][j] = sum;
                }
            }
        }
    }
}
void matMultiply_AVX(matCoo& x1, mat& x2, mat& res, int blockSize = 64) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);
    
    // for (int ii = 0; ii < n; ii += blockSize) {
    //     for (int jj = 0; jj < m; jj += blockSize) {
    //         for (int i = ii; i < min(ii + blockSize, n); i++) {
    //             long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), matCoo::elems{ 0, i, -1 }) - x1.elem.begin();
    //             for (int j = jj; j < min(jj + blockSize, m); j += 4) {
    //                 if (j + 4 <= m) {
    //                     __m256d sum = _mm256_setzero_pd();
    //                     int q = p;
    //                     while (q < x1.elem.size() && x1.elem[q].row == i) {
    //                         __m256d a = _mm256_set1_pd(x1.elem[q].v);
    //                         __m256d b = _mm256_load_pd(&x2.v[x1.elem[q].col][j]);
    //                         sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
    //                         q++;
    //                     }
    //                     double* f = (double*)&sum;
    //                     cout<<888<<endl;
    //                     for (int k = 0; k < 4; k++) {
    //                         res.v[i][j + k] = f[k];
    //                     }
    //                 }
    //                 else {
    //                     cout<<555<<endl;
    //                     for (int j2 = j; j2 < m; j2++) {
    //                         double sum = 0.0;
    //                         int q = p;
    //                         while (q < x1.elem.size() && x1.elem[q].row == i) {
    //                             double a = x1.elem[q].v;
    //                             double b = x2.v[x1.elem[q].col][j2];
    //                             sum += a * b;
    //                             q++;
    //                         }
    //                         res.v[i][j2] = sum;
    //                     }
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    // }
    for (int i = 0; i < n; i++) {
        long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
        for (int j = 0; j + 4 < m; j += 4) {
            __m256d sum = _mm256_setzero_pd();
            int q = p;
            while (q < x1.elem.size() && x1.elem[q].row == i) {
                __m256d x2_v = _mm256_loadu_pd(&x2.v[x1.elem[q].col][j]);
                __m256d x1_v = _mm256_set1_pd(x1.elem[q].v);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(x1_v, x2_v));
                q++;
            }
            _mm256_storeu_pd(&res.v[i][j], sum);
        }
        for (int j = m - m % 4; j < m; j++) {
            double sum = 0.0;
            int q = p;
            while (q < x1.elem.size() && x1.elem[q].row == i) {
                sum += x1.elem[q].v * x2.v[x1.elem[q].col][j];
                q++;
            }
            res.v[i][j] = sum;
        }
    }
    return;
}

struct ThreadDataMul {
    matCoo* x1;
    mat* x2;
    mat* res;
    int startRow;
    int endRow;
    int m;
};

void* matMultiply_AVX_Thread(void* arg) {
    ThreadDataMul* data = (ThreadDataMul*)arg;
    for (int i = data->startRow; i < data->endRow; i++) {
        long long p = std::lower_bound(data->x1->elem.begin(), data->x1->elem.end(), i) - data->x1->elem.begin();
        for (int j = 0; j + 4 < data->m; j += 4) {
            __m256d sum = _mm256_setzero_pd();
            int q = p;
            while (q < data->x1->elem.size() && data->x1->elem[q].row == i) {
                __m256d x2_v = _mm256_loadu_pd(&data->x2->v[data->x1->elem[q].col][j]);
                __m256d x1_v = _mm256_set1_pd(data->x1->elem[q].v);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(x1_v, x2_v));
                q++;
            }
            _mm256_storeu_pd(&data->res->v[i][j], sum);
        }
        for (int j = data->m - data->m % 4; j < data->m; j++) {
            double sum = 0.0;
            int q = p;
            while (q < data->x1->elem.size() && data->x1->elem[q].row == i) {
                sum += data->x1->elem[q].v * data->x2->v[data->x1->elem[q].col][j];
                q++;
            }
            data->res->v[i][j] = sum;
        }
    }
    return NULL;
}

void matMultiply_AVX_pth(matCoo& x1, mat& x2, mat& res, int ThreadNum=8) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);
    
    std::vector<pthread_t> threads(ThreadNum);
    std::vector<ThreadDataMul> tData(ThreadNum);

    int rowsPerThread = (n + ThreadNum - 1) / ThreadNum;

    for (int i = 0; i < ThreadNum; i++) {
        int startRow = i * rowsPerThread;
        int endRow = std::min(startRow + rowsPerThread, n);
        tData[i] = {&x1, &x2, &res, startRow, endRow, m};
        pthread_create(&threads[i], NULL, matMultiply_AVX_Thread, &tData[i]);
    }

    for (int i = 0; i < ThreadNum; i++) {
        pthread_join(threads[i], NULL);
    }
}

void matMultiplyOmp(matCoo& x1, mat& x2, mat& res, int ThreadNum) {
    omp_set_num_threads(ThreadNum);

    int n = x1.n, m = x2.m;
    res.createMat(n, m);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            int q = p;
            while (q < x1.elem.size() && x1.elem[q].row == i) {
                sum += x1.elem[q].v * x2.v[x1.elem[q].col][j];
                q++;
            }
            res.v[i][j] = sum;
        }
    }
    return;
}

void matMultiplyOmp1(matCoo& x1, mat& x2, mat& res, int ThreadNum) {
    omp_set_num_threads(ThreadNum);

    int n = x1.n, m = x2.m;
    res.createMat(n, m);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            int q = p;
            while (q < x1.elem.size() && x1.elem[q].row == i) {
                sum += x1.elem[q].v * x2.v[x1.elem[q].col][j];
                q++;
            }
            res.v[i][j] = sum;
        }
    }
    return;
}
void matMultiplyOmp2(matCoo& x1, mat& x2, mat& res, int ThreadNum) {
    omp_set_num_threads(ThreadNum);

    int n = x1.n, m = x2.m;
    res.createMat(n, m);

#pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            int q = p;
            while (q < x1.elem.size() && x1.elem[q].row == i) {
                sum += x1.elem[q].v * x2.v[x1.elem[q].col][j];
                q++;
            }
            res.v[i][j] = sum;
        }
    }
    return;
}
void matMultiplyOmpAvx(matCoo& x1, mat& x2, mat& res, int ThreadNum) {
    omp_set_num_threads(ThreadNum);

    int n = x1.n, m = x2.m;
    res.createMat(n, m);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
        for (int j = 0; j + 4 < m; j += 4) {
            __m256d sum = _mm256_setzero_pd();
            int q = p;
            while (q < x1.elem.size() && x1.elem[q].row == i) {
                __m256d x2_v = _mm256_loadu_pd(&x2.v[x1.elem[q].col][j]);
                __m256d x1_v = _mm256_set1_pd(x1.elem[q].v);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(x1_v, x2_v));
                q++;
            }
            _mm256_storeu_pd(&res.v[i][j], sum);
        }
        for (int j = m - m % 4; j < m; j++) {
            double sum = 0.0;
            int q = p;
            while (q < x1.elem.size() && x1.elem[q].row == i) {
                sum += x1.elem[q].v * x2.v[x1.elem[q].col][j];
                q++;
            }
            res.v[i][j] = sum;
        }
    }
    return;
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
void rectify(matCoo& x, mat& y_label, mat& y_ori, mat& y_new,int sort_thread=4,int sort_mode=1)
{
    y_new = y_ori;
    std::unordered_map<int, int>p;
    // std::sort(x.elem.begin(), x.elem.end());
    // multi_thread_sort(x.elem,sort_thread);
    if(sort_thread<=0)  std::sort(x.elem.begin(), x.elem.end());
    else if (sort_mode==1)  multi_thread_sort(x.elem,sort_thread);
    else  openmp_sort(x.elem,sort_thread);
    int p1, p2;
    int temp = 0;
    for (int i = 0; i < y_label.n; i++)
    {
        if (y_label.v[i][0] != -1)
        {
            p1 = std::lower_bound(x.elem.begin(), x.elem.end(), i) - x.elem.begin();
            p2 = std::lower_bound(x.elem.begin(), x.elem.end(), i + 1) - x.elem.begin();
            p.clear();
            p[y_ori.v[i][0]] = 1;
            int maxx = 1, maxxj = y_ori.v[i][0], tmp = 0, val;
            for (int j = p1; j < p2; j++)
            {
                val = y_ori.v[x.elem[j].col][0];
                if (p[val])  p[val]++;
                else  p[val] = 1;
                maxxj = (p[val] > maxx) ? val : maxxj;
                maxx = max(p[val], maxx);
            }
            y_new.v[i][0] = maxxj;
            if (maxxj != y_ori.v[i][0])  temp++;
        }
    }
    ofstream out2("res_r.txt");
    for (int i = 0; i < y_new.n; ++i)
    {
        out2 << y_new.v[i][0] << endl;
    }
    //std::cout << temp << std::endl << std::endl;
    return;
}
void labelPropagation(matCoo& X, mat& y_label, mat& y_pred, mat& y_res, int mode = 0, double alpha = 0.5, int max_iter = 1000,int sort_thread=8,int sort_mode=1)
{
    int n_samples = X.n;
    int n_classes = y_label.m;
    double diff2 = 0, diff1 = 0, diff = 0;
    // Initialize
    mat Y = y_label;
    // Compute similarity matrix
    matCoo W(n_samples, n_samples);
    for (int i = 0; i < X.totalElements; ++i) {
        int row = X.elem[i].row;
        int col = X.elem[i].col;
        double val = X.elem[i].v;
        double dist = val * val;
        double similarity = exp(-alpha * dist);
        W.append(row, col, similarity);
        // W.row.push_back(row);
        // W.col.push_back(col);
        // W.v.push_back(similarity);
    }
    W.totalElements = int(W.elem.size());
    if(sort_thread<=0)  std::sort(W.elem.begin(), W.elem.end());
    else if (sort_mode==1)  multi_thread_sort(W.elem,sort_thread);
    else  openmp_sort(W.elem,sort_thread);
    mat Y_old;
    mat Y_new;
    // LPA
    for (int iter = 0; iter < max_iter; ++iter) {
        //cout << iter << endl;
        Y_old = Y_new;
        switch (mode)
        {
        // case 1:matMultiplyOmp(W, Y, Y_new, 1); break;
        // case 2:matMultiplyOmp(W, Y, Y_new, 2); break;
        // case 3:matMultiplyOmp(W, Y, Y_new, 4); break;
        // case 4:matMultiplyOmp(W, Y, Y_new, 8); break;
        // case 5:matMultiplyOmp(W, Y, Y_new, 16); break;
        // case 6:matMultiplyOmp(W, Y, Y_new, 32); break;
        // case 7:matMultiplyOmp(W, Y, Y_new, 64); break;
        // case 8:matMultiplyOmp(W, Y, Y_new, 128); break;
        // case 9:matMultiplyOmp(W, Y, Y_new, 256); break;
        // case 10:matMultiplyOmp(W, Y, Y_new, 512); break;

        // case 11:matMultiplyOmp1(W, Y, Y_new, 1); break;
        // case 12:matMultiplyOmp1(W, Y, Y_new, 2); break;
        // case 13:matMultiplyOmp1(W, Y, Y_new, 4); break;
        // case 14:matMultiplyOmp1(W, Y, Y_new, 8); break;
        default:matMultiplyOmp1(W, Y, Y_new, 16); break;
        // case 16:matMultiplyOmp1(W, Y, Y_new, 32); break;
        // case 17:matMultiplyOmp1(W, Y, Y_new, 64); break;
        // case 18:matMultiplyOmp1(W, Y, Y_new, 128); break;
        // case 19:matMultiplyOmp1(W, Y, Y_new, 256); break;
        // case 20:matMultiplyOmp1(W, Y, Y_new, 512); break;

        // case 21:matMultiplyOmp2(W, Y, Y_new, 1); break;
        // case 22:matMultiplyOmp2(W, Y, Y_new, 2); break;
        // case 23:matMultiplyOmp2(W, Y, Y_new, 4); break;
        // case 24:matMultiplyOmp2(W, Y, Y_new, 8); break;
        // case 25:matMultiplyOmp2(W, Y, Y_new, 16); break;
        // case 26:matMultiplyOmp2(W, Y, Y_new, 32); break;
        // case 27:matMultiplyOmp2(W, Y, Y_new, 64); break;
        // case 28:matMultiplyOmp2(W, Y, Y_new, 128); break;
        // case 29:matMultiplyOmp2(W, Y, Y_new, 256); break;
        // case 30:matMultiplyOmp2(W, Y, Y_new, 512); break;
        // case 11:matMultiplyOmpAvx(W, Y, Y_new, 1); break;
        // case 12:matMultiplyOmpAvx(W, Y, Y_new, 2); break;
        // case 13:matMultiplyOmpAvx(W, Y, Y_new, 4); break;
        // case 14:matMultiplyOmpAvx(W, Y, Y_new, 8); break;
        // case 15:matMultiplyOmpAvx(W, Y, Y_new, 16); break;
        // case 16:matMultiplyOmpAvx(W, Y, Y_new, 32); break;
        // case 17:matMultiplyOmpAvx(W, Y, Y_new, 64); break;
        // case 18:matMultiplyOmpAvx(W, Y, Y_new, 128); break;
        // case 19:matMultiplyOmpAvx(W, Y, Y_new, 256); break;
        // case 20:matMultiplyOmpAvx(W, Y, Y_new, 512); break;

        // case 21:matMultiply_AVX_pth(W, Y, Y_new, 1); break;
        // case 22:matMultiply_AVX_pth(W, Y, Y_new, 2); break;
        // case 23:matMultiply_AVX_pth(W, Y, Y_new, 4); break;
        // case 24:matMultiply_AVX_pth(W, Y, Y_new, 8); break;
        // case 25:matMultiply_AVX_pth(W, Y, Y_new, 16); break;
        // case 26:matMultiply_AVX_pth(W, Y, Y_new, 32); break;
        // case 27:matMultiply_AVX_pth(W, Y, Y_new, 64); break;
        // case 28:matMultiply_AVX_pth(W, Y, Y_new, 128); break;
        // case 29:matMultiply_AVX_pth(W, Y, Y_new, 256); break;
        // case 30:matMultiply_AVX_pth(W, Y, Y_new, 512); break;

        // case 14:matMultiply_AVX_pth(W, Y, Y_new); break;
        // case 31:matMultiplyBlock(W, Y, Y_new); break;
        // case 32:matMultiply_AVX(W, Y, Y_new); break;
        
        // case 15:matMultiplyOmpAvx(W, Y, Y_new, 128); break;
        // default:matMultiply(W, Y, Y_new);
        // default:matMultiplyBlock_AVX(W, Y, Y_new); break;
        }

        //matMultiplyBlock(W, Y, Y_new);
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
        diff = Y_old.findDiff(Y_new);

        diff = diff / n_samples;
        if (iter == 0)  continue;
        if (diff < 1e-5) {
            break;
        }
    }

    // result
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
    //rectify(X, y_label, y_pred, y_res,sort_thread);
    ofstream out1("res_o.txt");
    for (int i = 0; i < y_res.n; ++i)
    {
        out1 << y_res.v[i][0] << endl;
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
    matCoo X(m0, m0);
    mat y_label(m0, m1);
    int x, y;
    double z;
    int temp = 0;
    ios::sync_with_stdio(false);
    for (int i = 0; i < n0; i++)
    {
        in1 >> x >> y >> z;
        X.append(x, y, z);
    }
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int> rdr(0, 10);
    for (int i = 0; i < m0; i++)
    {
        y = rdr(eng);
        in2 >> x;
        if (y == 1)
        {
            temp++;
            for (int r = 0; r < m1; r++) y_label.v[i][r] = (r == x) ? 1 : 0;
        }

        else
            for (int r = 0; r < m1; r++)  y_label.v[i][r] = -1;

    }
    std::cout << m0 << " " << temp << std::endl << std::endl;
    mat res;
    mat y_pred;
    mat y_res;
    mat y_label_new(y_label.n, y_label.m);
    y_label_new = y_label;
    dataProcess(y_label, y_label_new);
    for(int t=1;t<=1;)
    {
        cout<<"sorT = "<<t<<endl;
        for(int s=1;s<=1;s++)
        {
            for (int i = 1; i <=1; i++)
            {
                LARGE_INTEGER head, tail, freq;
                QueryPerformanceFrequency(&freq);
                QueryPerformanceCounter(&head);
                for (int j = 0; j < 10; j++)  labelPropagation(X, y_label_new, y_pred, y_res,i,0.5,1000,t,s);
                QueryPerformanceCounter(&tail);
                outTime<<(1<<(i%10-1))<<","<<i/10+1<<","<<(tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart<<endl;
                cout <<i<< ": Time Elapsed: " << (tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart << "ms" << endl;
                if (i % 10 == 0)  cout << endl;
            }
        }
        
        t=t?t<<1:1;
    }
    // getchar();
}
// PYBIND11_MODULE(label_propagation, m) {
//     py::class_<matCoo>(m, "matCoo")
//         .def(py::init<int, int>())
//         .def("createMat", &matCoo::createMat)
//         .def("matTimes", &matCoo::matTimes)
//         .def("append", &matCoo::append);

//     py::class_<mat>(m, "mat")
//         .def(py::init<int, int>())
//         .def("createMat", &mat::createMat)
//         .def("matTimes", &mat::matTimes)
//         .def("findDiff", &mat::findDiff)
//         .def("editval",&mat::editval)
//         .def("getval",&mat::getval)
//         .def("setneg",&mat::setneg)
//         .def("editval2",&mat::editval2);
//     m.def("labelPropagation", &labelPropagation);
//     m.def("dataProcess",&dataProcess);
// }
/*
Time Elapsed: 170174ms
Time Elapsed: 169375ms
Time Elapsed: 152085ms
*/