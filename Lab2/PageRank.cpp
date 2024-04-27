//g++ -fopenmp -mavx2
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <random>
#include <immintrin.h>
#include <unordered_map>
#include <xmmintrin.h>
#include <fstream>
#include <windows.h>
#include <omp.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

using namespace std;

class matCoo {
public:
    struct elems {
        double v;
        int row, col;
        bool operator<(const elems& other) {
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
    void output();
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
    void output();
};

void matMultiply(matCoo& x1, mat& x2, mat& res);
void matMultiplyBlock(matCoo& x1, mat& x2, mat& res, int blockSize);
void matAdd(matCoo& x1, matCoo& x2, matCoo& res);
void matMultiplyBlock_SSE(const matCoo& x1, const mat& x2, mat& res, int blockSize);
void matMultiplyBlock_AVX(const matCoo& x1, const mat& x2, mat& res, int blockSize);
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

void mat::output()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cout << v[i][j] << " ";
        }
        cout << endl;
    }
}
bool isZero(double a)
{
    return abs(a) < 0.0000000001 ? true : false;
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
void matCoo::output()
{
    std::sort(elem.begin(), elem.end());
    int t = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (t < totalElements && elem[t].row == i && elem[t].col == j)
            {
                cout << elem[t].v << " ";
                t++;
            }
            else  cout << 0 << " ";
        }
        cout << endl;
    }
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
    std::sort(x1.elem.begin(), x1.elem.end());

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



void matMultiplyOmp(matCoo& x1, mat& x2, mat& res,int ThreadNum) {
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

void matMultiplyOmpAvx(matCoo& x1, mat& x2, mat& res, int ThreadNum) {
    omp_set_num_threads(ThreadNum);

    int n = x1.n, m = x2.m;
    res.createMat(n, m);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), i) - x1.elem.begin();
        for (int j = 0; j+4 < m; j += 4) {
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

void matAdd(matCoo& x1, matCoo& x2, matCoo& res)
{
    res.createMat(x1.n, x1.m);
    std::sort(x1.elem.begin(), x1.elem.end());
    std::sort(x2.elem.begin(), x2.elem.end());
    int now1 = 0, now2 = 0;
    for (int i = 0; i < x1.n; i++)
    {
        while (x1.elem[now1].row == i || x2.elem[now2].row == i)
        {
            // cout<<now1<<" "<<now2<<endl;
            if ((x1.elem[now1].col < x2.elem[now2].col || now2 >= x2.totalElements) && now1 < x1.totalElements)
            {
                while ((x1.elem[now1].col < x2.elem[now2].col || now2 >= x2.totalElements) && x1.elem[now1].row == i)
                {
                    res.append(x1.elem[now1].row, x1.elem[now1].col, x1.elem[now1].v);
                    now1++;
                }
            }
            else if ((x2.elem[now2].col < x1.elem[now1].col || now1 >= x1.totalElements) && now2 < x2.totalElements)
            {
                while ((x2.elem[now2].col < x1.elem[now1].col || now1 >= x1.totalElements) && x2.elem[now2].row == i)
                {
                    res.append(x2.elem[now2].row, x2.elem[now2].col, x2.elem[now2].v);
                    now2++;
                }
            }
            else if (now1 < x1.totalElements && now2 < x2.totalElements && x1.elem[now1].col == x2.elem[now2].col)
            {
                if (!isZero(x1.elem[now1].v + x2.elem[now2].v))
                    res.append(x1.elem[now1].row, x1.elem[now1].col, x1.elem[now1].v + x2.elem[now2].v);
                now1++, now2++;
            }
            if (now1 > x1.totalElements && now2 > x2.totalElements)  return;
        }
    }
}


void matMultiplyBlock_SSE(matCoo& x1, mat& x2, mat& res, int blockSize = 64) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);
    std::sort(x1.elem.begin(), x1.elem.end());

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < m; jj += blockSize) {
            for (int i = ii; i < min(ii + blockSize, n); i++) {
                long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), matCoo::elems{ 0, i, -1 }) - x1.elem.begin();
                for (int j = jj; j < min(jj + blockSize, m); j++) {
                    __m128d sum = _mm_setzero_pd();
                    int q = p;
                    while (q < x1.elem.size() && x1.elem[q].row == i) {
                        __m128d a = _mm_set1_pd(x1.elem[q].v);
                        __m128d b = _mm_load_sd(&x2.v[x1.elem[q].col][j]);
                        sum = _mm_add_sd(sum, _mm_mul_sd(a, b));
                        q++;
                    }
                    double* f = (double*)&sum;
                    res.v[i][j] = f[0];
                }
            }
        }
    }
}

void matMultiplyBlock_AVX(matCoo& x1, mat& x2, mat& res, int blockSize = 64) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);
    std::sort(x1.elem.begin(), x1.elem.end());

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < m; jj += blockSize) {
            for (int i = ii; i < min(ii + blockSize, n); i++) {
                long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), matCoo::elems{ 0, i, -1 }) - x1.elem.begin();
                for (int j = jj; j < min(jj + blockSize, m); j++) {
                    __m256d sum = _mm256_setzero_pd();
                    int q = p;
                    while (q < x1.elem.size() && x1.elem[q].row == i) {
                        __m256d a = _mm256_set1_pd(x1.elem[q].v);
                        __m256d b = _mm256_broadcast_sd(&x2.v[x1.elem[q].col][j]);
                        sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
                        q++;
                    }
                    double* f = (double*)&sum;
                    res.v[i][j] = f[0];
                }
            }
        }
    }
}

void matMultiplyBlock_AVX(matCoo& x1, mat& x2, mat& res, int blockSize = 64) {
    int n = x1.n, m = x2.m;
    res.createMat(n, m);
    std::sort(x1.elem.begin(), x1.elem.end());

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < m; jj += blockSize) {
            for (int i = ii; i < min(ii + blockSize, n); i++) {
                long long p = std::lower_bound(x1.elem.begin(), x1.elem.end(), matCoo::elems{ 0, i, -1 }) - x1.elem.begin();
                for (int j = jj; j < min(jj + blockSize, m); j += 4) {
                    if (j + 4 <= m) {
                        __m256d sum = _mm256_setzero_pd();
                        int q = p;
                        while (q < x1.elem.size() && x1.elem[q].row == i) {
                            __m256d a = _mm256_set1_pd(x1.elem[q].v);
                            __m256d b = _mm256_load_pd(&x2.v[x1.elem[q].col][j]);
                            sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
                            q++;
                        }
                        double* f = (double*)&sum;
                        for (int k = 0; k < 4; k++) {
                            res.v[i][j + k] = f[k];
                        }
                    }
                    else {
                        for (int j2 = j; j2 < m; j2++) {
                            double sum = 0.0;
                            int q = p;
                            while (q < x1.elem.size() && x1.elem[q].row == i) {
                                double a = x1.elem[q].v;
                                double b = x2.v[x1.elem[q].col][j2];
                                sum += a * b;
                                q++;
                            }
                            res.v[i][j2] = sum;
                        }
                        break;
                    }
                }
            }
        }
    }
}


class pageRes {
public:
    int id;
    double val;
    bool operator<(const pageRes& other) {
        return this->val < other.val || this->val == other.val && this->id < other.id;
    }
    bool operator<(double i) const {
        return val < i;
    }
};
matCoo zeroOne;
vector<int> out_degree;

void readMat(matCoo& res, int& maxx, int& minn)
{
    ifstream in1("WikiData.txt");
    int x, y;
    maxx = 0;
    minn = 1919810;
    out_degree.clear();
    out_degree.resize(200000, 0);
    while (in1.eof() != true)
    {
        in1 >> x >> y;
        res.append(x - 1, y - 1, 1.0);
        out_degree[x - 1]++;
        maxx = max(x, maxx);
        maxx = max(y, maxx);
        minn = min(x, minn);
        minn = min(y, minn);
    }
}
void calGM(matCoo& zm, matCoo& gm, double alpha = 0.85)
{
    gm.createMat(zm.n, zm.m);
    std::sort(zm.elem.begin(), zm.elem.end());
    int nowi = 0, p1, p2 = -1, p3;
    double tot = 0;
    for (int i = 0; i < zm.n; i++)
    {
        p1 = std::lower_bound(zm.elem.begin(), zm.elem.end(), i) - zm.elem.begin();
        p2 = std::lower_bound(zm.elem.begin(), zm.elem.end(), i + 1) - zm.elem.begin();
        p3 = p2 - p1;
        tot = alpha / p3;
        for (int j = p1; j < p2; j++)
        {
            gm.append(zm.elem[j].col, zm.elem[j].row, tot);
        }
    }
}
void PageRank(matCoo& x1, mat& res, int mode = 0, double alpha = 0.85, int maxiter = 1000)
{
    matCoo gm;
    calGM(x1, gm, alpha);
    res.createMat(x1.n, 1);
    mat tmp1(x1.n, 1);
    mat tmp2(x1.n, 1);
    for (int i = 0; i < x1.n; i++)
    {
        tmp1.v[i][0] = 1.0 / x1.n;
    }
    std::sort(gm.elem.begin(), gm.elem.end());
    for (int iter = 0; iter < maxiter; iter++)
    {
        if (iter)  tmp1 = tmp2;
        //std::cout << iter << std::endl;
        //LARGE_INTEGER head, tail, freq;
        //QueryPerformanceFrequency(&freq);
        //QueryPerformanceCounter(&head);
        switch (mode)
        {
        case 1:matMultiplyBlock(gm, tmp1, tmp2, 64); break;
        case 2:matMultiplyBlock_SSE(gm, tmp1, tmp2, 64); break;
        case 3:matMultiplyBlock_AVX(gm, tmp1, tmp2, 64); break;
        // case 4:matMultiplyOmp(gm, tmp1, tmp2, 8); break;
        // case 5:matMultiplyOmp(gm, tmp1, tmp2, 16); break;

        // case 7:matMultiplyOmpAvx(gm, tmp1, tmp2, 1); break;
        // case 8:matMultiplyOmpAvx(gm, tmp1, tmp2, 2); break;
        // case 9:matMultiplyOmpAvx(gm, tmp1, tmp2, 4); break;
        // case 10:matMultiplyOmpAvx(gm, tmp1, tmp2, 8); break;
        // case 11:matMultiplyOmpAvx(gm, tmp1, tmp2, 16); break;

        // /*case 12:matMultiplyOmpAvxVec(gm, tmp1, tmp2, 1); break;
        // case 13:matMultiplyOmpAvxVec(gm, tmp1, tmp2, 2); break;
        // case 14:matMultiplyOmpAvxVec(gm, tmp1, tmp2, 4); break;
        // case 15:matMultiplyOmpAvxVec(gm, tmp1, tmp2, 8); break;
        // case 16:matMultiplyOmpAvxVec(gm, tmp1, tmp2, 16); break;*/
        default:matMultiply(gm, tmp1, tmp2); break;
        }
        //QueryPerformanceCounter(&tail);
        //cout << "Time Elapsed: " << (tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart << "ms" << endl;

        double flag = 0.0;
        double sum = 0.0;
        double sum_no_out_degree = 0.0;
        for (int i = 0; i < x1.n; i++)
        {
            if (out_degree[i] == 0)
            {
                sum_no_out_degree += tmp1.v[i][0];
            }
        }
        for (int i = 0; i < x1.n; i++)
        {
            tmp2.v[i][0] += (1.0 - alpha) / double(x1.n) + alpha * sum_no_out_degree / x1.n;
        }
        for (int i = 0; i < x1.n; i++)
        {
            sum += tmp2.v[i][0];
            flag += abs(tmp1.v[i][0] - tmp2.v[i][0]);
        }
        //std::cout << sum << std::endl;
        //tmp2.output();
        if (flag < 1e-5)
        {
            //std::cout << iter << std::endl;
            res = tmp2;
            return;
        }
    }
    //std::cout << maxiter << std::endl;
    res = tmp2;
}
void resOutput(mat& res)
{
    ofstream out1("Res.txt");
    pageRes tmp;
    std::vector<pageRes>pageres;
    for (int i = 0; i < res.n; i++)
    {
        tmp.id = i;
        tmp.val = res.v[i][0];
        pageres.push_back(tmp);
    }
    std::sort(pageres.begin(), pageres.end());
    double ans = 0;
    for (int i = 0; i < res.n; i++)  ans += res.v[i][0];
    //cout << ans << endl;
    for (int i = res.n - 1; i >= res.n - 100 && i >= 0; i--)
    {
        out1 << pageres[i].id + 1 << " " << pageres[i].val / ans << std::endl;
    }
    return;
}
// int main()
// {
//     int n=readMat();
//     zeroOne.m=zeroOne.n=n;
//     matCoo gM(n,n),baseMat(n,n);
//     matCoo *gm=&gM;
//     matCoo *zm=&zeroOne;
//     matCoo *bse=&baseMat;
//     calGM(zm,gm);
//     gm->output();
//     genBase(bse);
//     bse->output();
// }
string Mode[20] = { "Non-Block","Block","SSE","AVX" };
int main()
{
    for (int p = 0; p <= 3; p++)
    {
        cout << endl << endl << Mode[p] << " Mode: " << endl;
        LARGE_INTEGER head, tail, freq;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&head);
        for (int t = 0; t <= 10; t++)
        {
            int n0, m0;
            matCoo m1(200000, 200000);
            readMat(m1, n0, m0);
            //std::cout << n0 << " " << m0 << endl;
            matCoo m2(n0, n0);
            for (int i = 0; i < m1.totalElements; i++)
            {
                m2.append(m1.elem[i].row, m1.elem[i].col, 1.0);
            }
            //m1.append(0,1,1);
            //m1.append(0,2,1);
            //m1.append(0,3,1);
            //m1.append(1,2,1);
            //m1.append(2,3,1);
            //m1.append(3,0,1);
            //m1.append(3,1,1);
            //matCoo m2(4, 4);
            //calGM(m1,m2,1);
            //m2.output();
            mat res;
            
            PageRank(m2, res, p);
            
            //resOutput(res);
            //res.output();
        }
        QueryPerformanceCounter(&tail);
        cout << "Time Elapsed: " << (tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart << "ms" << endl;

    }
    getchar();
    return 0;
}