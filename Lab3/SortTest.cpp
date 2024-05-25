// g++ -fopenmp -mavx2 -lpthread -o SortTest.exe SortTest.cpp
#include <iostream>
#include <vector>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <thread>
#include <queue>
#include <functional>
#include <ctime>
#include <fstream>
#include <windows.h>
#include <algorithm>
#include <random>
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
void matCoo::createMat(int n0, int m0)
{
    n = n0;
    m = m0;
    totalElements = 0;
    elem.clear();
    return;
}

// int main() {
//     srand(time(NULL));
//     std::cout<<"1\n";
//     // 初始化矩阵
//     int n_samples = 10;
//     matCoo W(n_samples, n_samples);
//     for (int i = n_samples; i >=0 ; i--) {
//         for (int j = n_samples-1; j>=0  ; j--) {
//             double v = rand() / (double)RAND_MAX;
//             W.elem.push_back({v, i, j});
//         }
//     }
//     std::cout<<"Initialize end\n";
//     std::sort(W.elem.begin(),W.elem.end());
//     std::cout<<"Sort end\n";
//     for (int i = 0; i < W.elem.size(); i++) {
//         std::cout << W.elem[i].v << " (" << W.elem[i].row << ", " << W.elem[i].col << ")" << std::endl;
//     }

//     return 0;
// }

struct ThreadArg {
    std::vector<matCoo::elems>::iterator start;
    std::vector<matCoo::elems>::iterator end;
};

void* thread_sort(void* arg) {
    ThreadArg* threadArg = (ThreadArg*)arg;
    std::sort(threadArg->start, threadArg->end);
    return NULL;
}

void multi_thread_sort(std::vector<matCoo::elems>& vec, int n_threads) {
    std::vector<pthread_t> threads(n_threads);
    std::vector<ThreadArg> args(n_threads);
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
        inplace_merge(vec.begin(), args[i].start, args[i].end);
    }
}

void thread_sort2(ThreadArg arg) {
    std::sort(arg.start, arg.end);
}

void multi_threadpool_sort(std::vector<matCoo::elems>& vec, int n_threads, int n_tasks) {
    std::vector<ThreadArg> args(n_tasks);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    int len = vec.size() / n_tasks;
    int remainder = vec.size() % n_tasks;

    for (int i = 0; i < n_tasks; i++) {
        args[i].start = vec.begin() + i * len + std::min(i, remainder);
        if (i < remainder) {
            args[i].end = args[i].start + len + 1;
        } else {
            args[i].end = args[i].start + len;
        }
    }
    std::queue<ThreadArg> tasks_queue;
    for (auto &arg : args) {
        tasks_queue.push(arg);
    }
    auto worker = [&]() {
    while (!tasks_queue.empty()) {
        ThreadArg arg = tasks_queue.front();
        tasks_queue.pop();
        thread_sort2(arg);
    }
    };
    for (int i = 0; i < n_threads; i++) {
        threads.emplace_back(worker);
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (int i = 1; i < n_tasks; i++) {
        std::inplace_merge(vec.begin(), args[i].start, args[i].end);
    }
}

void openmp_sort1(std::vector<matCoo::elems>&vec,int n_threads)
{
    omp_set_num_threads(n_threads);
    std::vector<matCoo::elems>::iterator start;
    std::vector<matCoo::elems>::iterator end;
    int len=vec.size()/n_threads;
    int reminder=vec.size()%n_threads;
#pragma omp parallel for
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
void openmp_sort2(std::vector<matCoo::elems>&vec,int n_threads)
{
    omp_set_num_threads(n_threads);
    std::vector<matCoo::elems>::iterator start;
    std::vector<matCoo::elems>::iterator end;
    int len=vec.size()/n_threads;
    int reminder=vec.size()%n_threads;
#pragma omp parallel for schedule(static)
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
void openmp_sort3(std::vector<matCoo::elems>&vec,int n_threads)
{
    omp_set_num_threads(n_threads);
    std::vector<matCoo::elems>::iterator start;
    std::vector<matCoo::elems>::iterator end;
    int len=vec.size()/n_threads;
    int reminder=vec.size()%n_threads;
#pragma omp parallel for schedule(guided)
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
int main() {
    srand(time(NULL));
    std::cout<<"1\n";
    // int n_samples = 4000;
    // int x0;
    // matCoo W(n_samples, n_samples);
    // for (int i = n_samples; i >=0 ; i--) {
    //     for (int j = n_samples-1; j>=0  ; j--) {
    //         double v = rand() / (double)RAND_MAX;
    //         x0=rand();
    //         W.elem.push_back({v, x0, j});
    //     }
    // }
    // for (int i = 0; i < W.elem.size(); i++) {
    //     std::cout << W.elem[i].v << " (" << W.elem[i].row << ", " << W.elem[i].col << ")" << std::endl;
    // }
    std::cout<<"Initialize end\n";
    ofstream out1("SortTime.txt");
    int n_threads = 8;
    LARGE_INTEGER head, tail, freq;
    for(int t=15;t<=256;t<<=1)
    {
        int x0;
        matCoo W(2048,2048);
        for (int i = 2048; i >=0 ; i--) {
            for (int j = 2047; j>=0  ; j--) {
                double v = rand() / (double)RAND_MAX;
                x0=rand();
                W.elem.push_back({v, x0, j});
            }
        }
        cout<<"T="<<t<<endl;
        for(int i=1;i<=3;i++)
        {
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&head);
            for (int j = 0; j < 10; j++){
                switch(i)
                {
                    // case 1:openmp_sort1(W.elem,n_threads);break;
                    case 1:openmp_sort2(W.elem,t);break;
                    case 2:openmp_sort3(W.elem,t);break;
                    case 3:openmp_sort(W.elem,t);break;
                    // case 2:openmp_sort(W.elem,t);break;
                    // case 3:multi_thread_sort(W.elem, t);break;
                    // case 4:std::sort(W.elem.begin(),W.elem.end());break;
                }            
            }
            QueryPerformanceCounter(&tail);
            cout <<i<< ": Time Elapsed: " << (tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart << "ms" << endl;
            out1<<t<<","<<i<<","<<(tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart<<endl;
        }
    }
    cout<<"end"<<endl;
    // for (int i = 0; i < W.elem.size(); i++) {
    //     std::cout << W.elem[i].v << " (" << W.elem[i].row << ", " << W.elem[i].col << ")" << std::endl;
    // }
    return 0;
}
