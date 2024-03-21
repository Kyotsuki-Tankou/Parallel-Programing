#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;

#define ull unsigned long long int

const ull N = 214748364;
ull a[N];
ull pow2(int n)
{
    ull res=1;
    for(int i=1;i<=n;i++)  res<<=1;
    return res;
}
void init(ull n)
{
    for(ull i=0;i<n;i++)  a[i]=i;
    return;
}

ull mainAdd(int n)
{   
    init(n);
    ull sum1=0,sum2=0;
    for(ull i=0;i<n;i+=2)
    {
        sum1+=a[i];
        sum2+=a[i+1];
    }
    return sum1+sum2;
}
void timeAssess(int n)
{
    LARGE_INTEGER head,tail,freq;
    ull res=pow2(n);
    ull ans=0;
    struct timeval start;
    struct timeval end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&head);
    for(int i=1;i<=10;i++)  ans=mainAdd(res);
    QueryPerformanceCounter(&tail);
    cout<<n<<","<<ans<<","<<(tail.QuadPart-head.QuadPart)*100.0/freq.QuadPart<<endl;
    return;
}

int main()
{
    for(int i=10;i<=27;i++)
    {
        timeAssess(i);
    }
    return 0;
}