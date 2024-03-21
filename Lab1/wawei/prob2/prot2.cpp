#include <iostream>
#include <sys/time.h>
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
    ull res=pow2(n);
    ull ans=0;
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int i=1;i<=10;i++)  ans=mainAdd(res);
    gettimeofday(&end,NULL);
    cout<<n<<","<<ans<<","<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/10000<<endl;
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