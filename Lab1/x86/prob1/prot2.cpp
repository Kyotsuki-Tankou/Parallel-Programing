#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;

#define ull unsigned long long int

const ull N = 11451;
ull a[N];
ull b[N][N];
ull sum[N];
ull tmp1,tmp2,tmp3,tmp4,tmp5;
int nlist[30]={10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4500,5000,7500,10000};

void init(ull n)
{
    for(ull i=0;i<n;i++)  a[i]=i;
    for(ull i=0;i<n;i++)
      for(ull j=0;j<n;j++)
        b[i][j]=i+j;
    for(ull i=0;i<n;i++)  sum[i]=0;
    return;
}

ull mainMul(int n)
{   
    ull res=0;
    init(n);
    for(int j=0;j<n;j+=5)
    {
        tmp1=0,tmp2=0,tmp3=0,tmp4=0,tmp5=0;
        for(int i=0;i<n;i++)
        {
            tmp1+=a[j+0]*b[j+0][i];
            tmp2+=a[j+1]*b[j+1][i];
            tmp3+=a[j+2]*b[j+2][i];
            tmp4+=a[j+3]*b[j+3][i];
            tmp5+=a[j+4]*b[j+4][i];
        }
        sum[j+0]=tmp1;
        sum[j+1]=tmp2;
        sum[j+2]=tmp3;
        sum[j+3]=tmp4;
        sum[j+4]=tmp5;
    }
    for(int i=0;i<n;i++)  res+=sum[i];
    return res;
}
void timeAssess(int n)
{
    LARGE_INTEGER head,tail,freq;
    ull ans=0;
    struct timeval start;
    struct timeval end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&head);
    for(int i=1;i<=10;i++)  ans=mainMul(n);
    QueryPerformanceCounter(&tail);
    cout<<n<<","<<ans<<","<<(tail.QuadPart-head.QuadPart)*100.0/freq.QuadPart<<endl;
    return;
}

int main()
{
    for(int i=0;i<=28;i++)
    {
        timeAssess(nlist[i]);
    }
    return 0;
}