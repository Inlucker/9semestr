#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>
#include <string>
extern "C"
{
  #include <cblas.h>
}

#define N 1500 //500 1000 1500

using namespace std;

typedef std::chrono::high_resolution_clock Clock;
typedef float t_myfloat;

void randMtrx(t_myfloat* mtrx, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      mtrx[n * j + i] = rand() % 10;
}

t_myfloat* createMtrx(int n)
{
  t_myfloat* mtrx = new t_myfloat[n * n];
  randMtrx(mtrx, n);
  return mtrx;
}

void deleteMtrx(t_myfloat*& mtrx)
{
  delete[] mtrx;
  mtrx = NULL;
}

void printMtrx(t_myfloat* mtrx, int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
      cout << mtrx[n * j + i] << " ";
    cout << endl;
  }
  /*for (int i = 0; i < n * n; i++)
    cout << mtrx[i] << " ";
  cout << endl;*/
  cout << endl;
}

long double getGflops(long long n, double time)
{
  long double fl_opers = 1e-9;
  fl_opers *= n * n * n * 2;
  return fl_opers / time;
}

void blas_dgemm(int n, t_myfloat* a, t_myfloat* b, t_myfloat* c)
{
  int i, j, k;
#pragma omp parallel for shared(a, b, c) private(i, j, k)
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      c[n * j + i] = 0;
      for (k = 0; k < n; k++)
        c[n * j + i] += (a[n * k + i] * b[n * j + k]);
    }
}

bool compareMtrx(int n, t_myfloat* a, t_myfloat* b)
{
  for (int i = 0; i < n * n; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

int main(int argc, char** argv)
{
#pragma omp parallel
  {
    if (omp_get_thread_num() == 0)
      cout << "omp_get_num_threads() = " << omp_get_num_threads() << endl;
  }

  srand(time(NULL));

  t_myfloat* a = NULL, * b = NULL, * c = NULL;

  cout << "OpenBlas cblas_sgemm():" << endl;
  string timesStr = "";
  string gflopsStr = "";
  int n = N;
  while (n > 0)
  {
    a = createMtrx(n);
    b = createMtrx(n);
    c = createMtrx(n);

    //printMtrx(a, n);
    //printMtrx(b, n);

    randMtrx(a, n);
    randMtrx(b, n);

    chrono::time_point<Clock> start = Clock::now();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
    chrono::time_point<Clock> end = Clock::now();
    chrono::nanoseconds diff = chrono::duration_cast<chrono::nanoseconds>(end - start);
    double time = diff.count() / 1000000000.;
    double gflops = getGflops(n, time);
    cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    //printMtrx(c, n);

    deleteMtrx(a);
    deleteMtrx(b);
    deleteMtrx(c);

    n -= 500;
  }
  cout << timesStr << endl;
  cout << gflopsStr << endl;
  cout << "\n\n";

  cout << "blas_dgemm():" << endl;
  timesStr = "";
  gflopsStr = "";
  n = N;
  while (n > 0)
  {
    a = createMtrx(n);
    b = createMtrx(n);
    c = createMtrx(n);

    //printMtrx(a, n);
    //printMtrx(b, n);

    randMtrx(a, n);
    randMtrx(b, n);

    chrono::time_point<Clock> start = Clock::now();
    blas_dgemm(n, a, b, c);
    chrono::time_point<Clock> end = Clock::now();
    chrono::nanoseconds diff = chrono::duration_cast<chrono::nanoseconds>(end - start);
    double time = diff.count() / 1000000000.;
    double gflops = getGflops(n, time);
    cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    //printMtrx(c, n);

    deleteMtrx(a);
    deleteMtrx(b);
    deleteMtrx(c);

    n -= 500;
  }
  cout << timesStr << endl;
  cout << gflopsStr << endl;
  cout << "\n\n";

  return 0;
}