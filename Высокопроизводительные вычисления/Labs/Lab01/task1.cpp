#include <iostream>
#include <random>
#include <omp.h>

#define N 500 //500 1000 1500

using namespace std;

void randMtrx(double* mtrx, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      mtrx[n * j + i] = rand() % 10;
}

double* createMtrx(int n)
{
  double* mtrx = new double[n * n];
  randMtrx(mtrx, n);
  return mtrx;
}

void deleteMtrx(double*& mtrx)
{
  delete[] mtrx;
  mtrx = NULL;
}

void printMtrx(double* mtrx, int n)
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

void seq_dgemm(int n, double* a, double* b, double* c)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
      c[n * j + i] = 0;
      for (int k = 0; k < n; k++)
        c[n * j + i] += (a[n * k + i] * b[n * j + k]);
    }
}

void blas_dgemm(int n, double* a, double* b, double* c)
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

bool compareMtrx(int n, double* a, double* b)
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

  double* a = NULL, * b = NULL, * c = NULL, * d = NULL;
  int n = N;

  a = createMtrx(n);
  b = createMtrx(n);
  c = createMtrx(n);
  d = createMtrx(n);

  //printMtrx(a, n);
  //printMtrx(b, n);

  seq_dgemm(n, a, b, c);
  blas_dgemm(n, a, b, d);

  //printMtrx(c, n);
  //printMtrx(d, n);

  if (compareMtrx(n, c, d))
    cout << "blas_dgemm() == seq_dgemm()" << endl;
  else
    cout << "blas_dgemm() != seq_dgemm()" << endl;

  deleteMtrx(a);
  deleteMtrx(b);
  deleteMtrx(c);
  deleteMtrx(d);

  return 0;
}