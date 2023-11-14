#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <time.h>

#define N (1500)
#define ITERS (10)

void printMtrx(float* mtrx, int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
      printf("%3.0f ", mtrx[n * j + i]);
    printf("\n");
  }
  printf("\n");
}

long double getGflops(long long n, double time)
{
  long double fl_opers = 1e-9;
  fl_opers *= n * n * n * 2;
  return fl_opers / time;
}

void seq_dgemm(int n, float* a, float* b, float* c)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
      c[n * j + i] = 0;
      for (int k = 0; k < n; k++)
        c[n * j + i] += (a[n * k + i] * b[n * j + k]);
    }
}

void omp_dgemm(int n, float* A, float* B, float* C)
{
#pragma omp target teams distribute parallel for collapse(2) map(to: A[0:n*n], B[0:n*n]) map(from: C[0:n*n])
//  {
//#pragma omp parallel for collapse(2)
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < n; i++)
      {
        float sum = 0.0;
        for (int k = 0; k < n; k++)
          sum += A[k * n + i] * B[j * n + k];
        C[j * n + i] = sum;
      }
    }
//  }
}

int main()
{
  srand(time(NULL));
  printf("N = %d\n", N);
  int n = N;

  float* a = (float*)malloc(N * N * sizeof(float));
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      a[n * j + i] = rand() % 10;
  float* b = (float*)malloc(N * N * sizeof(float));
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      b[n * j + i] = rand() % 10;
  float* c = (float*)malloc(N * N * sizeof(float));

  //printMtrx(a, N);
  //printMtrx(b, N);
  seq_dgemm(N, a, b, c);
  //printMtrx(c, N);

  float* d = (float*)malloc(N * N * sizeof(float));
  omp_dgemm(N, a, b, d);

  //printMtrx(d, N);

  bool equals = true;
  for (int i = 0; i < n * n; i++)
    if (c[i] != d[i])
    {
      equals = false;
      break;
    }

  if (equals)
    printf("seq_dgemm() == omp_dgemm()\n");
  else
    printf("seq_dgemm() != omp_dgemm()\n");

  free(a);
  free(b);
  free(c);
  free(d);


  //Time comparation
  printf("omp omp_dgemm():\n");
  //std::string timesStr = "";
  //std::string gflopsStr = "";
  n = N;
  while (n > 0)
  {
    float* a2 = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        a2[n * j + i] = rand() % 10;
    float* b2 = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        b2[n * j + i] = rand() % 10;
    float* c2 = (float*)malloc(N * N * sizeof(float));

    double startTime = omp_get_wtime();

    for (int it = 0; it < ITERS; it++)
      omp_dgemm(n, a2, b2, c2);

    double endTime = omp_get_wtime();
    double elapsedTime = endTime - startTime;

    double time = elapsedTime / ITERS;
    double gflops = getGflops(n, time);
    printf("N = %d matrices time : %f s; GFLOPS = %f\n", n, time, gflops);
    //std::cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << std::endl;
    //timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    //gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    //printMtrx(c, n);

    free(a2);
    free(b2);
    free(c2);

    n -= 500;
  }
  //std::cout << timesStr << std::endl;
  //std::cout << gflopsStr << std::endl;
  //std::cout << "\n\n";
  printf("\n\n");

  /*
  printf("omp seq_dgemm():\n");
  n = N;
  while (n > 0)
  {
    float* a2 = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        a2[n * j + i] = rand() % 10;
    float* b2 = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        b2[n * j + i] = rand() % 10;
    float* c2 = (float*)malloc(N * N * sizeof(float));

    double startTime = omp_get_wtime();

    seq_dgemm(n, a2, b2, c2);

    double endTime = omp_get_wtime();
    double elapsedTime = endTime - startTime;

    double time = elapsedTime;
    double gflops = getGflops(n, time);
    printf("N = %d matrices time : %f s; GFLOPS = %f\n", n, time, gflops);

    free(a2);
    free(b2);
    free(c2);

    n -= 500;
  }
  printf("\n\n");
  */

  return 0;
}