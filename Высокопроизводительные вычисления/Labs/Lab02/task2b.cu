#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <string>

#define N (500)
#define THREADS_N (32)
#define ITERS (100)

void randMtrx(float* mtrx, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      mtrx[n * j + i] = rand() % 10;
}

float* createMtrx(int n)
{
  float* mtrx = (float*)malloc(N * N * sizeof(float));
  randMtrx(mtrx, n);
  return mtrx;
}

float* createMtrxUnified(int n)
{
  float* mtrx;
  // Unified Memory
  cudaMallocManaged((void**)&mtrx, N * N * sizeof(float));
  randMtrx(mtrx, n);
  return mtrx;
}

float* createMtrxOnDevice(int n)
{
  float* mtrx_dev = NULL;
  cudaMalloc((void**)&mtrx_dev, n * n * sizeof(float));
  return mtrx_dev;
}

/*float* copyMtrxToDevice(int n, float*& mtrx)
{
  float* mtrx_dev = NULL;
  cudaMalloc((void**)&mtrx_dev, n * n * sizeof(float));
  cudaMemcpy(mtrx_dev, mtrx, n * n * sizeof(float), cudaMemcpyHostToDevice);
  return mtrx_dev;
}*/

float* copyMtrxFromDevice(int n, float*& mtrx_dev)
{
  float* mtrx = (float*)malloc(N * N * sizeof(float));
  cudaMemcpy(mtrx, mtrx_dev, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  return mtrx;
}

void deleteMtrx(float*& mtrx)
{
  free(mtrx);
  mtrx = NULL;
}

void deleteMtrxFromDevice(float*& mtrx_dev)
{
  if (cudaFree(mtrx_dev) != cudaSuccess)
    printf("Error in deleteMtrxFromDevice()");
  mtrx_dev = NULL;
}

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

__global__ void cuda_dgemm(int n, float* a, float* b, float* c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  //printf("%d, %d\n", idx, idy);
  if (idx >= N || idy >= N)
    return;
  int i = idx;
  int j = idy;

  float res = 0;
  for (int k = 0; k < n; k++)
    res += (a[n * k + i] * b[n * j + k]);
  c[n * j + i] = res;
}

bool compareMtrx(int n, float* a, float* b)
{
  for (int i = 0; i < n * n; i++)
    if (a[i] != b[i])
    {
      int i2 = i / N;
      int j = i % N;
      printf("compareMtrx() Error: i = %d, j = %d, %.0f != %.0f\n", i2, j, a[i], b[i]);
      return false;
    }
  return true;
}

__global__ void cuda_hello()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  printf("Hello World from GPU! %d %d\n", idx, idy);
}

int main()
{
  srand(time(NULL));
  printf("N = %d\n", N);
  dim3 threads(THREADS_N, THREADS_N);
  printf("threads.x = %d threads.y = %d\n", threads.x, threads.y);
  dim3 blocks((N - 1) / THREADS_N + 1, (N - 1) / THREADS_N + 1);
  printf("blocks.x = %d blocks.y = %d\n\n", blocks.x, blocks.y);

  float* a_unif = createMtrxUnified(N);
  float* b_unif = createMtrxUnified(N);
  float* c_unif = createMtrxUnified(N);
  float* c = copyMtrxFromDevice(N, c_unif);

  //printMtrx(a, N);
  //printMtrx(b, N);
  seq_dgemm(N, a_unif, b_unif, c);
  //printMtrx(c, N);

  cuda_dgemm << < blocks, threads >> > (N, a_unif, b_unif, c_unif);
  float* d = copyMtrxFromDevice(N, c_unif);
  //printMtrx(cdev, N);

  if (compareMtrx(N, c, d))
    printf("Unified Memory cuda_dgemm() == seq_dgemm()\n\n");
  else
    printf("Unified Memory cuda_dgemm() != seq_dgemm()\n\n");

  deleteMtrx(c);
  deleteMtrx(d);
  deleteMtrxFromDevice(a_unif);
  deleteMtrxFromDevice(b_unif);
  deleteMtrxFromDevice(c_unif);

  return 0;
}