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
  float* mtrx = (float*)malloc(n * n * sizeof(float));
  randMtrx(mtrx, n);
  return mtrx;
}

float* createMtrxPinned(int n)
{
  float* mtrx;
  // Pinned Memory
  cudaMallocHost((void**)&mtrx, n * n * sizeof(float));
  randMtrx(mtrx, n);
  return mtrx;
}

float* createMtrxOnDevice(int n)
{
  float* mtrx_dev = NULL;
  cudaMalloc((void**)&mtrx_dev, n * n * sizeof(float));
  return mtrx_dev;
}

float* copyMtrxToDevice(int n, float*& mtrx)
{
  float* mtrx_dev = NULL;
  cudaMalloc((void**)&mtrx_dev, n * n * sizeof(float));
  cudaMemcpy(mtrx_dev, mtrx, n * n * sizeof(float), cudaMemcpyHostToDevice);
  return mtrx_dev;
}

float* copyMtrxFromDevice(int n, float*& mtrx_dev)
{
  float* mtrx = (float*)malloc(n * n * sizeof(float));
  cudaMemcpy(mtrx, mtrx_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);
  return mtrx;
}

void deleteMtrx(float*& mtrx)
{
  free(mtrx);
  mtrx = NULL;
}

void deleteMtrxPinned(float*& mtrx)
{
  // Pinned Memory
  if (cudaFreeHost(mtrx) != cudaSuccess)
    printf("Error in deleteMtrxPinned()\n");
  mtrx = NULL;
}

void deleteMtrxFromDevice(float*& mtrx_dev)
{
  if (cudaFree(mtrx_dev) != cudaSuccess)
    printf("Error in deleteMtrxFromDevice()\n");
  mtrx_dev = NULL;
}

void printMtrx(float* mtrx, int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
      printf("%f ", mtrx[n * j + i]);
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
  if (idx >= n || idy >= n)
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
      return false;
  return true;
}

int div_up(int x, int y)
{
  return (x - 1) / y + 1;
}

int main()
{
  srand(time(NULL));
  printf("N = %d\n", N);
  dim3 threads(THREADS_N, THREADS_N);
  printf("threads.x = %d threads.y = %d\n", threads.x, threads.y);
  dim3 blocks(div_up(N, threads.x), div_up(N, threads.y));
  printf("blocks.x = %d blocks.y = %d\n\n", blocks.x, blocks.y);

  float* a = createMtrxPinned(N);
  float* b = createMtrxPinned(N);
  float* c = createMtrxPinned(N);
  float* adev = copyMtrxToDevice(N, a);
  float* bdev = copyMtrxToDevice(N, b);
  float* cdev = copyMtrxToDevice(N, c);

  //printMtrx(a, N);
  //printMtrx(b, N);
  seq_dgemm(N, a, b, c);
  //printMtrx(c, N);

  cuda_dgemm << < blocks, threads >> > (N, adev, bdev, cdev);
  float* d = copyMtrxFromDevice(N, cdev);
  //printMtrx(d, N);

  if (compareMtrx(N, c, d))
    printf("Pinned Memory cuda_dgemm() == seq_dgemm()\n\n");
  else
    printf("Pinned Memory cuda_dgemm() != seq_dgemm()\n\n");

  deleteMtrxPinned(a);
  deleteMtrxPinned(b);
  deleteMtrxPinned(c);
  deleteMtrx(d);
  deleteMtrxFromDevice(adev);
  deleteMtrxFromDevice(bdev);
  deleteMtrxFromDevice(cdev);

  //Time comparation
  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);

  std::cout << "Pinned Memory copyMtrxToDevice() time:" << std::endl;
  std::string timesStr = "";
  int n = 5000;
  while (n > 0)
  {
    float elapsedTime = 0;
    float* a2 = createMtrxPinned(n);
    for (int it = 0; it < ITERS; it++)
    {
      cudaEventRecord(e_start, 0);

      float* adev2 = copyMtrxToDevice(n, a2);

      cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
      cudaEventSynchronize(e_stop);

      deleteMtrxFromDevice(adev2);

      float tmpTime = 0;
      cudaEventElapsedTime(&tmpTime, e_start, e_stop);
      elapsedTime += tmpTime;
    }
    deleteMtrxPinned(a2);
    elapsedTime /= ITERS;

    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    double time = elapsedTime / 1000.;
    std::cout << "N = " << n << " matrices copy time : " << time << " s" << std::endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;

    n -= 1000;
  }
  std::cout << timesStr << std::endl;
  std::cout << "\n\n";

  std::cout << "Usual Memory copyMtrxToDevice() time:" << std::endl;
  timesStr = "";
  n = 5000;
  while (n > 0)
  {
    float elapsedTime = 0;
    float* a2 = createMtrx(n);
    for (int it = 0; it < ITERS; it++)
    {
      cudaEventRecord(e_start, 0);

      float* adev2 = copyMtrxToDevice(n, a2);

      cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
      cudaEventSynchronize(e_stop);

      deleteMtrxFromDevice(adev2);

      float tmpTime = 0;
      cudaEventElapsedTime(&tmpTime, e_start, e_stop);
      elapsedTime += tmpTime;
    }
    deleteMtrx(a2);
    elapsedTime /= ITERS;

    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    double time = elapsedTime / 1000.;
    std::cout << "N = " << n << " matrices copy time : " << time << " s" << std::endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;

    n -= 1000;
  }
  std::cout << timesStr << std::endl;
  std::cout << "\n\n";

  return 0;
}