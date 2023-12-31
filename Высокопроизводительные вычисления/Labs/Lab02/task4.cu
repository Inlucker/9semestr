﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <cublas_v2.h>
#include <iostream>
#include <string>

#define N (1500)
#define THREADS_N (32)
#define ITERS 500

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

void deleteMtrxFromDevice(float*& mtrx_dev)
{
  auto ret = cudaFree(mtrx_dev);
  if (ret != cudaSuccess)
    printf("Error in deleteMtrxFromDevice() %d: %s\n", ret, cudaGetErrorString(ret));
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
  dim3 blocks((N - 1) / THREADS_N + 1, (N - 1) / THREADS_N + 1);
  printf("blocks.x = %d blocks.y = %d\n\n", blocks.x, blocks.y);

  float* a = createMtrx(N);
  float* b = createMtrx(N);
  float* c = createMtrx(N);
  float* adev = copyMtrxToDevice(N, a);
  float* bdev = copyMtrxToDevice(N, b);
  float* cdev = createMtrxOnDevice(N);

  //Result comparation

  //printMtrx(a, N);
  //printMtrx(b, N);
  float* cdev2 = createMtrxOnDevice(N);
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, adev, N, bdev, N, &beta, cdev2, N);
  c = copyMtrxFromDevice(N, cdev2);
  //printMtrx(c, N);

  cuda_dgemm << < blocks, threads >> > (N, adev, bdev, cdev);
  float* d = copyMtrxFromDevice(N, cdev);
  //printMtrx(d, N);

  if (compareMtrx(N, d, c))
    printf("cuda_dgemm() == cublasSgemm()\n\n");
  else
    printf("cuda_dgemm() != cublasSgemm()\n\n");

  deleteMtrx(a);
  deleteMtrx(b);
  deleteMtrx(c);
  deleteMtrx(d);
  deleteMtrxFromDevice(adev);
  deleteMtrxFromDevice(bdev);
  deleteMtrxFromDevice(cdev);
  deleteMtrxFromDevice(cdev2);


  //Time comparation
  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);

  std::cout << "My cuda_dgemm():" << std::endl;
  std::string timesStr = "";
  std::string gflopsStr = "";
  int n = N;
  while (n > 0)
  {
    float* a2 = createMtrx(n);
    float* b2 = createMtrx(n);
    //float* c = (float*)malloc(N * N * sizeof(float));
    float* adev2 = copyMtrxToDevice(n, a2);
    float* bdev2 = copyMtrxToDevice(n, b2);
    float* cdev2 = createMtrxOnDevice(n);

    dim3 threads(THREADS_N, THREADS_N);
    dim3 blocks(div_up(threads.x, n), div_up(threads.y, n));

    cudaEventRecord(e_start, 0);

    for (int it = 0; it < ITERS; it++)
      cuda_dgemm << < blocks, threads >> > (n, adev2, bdev2, cdev2);

    cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    cudaEventSynchronize(e_stop);

    float* c2 = copyMtrxFromDevice(n, cdev2);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    double time = elapsedTime / 1000. / ITERS;
    double gflops = getGflops(n, time);
    std::cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << std::endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    //printMtrx(c, n);

    deleteMtrx(a2);
    deleteMtrx(b2);
    deleteMtrx(c2);
    deleteMtrxFromDevice(adev2);
    deleteMtrxFromDevice(bdev2);
    deleteMtrxFromDevice(cdev2);

    n -= 500;
  }
  std::cout << timesStr << std::endl;
  std::cout << gflopsStr << std::endl;
  std::cout << "\n\n";

  std::cout << "cuBlas cublasSgemm():" << std::endl;
  timesStr = "";
  gflopsStr = "";
  n = N;
  while (n > 0)
  {
    float* a2 = createMtrx(n);
    float* b2 = createMtrx(n);
    //float* c = (float*)malloc(N * N * sizeof(float));
    float* adev2 = copyMtrxToDevice(n, a2);
    float* bdev2 = copyMtrxToDevice(n, b2);
    float* cdev2 = createMtrxOnDevice(n);

    cudaEventRecord(e_start, 0);

    for (int it = 0; it < ITERS; it++)
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, adev2, n, bdev2, n, &beta, cdev2, n);

    cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    cudaEventSynchronize(e_stop);

    float* c2 = copyMtrxFromDevice(n, cdev2);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    double time = elapsedTime / 1000. / ITERS;
    double gflops = getGflops(n, time);
    std::cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << std::endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    //printMtrx(c, n);

    deleteMtrx(a2);
    deleteMtrx(b2);
    deleteMtrx(c2);
    deleteMtrxFromDevice(adev2);
    deleteMtrxFromDevice(bdev2);
    deleteMtrxFromDevice(cdev2);

    n -= 500;
  }
  std::cout << timesStr << std::endl;
  std::cout << gflopsStr << std::endl;
  std::cout << "\n\n";

  cublasDestroy(handle);

  return 0;
}