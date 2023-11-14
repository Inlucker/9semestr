#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <string>

#define N (512)
#define THREADS_N (32)
#define ITERS (10)
#define CUDA_STREAMS_NUM2 (2)
#define CUDA_STREAMS_NUM (CUDA_STREAMS_NUM2*CUDA_STREAMS_NUM2)

void randMtrx(float* mtrx, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      mtrx[n * j + i] = rand() % 9 + 1;
}

/*float* createMtrx(int n)
{
  float* mtrx = (float*)malloc(n * n * sizeof(float));
  randMtrx(mtrx, n);
  return mtrx;
}*/

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
  if (idx >= n || idy >= n)
    return;
  int i = idx;
  int j = idy;

  float res = 0;
  for (int k = 0; k < n; k++)
    res += (a[n * k + i] * b[n * j + k]);
  c[n * j + i] = res;
}

__global__ void cuda_dgemmAsync(int n, float* a, float* b, float* c, int iter)
{
  int idx2d = blockIdx.x * blockDim.x + threadIdx.x;
  int idy2d = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = iter + idy2d * gridDim.y * THREADS_N + idx2d;
  //printf("idx = %d\n", idx);
  int i = idx % n;
  int j = idx / n;
  if (i >= n || j >= n)
    return;

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
      int i2 = i / n;
      int j = i % n;
      printf("compareMtrx() Error: i = %d, j = %d, %.0f != %.0f\n", i2, j, a[i], b[i]);
      return false;
    }
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

  cudaStream_t stream[CUDA_STREAMS_NUM];
  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    cudaStreamCreate(&stream[i]);

  float* a = createMtrxPinned(N);
  float* b = createMtrxPinned(N);
  float* c = createMtrxPinned(N);

  int full_size = N * N;
  int part_size = full_size / CUDA_STREAMS_NUM;
  int mem_full_size = N * N * sizeof(float);
  int mem_part_size = mem_full_size / CUDA_STREAMS_NUM;
  float* adev = createMtrxOnDevice(N);
  float* bdev = createMtrxOnDevice(N);
  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
  {
    cudaMemcpyAsync(adev + i * part_size, a + i * part_size, mem_part_size, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(bdev + i * part_size, b + i * part_size, mem_part_size, cudaMemcpyHostToDevice, stream[i]);
  }

  float* cdev = createMtrxOnDevice(N);

  //printMtrx(a, N);
  //printMtrx(b, N);
  seq_dgemm(N, a, b, c);
  //printMtrx(c, N);

  int dx = div_up(N * N, CUDA_STREAMS_NUM);

  dim3 threads(THREADS_N, THREADS_N);
  dim3 blocks(div_up(N, threads.x * CUDA_STREAMS_NUM2), div_up(N, threads.y * CUDA_STREAMS_NUM2));
  printf("threads.x = %d threads.y = %d\n", threads.x, threads.y);
  printf("blocks.x = %d blocks.y = %d\n\n", blocks.x, blocks.y);

  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    cuda_dgemmAsync << < blocks, threads, 0, stream[i] >> > (N, adev, bdev, cdev, dx * i);

  float* d = (float*)malloc(N * N * sizeof(float));
  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    cudaMemcpyAsync(d + i * part_size, cdev + i * part_size, mem_part_size, cudaMemcpyDeviceToHost, stream[i]);

  if (cudaDeviceSynchronize() != cudaSuccess)
    printf("cudaDeviceSynchronize() Error\n");
  //printMtrx(d, N);

  if (compareMtrx(N, c, d))
    printf("Cuda Streams cuda_dgemm() == seq_dgemm()\n\n");
  else
    printf("Cuda Streams cuda_dgemm() != seq_dgemm()\n\n");

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

  int start_n = 5000;
  int step_n = 1000;

  std::cout << "Cuda Streams cuda_dgemm():" << std::endl;
  std::string timesStr = "";
  std::string gflopsStr = "";
  int n = start_n;
  while (n > 0)
  {
    float* a2 = createMtrxPinned(n);
    float* b2 = createMtrxPinned(n);
    float* d2 = (float*)malloc(n * n * sizeof(float));
    float* adev2 = createMtrxOnDevice(n);
    float* bdev2 = createMtrxOnDevice(n);
    float* cdev2 = createMtrxOnDevice(n);

    cudaEventRecord(e_start, 0);

    for (int it = 0; it < ITERS; it++)
    {
      int full_size = n * n;
      int part_size = full_size / CUDA_STREAMS_NUM;
      int mem_full_size = n * n * sizeof(float);
      int mem_part_size = mem_full_size / CUDA_STREAMS_NUM;
      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
      {
        cudaMemcpyAsync(adev2 + i * part_size, a2 + i * part_size, mem_part_size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(bdev2 + i * part_size, b2 + i * part_size, mem_part_size, cudaMemcpyHostToDevice, stream[i]);
      }

      int dx = div_up(n * n, CUDA_STREAMS_NUM);

      dim3 threads(THREADS_N, THREADS_N);
      dim3 blocks(div_up(N, threads.x * CUDA_STREAMS_NUM2), div_up(N, threads.y * CUDA_STREAMS_NUM2));

      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
        cuda_dgemmAsync << < blocks, threads, 0, stream[i] >> > (N, adev2, bdev2, cdev2, dx * i);

      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
        cudaMemcpyAsync(d2 + i * part_size, cdev2 + i * part_size, mem_part_size, cudaMemcpyDeviceToHost, stream[i]);

      cudaDeviceSynchronize();
    }

    cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    cudaEventSynchronize(e_stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    double time = elapsedTime / 1000. / ITERS;
    double gflops = getGflops(n, time);
    std::cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << std::endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    deleteMtrxPinned(a2);
    deleteMtrxPinned(b2);
    deleteMtrx(d2);
    deleteMtrxFromDevice(adev2);
    deleteMtrxFromDevice(bdev2);
    deleteMtrxFromDevice(cdev2);

    n -= step_n;
  }
  std::cout << timesStr << std::endl;
  std::cout << gflopsStr << std::endl;
  std::cout << "\n\n";

  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    cudaStreamDestroy(stream[i]);

  std::cout << "Usual cuda_dgemm():" << std::endl;
  timesStr = "";
  gflopsStr = "";
  n = start_n;
  while (n > 0)
  {
    float* a2 = createMtrxPinned(n);
    float* b2 = createMtrxPinned(n);
    float* c2 = (float*)malloc(n * n * sizeof(float));
    float* adev2 = NULL; cudaMalloc((void**)&adev2, n * n * sizeof(float));
    float* bdev2 = NULL; cudaMalloc((void**)&bdev2, n * n * sizeof(float));
    float* cdev2 = createMtrxOnDevice(n);

    cudaEventRecord(e_start, 0);

    for (int it = 0; it < ITERS; it++)
    {
      cudaMemcpy(adev2, a2, n * n * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(bdev2, b2, n * n * sizeof(float), cudaMemcpyHostToDevice);

      dim3 threads(THREADS_N, THREADS_N);
      dim3 blocks(div_up(threads.x, n), div_up(threads.y, n));

      cuda_dgemm << < blocks, threads >> > (n, adev2, bdev2, cdev2);

      cudaMemcpy(c2, cdev2, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    cudaEventSynchronize(e_stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    double time = elapsedTime / 1000. / ITERS;
    double gflops = getGflops(n, time);
    std::cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << std::endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    deleteMtrxPinned(a2);
    deleteMtrxPinned(b2);
    deleteMtrx(c2);
    deleteMtrxFromDevice(adev2);
    deleteMtrxFromDevice(bdev2);
    deleteMtrxFromDevice(cdev2);

    n -= step_n;
  }
  std::cout << timesStr << std::endl;
  std::cout << gflopsStr << std::endl;
  std::cout << "\n\n";

  return 0;
}