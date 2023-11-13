#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <string>

#define N (500)
#define THREADS_N (16)
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

__global__ void cuda_dgemmShared(int n, float* A, float* B, float* C)
{
  // Индексы текущего потока
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Shared-память для подматриц A и B
  __shared__ float sA[THREADS_N][THREADS_N];
  __shared__ float sB[THREADS_N][THREADS_N];

  // Результирующее значение для текущего потока
  float result = 0.0;

  // Цикл по блокам
  for (int i = 0; i < ceil((float)n / THREADS_N); ++i)
  {
    // Загрузка блоков матриц A и B в shared-память
    if (col < n && i * THREADS_N + threadIdx.y < n)
      sA[threadIdx.y][threadIdx.x] = A[(i * THREADS_N + threadIdx.y) * n + col];
    else
      sA[threadIdx.y][threadIdx.x] = 0.0;

    if (row < n && i * THREADS_N + threadIdx.x < n)
      sB[threadIdx.y][threadIdx.x] = B[row * n + i * THREADS_N + threadIdx.x];
    else
      sB[threadIdx.y][threadIdx.x] = 0.0;

    // Синхронизация для завершения загрузки данных в shared-память
    __syncthreads();

    // Умножение подматрицы A на подматрицу B в shared-памяти
    for (int j = 0; j < THREADS_N; ++j)
      result += sA[j][threadIdx.x] * sB[threadIdx.y][j];

    // Синхронизация перед следующей итерацией
    __syncthreads();
  }

  // Запись результата в матрицу C
  if (row < n && col < n)
    C[row * n + col] = result;
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

  float* a = createMtrx(N);
  float* b = createMtrx(N);
  /*for (int i = 0; i < N * N; i++)
    a[i] = i + 1;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      b[i * N + j] = j * N + i + 1;*/
  float* c = createMtrx(N);
  float* adev = copyMtrxToDevice(N, a);
  float* bdev = copyMtrxToDevice(N, b);
  float* cdev = createMtrxOnDevice(N);

  //printMtrx(a, N);
  //printMtrx(b, N);
  seq_dgemm(N, a, b, c);
  //printMtrx(c, N);

  //cuda_dgemmShared << < blocks, threads >> > (N, adev, bdev, cdev);
  cuda_dgemmShared << < blocks, threads >> > (N, adev, bdev, cdev);
  float* d = copyMtrxFromDevice(N, cdev);
  //printMtrx(d, N);

  if (compareMtrx(N, c, d))
    printf("Shared Memory cuda_dgemm() == seq_dgemm()\n\n");
  else
    printf("Shared Memory cuda_dgemm() != seq_dgemm()\n\n");

  deleteMtrx(a);
  deleteMtrx(b);
  deleteMtrx(c);
  deleteMtrx(d);
  deleteMtrxFromDevice(adev);
  deleteMtrxFromDevice(bdev);
  deleteMtrxFromDevice(cdev);

  //Time comparation
  /*
  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);

  std::cout << "Shared Memory cuda_dgemm():" << std::endl;
  std::string timesStr = "";
  std::string gflopsStr = "";
  int n = 1500;
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
      cuda_dgemmShared << < blocks, threads >> > (n, adev2, bdev2, cdev2);

    cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    cudaEventSynchronize(e_stop);

    //float* c2 = copyMtrxFromDevice(n, cdev2);

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
    //deleteMtrx(c2);
    deleteMtrxFromDevice(adev2);
    deleteMtrxFromDevice(bdev2);
    deleteMtrxFromDevice(cdev2);

    n -= 500;
  }
  std::cout << timesStr << std::endl;
  std::cout << gflopsStr << std::endl;
  std::cout << "\n\n";

  std::cout << "Usual cuda_dgemm():" << std::endl;
  timesStr = "";
  gflopsStr = "";
  n = 1500;
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

    //float* c2 = copyMtrxFromDevice(n, cdev2);

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
    //deleteMtrx(c2);
    deleteMtrxFromDevice(adev2);
    deleteMtrxFromDevice(bdev2);
    deleteMtrxFromDevice(cdev2);

    n -= 500;
  }
  std::cout << timesStr << std::endl;
  std::cout << gflopsStr << std::endl;
  std::cout << "\n\n";*/

  return 0;
}