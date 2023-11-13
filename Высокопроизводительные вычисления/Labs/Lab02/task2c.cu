#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <string>

#define N (500)
#define THREADS_N (32)
#define ITERS (1)
#define CUDA_STREAMS_NUM (8)

void randMtrx(float* mtrx, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      mtrx[n * j + i] = rand() % 10;
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

float* copyMtrxToDeviceAsync(int n, float*& mtrx, cudaStream_t stream[CUDA_STREAMS_NUM])
{
  int full_size = n * n * sizeof(float);
  int part_size = full_size / CUDA_STREAMS_NUM;
  float* mtrx_dev = NULL;
  cudaMalloc((void**)&mtrx_dev, n * n * sizeof(float));
  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    cudaMemcpyAsync(mtrx_dev + i * part_size, mtrx + i * part_size, full_size, cudaMemcpyHostToDevice, stream[i]);
  return mtrx_dev;
}

float* copyMtrxFromDevice(int n, float*& mtrx_dev)
{
  float* mtrx = (float*)malloc(n * n * sizeof(float));
  cudaMemcpy(mtrx, mtrx_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);
  return mtrx;
}

float* copyMtrxFromDeviceAsync(int n, float*& mtrx_dev, cudaStream_t stream[CUDA_STREAMS_NUM])
{
  int full_size = n * n * sizeof(float);
  int part_size = full_size / CUDA_STREAMS_NUM;
  float* mtrx = (float*)malloc(n * n * sizeof(float));
  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    cudaMemcpyAsync(mtrx + i * part_size, mtrx_dev + i * part_size, full_size, cudaMemcpyDeviceToHost, stream[i]);
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
  int idx = iter + (blockIdx.x * blockDim.x + threadIdx.x);
  //printf("idx = %d\n", idx);
  int i = idx / n;
  int j = idx % n;
  if (i >= n || j >= n)
    return;
  //c[n * j + i] = 0;
  //for (int k = 0; k < n; k++)
  //  c[n * j + i] += (a[n * k + i] * b[n * j + k]);

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
  float* adev = copyMtrxToDeviceAsync(N, a, stream);
  float* bdev = copyMtrxToDeviceAsync(N, b, stream);
  //float* adev = copyMtrxToDevice(N, a);
  //float* bdev = copyMtrxToDevice(N, b);
  float* cdev = createMtrxOnDevice(N);

  //printMtrx(a, N);
  //printMtrx(b, N);
  seq_dgemm(N, a, b, c);
  //printMtrx(c, N);

  //Without cudaStream_t
  /*dim3 threads(THREADS_N, THREADS_N);
  printf("threads.x = %d threads.y = %d\n", threads.x, threads.y);
  dim3 blocks(div_up(N, threads.x), div_up(N, threads.y));
  printf("blocks.x = %d blocks.y = %d\n\n", blocks.x, blocks.y);
  cuda_dgemm << < blocks, threads >> > (N, adev, bdev, cdev);*/

  int full_size = N * N;
  int part_size = full_size / CUDA_STREAMS_NUM;

  int x[CUDA_STREAMS_NUM];
  int dx = div_up(N * N, CUDA_STREAMS_NUM);
  x[0] = 0;
  for (int i = 1; i < CUDA_STREAMS_NUM; i++)
    x[i] = x[i - 1] + dx;

  int threads_n = THREADS_N * THREADS_N;
  dim3 threads(threads_n);
  printf("threads.x = %d threads.y = %d\n", threads.x, threads.y);
  dim3 blocks(div_up(dx, threads_n));
  printf("blocks.x = %d blocks.y = %d\n\n", blocks.x, blocks.y);

  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    cuda_dgemmAsync << < div_up(dx, threads_n), threads_n, 0, stream[i] >> > (N, adev, bdev, cdev, x[i]);

  //Без этого не работает
  cudaDeviceSynchronize();

  float* d = copyMtrxFromDeviceAsync(N, cdev, stream);
  //float* d = copyMtrxFromDevice(N, cdev);

  if (cudaDeviceSynchronize() != cudaSuccess)
    printf("cudaDeviceSynchronize() Error\n");
  //printMtrx(d, N);

  //for (int i = 0; i < CUDA_STREAMS_NUM; i++)
  //  cudaStreamDestroy(stream[i]);

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

  //Time comparation //ПОПРОБОВАТЬ CHRONO??????????????????????????????????????????????????????????????????????????????
  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);

  std::cout << "Cuda Streams cuda_dgemm():" << std::endl;
  std::string timesStr = "";
  std::string gflopsStr = "";
  int n = 1500;
  while (n > 0)
  {
    float* a2 = createMtrxPinned(n);
    float* b2 = createMtrxPinned(n);

    cudaEventRecord(e_start, 0);
    //float* c = (float*)malloc(n * n * sizeof(float));
    float* adev2 = copyMtrxToDeviceAsync(n, a2, stream);
    float* bdev2 = copyMtrxToDeviceAsync(n, b2, stream);
    float* cdev2 = createMtrxOnDevice(n);

    //int full_size = n * n;
    //int part_size = full_size / CUDA_STREAMS_NUM;

    int x[CUDA_STREAMS_NUM];
    int dx = div_up(n * n, CUDA_STREAMS_NUM);
    x[0] = 0;
    for (int i = 1; i < CUDA_STREAMS_NUM; i++)
      x[i] = x[i - 1] + dx;

    //cudaEventRecord(e_start, 0);

    int threads_n = THREADS_N * THREADS_N;
    dim3 threads(threads_n);
    dim3 blocks(div_up(dx, threads_n));
    //printf("threads.x = %d threads.y = %d\n", threads.x, threads.y);
    //printf("blocks.x = %d blocks.y = %d\n", blocks.x, blocks.y);
    //for (int it = 0; it < ITERS; it++)
      //for (int i = 0; i < CUDA_STREAMS_NUM; i++)
        //cuda_dgemmAsync << < blocks, threads, 0, stream[i] >> > (N, adev2, bdev2, cdev2, x[i]);

    /*dim3 threads(THREADS_N, THREADS_N);
    dim3 blocks(div_up(n, threads.x*4), div_up(n, threads.y*4));
    printf("threads.x = %d threads.y = %d\n", threads.x, threads.y);
    printf("blocks.x = %d blocks.y = %d\n", blocks.x, blocks.y);
    for (int i = 0; i < CUDA_STREAMS_NUM; i++)
      cuda_dgemm << < blocks, threads, 0, stream[i] >> > (N, adev2, bdev2, cdev2);*/

    //cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    //cudaEventSynchronize(e_stop);

    //if (cudaDeviceSynchronize() != cudaSuccess)
    //  printf("cudaDeviceSynchronize() Error\n");

    float* c2 = copyMtrxFromDeviceAsync(n, cdev2, stream);

    //if (cudaDeviceSynchronize() != cudaSuccess)
    //  printf("cudaDeviceSynchronize() Error\n");

    cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    cudaEventSynchronize(e_stop);

    if (cudaDeviceSynchronize() != cudaSuccess)
      printf("cudaDeviceSynchronize() Error\n");

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    double time = elapsedTime / 1000.;
    double gflops = getGflops(n, time);
    std::cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << std::endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    //printMtrx(c, n);

    deleteMtrxPinned(a2);
    deleteMtrxPinned(b2);
    deleteMtrx(c2);
    deleteMtrxFromDevice(adev2);
    deleteMtrxFromDevice(bdev2);
    deleteMtrxFromDevice(cdev2);

    n -= 500;
  }
  std::cout << timesStr << std::endl;
  std::cout << gflopsStr << std::endl;
  std::cout << "\n\n";

  for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    cudaStreamDestroy(stream[i]);

  std::cout << "Usual cuda_dgemm():" << std::endl;
  timesStr = "";
  gflopsStr = "";
  n = 1500;
  while (n > 0)
  {
    float* a2 = createMtrxPinned(n);
    float* b2 = createMtrxPinned(n);

    cudaEventRecord(e_start, 0);

    //float* c = (float*)malloc(N * N * sizeof(float));
    float* adev2 = copyMtrxToDevice(n, a2);
    float* bdev2 = copyMtrxToDevice(n, b2);
    float* cdev2 = createMtrxOnDevice(n);

    dim3 threads(THREADS_N, THREADS_N);
    dim3 blocks(div_up(threads.x, n), div_up(threads.y, n));

    //cudaEventRecord(e_start, 0);

    //for (int it = 0; it < ITERS; it++)
      //cuda_dgemm << < blocks, threads >> > (n, adev2, bdev2, cdev2);

    //cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    //cudaEventSynchronize(e_stop);

    float* c2 = copyMtrxFromDevice(n, cdev2);

    cudaEventRecord(e_stop, 0);// 0 означает поток CUDA 0
    cudaEventSynchronize(e_stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    double time = elapsedTime / 1000. / ITERS;
    double gflops = getGflops(n, time);
    std::cout << "N = " << n << " matrices time : " << time << " s; GFLOPS = " << gflops << std::endl;
    timesStr = "(" + std::to_string(n) + ", " + std::to_string(time) + ")" + timesStr;
    gflopsStr = "(" + std::to_string(n) + ", " + std::to_string(gflops) + ")" + gflopsStr;

    //printMtrx(c, n);

    deleteMtrxPinned(a2);
    deleteMtrxPinned(b2);
    deleteMtrx(c2);
    deleteMtrxFromDevice(adev2);
    deleteMtrxFromDevice(bdev2);
    deleteMtrxFromDevice(cdev2);

    n -= 500;
  }
  std::cout << timesStr << std::endl;
  std::cout << gflopsStr << std::endl;
  std::cout << "\n\n";

  return 0;
}