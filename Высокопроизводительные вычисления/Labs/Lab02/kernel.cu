#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

//#define N (1024)
#define N (4)
#define THREADS_PER_BLOCK 512
//#define BLOCK_SIZE 16 // submatrix size
#define BLOCK_SIZE 4 // submatrix size

void randMtrx(float* mtrx, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      mtrx[n * j + i] = rand() % 10;
}

float* createMtrxCuda(int n, float*& mtrx)
{
  mtrx = new float[n * n];
  randMtrx(mtrx, n);
  float* mtrx_dev = NULL;
  cudaMalloc((void**)&mtrx_dev, n*n*sizeof(float));
  cudaMemcpy(mtrx_dev, mtrx, n * n * sizeof(float), cudaMemcpyHostToDevice);
  return mtrx_dev;
}

void deleteMtrx(float*& mtrx)
{
  delete[] mtrx;
  mtrx = NULL;
}

void deleteMtrxCuda(float*& mtrx)
{
  cudaFree(mtrx);
  mtrx = NULL;
}

void deleteMtrxs(float*& mtrx, float*& mtrx_dev)
{
  deleteMtrx(mtrx);
  deleteMtrxCuda(mtrx_dev);
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
  int j = idy;
  int i = idx;
  c[n * j + i] = 0;
  for (int k = 0; k < n; k++)
    c[n * j + i] += (a[n * k + i] * b[n * j + k]);
}

bool compareMtrx(int n, float* a, float* b)
{
  for (int i = 0; i < n * n; i++)
    if (a[i] != b[i])
      return false;
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
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks(N / threads.x, N / threads.y);
  //cuda_hello << < threads, blocks >> > ();

  float *a = NULL, *b = NULL, *c = NULL;
  float* adev = createMtrxCuda(N, a);
  float* bdev = createMtrxCuda(N, b);
  float* cdev = createMtrxCuda(N, c);

  printMtrx(a, N);
  printMtrx(b, N);
  seq_dgemm(N, a, b, c);
  printMtrx(c, N);

  cuda_dgemm << < threads, blocks >> > (N, adev, bdev, cdev);
  float* d = new float[N * N];
  cudaMemcpy(d, cdev, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  printMtrx(d, N);

  if (compareMtrx(N, c, d))
    printf("cuda_dgemm() == seq_dgemm()\n");
  else
    printf("cuda_dgemm() != seq_dgemm()\n");

  deleteMtrxs(a, adev);
  deleteMtrxs(b, bdev);
  deleteMtrxs(c, cdev);
  delete[] d;

  //cuda_hello << < N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > ();
  //printf("Hello World from CPU!\n");
  return 0;
}

/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/