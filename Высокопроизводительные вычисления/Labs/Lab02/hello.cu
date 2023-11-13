#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N (4)

__global__ void cuda_hello()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  printf("Hello World from GPU! %d %d\n", idx, idy);
}

int main()
{
  dim3 threads(N, N);
  dim3 blocks(N / threads.x, N / threads.y);
  cuda_hello << < threads, blocks >> > ();
  printf("Hello World from CPU!\n");
  return 0;
}