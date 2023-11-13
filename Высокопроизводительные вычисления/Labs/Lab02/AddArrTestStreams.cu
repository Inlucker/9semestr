#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <string>

#define N 67108864
#define THREADS_NN 256
#define CUDA_STREAMS_NUM 8
#define ITERS 10

// CUDA ���� ��� �������� ���� ��������
__global__ void addArrays(int* a, int* b, int* c, int size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("tid = %d\n", tid);
  if (tid < size)
  {
    c[tid] = a[tid] + b[tid];
    //printf("%d = %d + %d\n", c[tid], a[tid], b[tid]);
  }
}

int div_up(int x, int y)
{
  return (x - 1) / y + 1;
}

int main()
{
  //without cuda streams
  {
    // ������� ������� ��� ��������� ������� ����������
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ������ �������
    int size = N;

    // �������� ������ �� �����
    int* h_a = (int*)malloc(size * sizeof(int));
    int* h_b = (int*)malloc(size * sizeof(int));
    int* h_c = (int*)malloc(size * sizeof(int));

    // ������������� ��������
    for (int i = 0; i < size; i++)
    {
      h_a[i] = i;
      h_b[i] = 2 * i;
    }

    // �������� ������ �� ����������
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // ��������� ������
    cudaEventRecord(start);

    // ������ ������������ ������ � �����
    int threadsPerBlock = THREADS_NN;
    //int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = div_up(size, threadsPerBlock);
    //printf("threadsPerBlock = %d\nblocksPerGrid = %d\n", threadsPerBlock, blocksPerGrid);

    for (int it = 0; it < ITERS; it++)
    {
      // �������� ������ � ����� �� ����������
      cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

      // ��������� ������
      //cudaEventRecord(start);

      // �������� ���� ��� �������� �������� �� ����������
      addArrays << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, size);

      // ������������� ������
      //cudaEventRecord(stop);
      //cudaEventSynchronize(stop);
      // �������� ����� ����������
      //float milliseconds = 0;
      //cudaEventElapsedTime(&milliseconds, start, stop);
      //printf("             addArrays() time = %f ms\n", milliseconds);

      // �������� ��������� � ���������� �� ����
      cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // ������������� ������
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // �������� ����� ����������
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("             addArrays() time = %f ms\n", milliseconds / ITERS);

    //addArrays << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, size);
    //cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // ������� ���������
    for (int i = 0; i < size; i++)
      if (h_a[i] + h_b[i] != h_c[i])
        printf("Error: %d + %d != %d\n", h_a[i], h_b[i], h_c[i]);

    //for (int i = 0; i < size; i++)
    //  printf("%d ", h_c[i]);
    //printf("\n");

    // ����������� ������
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  //with cuda streams
  {
    // ������� ������� ��� ��������� ������� ����������
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ������ �������
    int size = N;

    // �������� ������ �� �����
    int* h_a = (int*)malloc(size * sizeof(int));
    int* h_b = (int*)malloc(size * sizeof(int));
    int* h_c = (int*)malloc(size * sizeof(int));

    // ������������� ��������
    for (int i = 0; i < size; i++)
    {
      h_a[i] = i;
      h_b[i] = 2 * i;
    }

    // �������� ������ �� ����������
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // �������� �������
    cudaStream_t stream[CUDA_STREAMS_NUM];
    for (int i = 0; i < CUDA_STREAMS_NUM; i++)
      cudaStreamCreate(&stream[i]);
    int full_mem_size = N * sizeof(int);
    int part_mem_size = full_mem_size / CUDA_STREAMS_NUM;
    int part_size = size / CUDA_STREAMS_NUM;

    // ��������� ������
    cudaEventRecord(start);

    // ������ ������������ ������ � �����
    int threadsPerBlock = THREADS_NN;
    //int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    //int blocksPerGrid = div_up(size, threadsPerBlock);
    int blocksPerGrid = div_up(div_up(size, threadsPerBlock), CUDA_STREAMS_NUM);
    //printf("threadsPerBlock = %d\nblocksPerGrid = %d\n", threadsPerBlock, blocksPerGrid);

    for (int it = 0; it < ITERS; it++)
    {
      // �������� ������ � ����� �� ����������
      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
      {
        cudaMemcpyAsync(d_a + i * part_size, h_a + i * part_size, part_mem_size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_b + i * part_size, h_b + i * part_size, part_mem_size, cudaMemcpyHostToDevice, stream[i]);
      }

      // ��������� ������
      //cudaEventRecord(start);

      // �������� ���� ��� �������� �������� �� ����������
      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
        addArrays << <blocksPerGrid, threadsPerBlock, 0, stream[i] >> > (d_a + i * part_size, d_b + i * part_size, d_c + i * part_size, part_size);

      // ������������� ������
      //cudaEventRecord(stop);
      //cudaEventSynchronize(stop);
      // �������� ����� ����������
      //float milliseconds = 0;
      //cudaEventElapsedTime(&milliseconds, start, stop);
      //printf("Cuda Streams addArrays() time = %f ms\n", milliseconds);

      // �������� ��������� � ���������� �� ����
      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
        cudaMemcpyAsync(h_c + i * part_size, d_c + i * part_size, part_mem_size, cudaMemcpyDeviceToHost, stream[i]);

      if (cudaDeviceSynchronize() != cudaSuccess)
        printf("cudaDeviceSynchronize() Error\n");
    }

    // ������������� ������
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // �������� ����� ����������
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Cuda Streams addArrays() time = %f ms\n", milliseconds / ITERS);

    //for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    //  addArrays << <blocksPerGrid, threadsPerBlock, 0, stream[i] >> > (d_a + i * part_size, d_b + i * part_size, d_c + i * part_size, part_size);
    //for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    //  cudaMemcpyAsync(h_c + i * part_size, d_c + i * part_size, part_mem_size, cudaMemcpyDeviceToHost, stream[i]);

    // ������� ���������
    for (int i = 0; i < size; i++)
      if (h_a[i] + h_b[i] != h_c[i])
        printf("Error: %d + %d != %d\n", h_a[i], h_b[i], h_c[i]);

    //for (int i = 0; i < size; i++)
    //  printf("%d ", h_c[i]);
    //printf("\n");

    // ����������� ������
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < CUDA_STREAMS_NUM; i++)
      cudaStreamDestroy(stream[i]);
  }

  return 0;
}
