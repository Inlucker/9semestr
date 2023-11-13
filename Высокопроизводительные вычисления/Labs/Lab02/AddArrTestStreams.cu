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

// CUDA ядро для сложения двух массивов
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
    // Создаем события для измерения времени выполнения
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Размер массива
    int size = N;

    // Выделяем память на хосте
    int* h_a = (int*)malloc(size * sizeof(int));
    int* h_b = (int*)malloc(size * sizeof(int));
    int* h_c = (int*)malloc(size * sizeof(int));

    // Инициализация массивов
    for (int i = 0; i < size; i++)
    {
      h_a[i] = i;
      h_b[i] = 2 * i;
    }

    // Выделяем память на устройстве
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // Запускаем таймер
    cudaEventRecord(start);

    // Задаем конфигурацию блоков и нитей
    int threadsPerBlock = THREADS_NN;
    //int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = div_up(size, threadsPerBlock);
    //printf("threadsPerBlock = %d\nblocksPerGrid = %d\n", threadsPerBlock, blocksPerGrid);

    for (int it = 0; it < ITERS; it++)
    {
      // Копируем данные с хоста на устройство
      cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

      // Запускаем таймер
      //cudaEventRecord(start);

      // Вызываем ядро для сложения массивов на устройстве
      addArrays << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, size);

      // Останавливаем таймер
      //cudaEventRecord(stop);
      //cudaEventSynchronize(stop);
      // Измеряем время выполнения
      //float milliseconds = 0;
      //cudaEventElapsedTime(&milliseconds, start, stop);
      //printf("             addArrays() time = %f ms\n", milliseconds);

      // Копируем результат с устройства на хост
      cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Останавливаем таймер
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Измеряем время выполнения
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("             addArrays() time = %f ms\n", milliseconds / ITERS);

    //addArrays << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, size);
    //cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Выводим результат
    for (int i = 0; i < size; i++)
      if (h_a[i] + h_b[i] != h_c[i])
        printf("Error: %d + %d != %d\n", h_a[i], h_b[i], h_c[i]);

    //for (int i = 0; i < size; i++)
    //  printf("%d ", h_c[i]);
    //printf("\n");

    // Освобождаем память
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
    // Создаем события для измерения времени выполнения
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Размер массива
    int size = N;

    // Выделяем память на хосте
    int* h_a = (int*)malloc(size * sizeof(int));
    int* h_b = (int*)malloc(size * sizeof(int));
    int* h_c = (int*)malloc(size * sizeof(int));

    // Инициализация массивов
    for (int i = 0; i < size; i++)
    {
      h_a[i] = i;
      h_b[i] = 2 * i;
    }

    // Выделяем память на устройстве
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // Создание потоков
    cudaStream_t stream[CUDA_STREAMS_NUM];
    for (int i = 0; i < CUDA_STREAMS_NUM; i++)
      cudaStreamCreate(&stream[i]);
    int full_mem_size = N * sizeof(int);
    int part_mem_size = full_mem_size / CUDA_STREAMS_NUM;
    int part_size = size / CUDA_STREAMS_NUM;

    // Запускаем таймер
    cudaEventRecord(start);

    // Задаем конфигурацию блоков и нитей
    int threadsPerBlock = THREADS_NN;
    //int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    //int blocksPerGrid = div_up(size, threadsPerBlock);
    int blocksPerGrid = div_up(div_up(size, threadsPerBlock), CUDA_STREAMS_NUM);
    //printf("threadsPerBlock = %d\nblocksPerGrid = %d\n", threadsPerBlock, blocksPerGrid);

    for (int it = 0; it < ITERS; it++)
    {
      // Копируем данные с хоста на устройство
      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
      {
        cudaMemcpyAsync(d_a + i * part_size, h_a + i * part_size, part_mem_size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_b + i * part_size, h_b + i * part_size, part_mem_size, cudaMemcpyHostToDevice, stream[i]);
      }

      // Запускаем таймер
      //cudaEventRecord(start);

      // Вызываем ядро для сложения массивов на устройстве
      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
        addArrays << <blocksPerGrid, threadsPerBlock, 0, stream[i] >> > (d_a + i * part_size, d_b + i * part_size, d_c + i * part_size, part_size);

      // Останавливаем таймер
      //cudaEventRecord(stop);
      //cudaEventSynchronize(stop);
      // Измеряем время выполнения
      //float milliseconds = 0;
      //cudaEventElapsedTime(&milliseconds, start, stop);
      //printf("Cuda Streams addArrays() time = %f ms\n", milliseconds);

      // Копируем результат с устройства на хост
      for (int i = 0; i < CUDA_STREAMS_NUM; i++)
        cudaMemcpyAsync(h_c + i * part_size, d_c + i * part_size, part_mem_size, cudaMemcpyDeviceToHost, stream[i]);

      if (cudaDeviceSynchronize() != cudaSuccess)
        printf("cudaDeviceSynchronize() Error\n");
    }

    // Останавливаем таймер
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Измеряем время выполнения
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Cuda Streams addArrays() time = %f ms\n", milliseconds / ITERS);

    //for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    //  addArrays << <blocksPerGrid, threadsPerBlock, 0, stream[i] >> > (d_a + i * part_size, d_b + i * part_size, d_c + i * part_size, part_size);
    //for (int i = 0; i < CUDA_STREAMS_NUM; i++)
    //  cudaMemcpyAsync(h_c + i * part_size, d_c + i * part_size, part_mem_size, cudaMemcpyDeviceToHost, stream[i]);

    // Выводим результат
    for (int i = 0; i < size; i++)
      if (h_a[i] + h_b[i] != h_c[i])
        printf("Error: %d + %d != %d\n", h_a[i], h_b[i], h_c[i]);

    //for (int i = 0; i < size; i++)
    //  printf("%d ", h_c[i]);
    //printf("\n");

    // Освобождаем память
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
