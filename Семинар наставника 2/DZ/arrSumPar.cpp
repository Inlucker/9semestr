#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>

#define N 1000000 // 10000000 1000000 100000 10000
#define ITERS 10 // 10 100

void printArr(int* arr, int n)
{
  for (int i = 0; i < n; i++)
    printf("%d ", arr[i]);
  printf("\n");
}

void addArrays(int* a, int* b, int* result, int size)
{
#pragma omp parallel for
  for (int i = 0; i < size; i++)
    result[i] = a[i] + b[i];
}

int main()
{
  srand(time(NULL));

  int threads_n_gl = -1;
#pragma omp parallel
  {
    if (omp_get_thread_num() == 0)
    {
      threads_n_gl = omp_get_max_threads();
      //threads_n_gl = 2;
      printf("omp_get_max_threads() = %d\n", threads_n_gl);
    }
  }

  for (int gl = 0; gl < 3; gl ++)
  {
    int threads_n = threads_n_gl;
    while (threads_n > 0)
    {
      omp_set_num_threads(threads_n);
      int* arr1 = (int*)malloc(N * sizeof(int));
      int* arr2 = (int*)malloc(N * sizeof(int));
      int* sum = (int*)malloc(N * sizeof(int));
      double time_par = 0;
      for (int it = 0; it < ITERS; it++)
      {
        // Инициализация массивов случайными значениями
        for (int i = 0; i < N; i++)
        {
          arr1[i] = rand() % 100;
          arr2[i] = rand() % 100;
          sum[i] = 0;
        }
        //printArr(arr1, N);
        //printArr(arr2, N);

        int max_d = 0;
        int sum_th_n = 1;
        int cur_th_n = 1;
        while (sum_th_n < threads_n)
        {
          cur_th_n *= 2;
          sum_th_n += cur_th_n;
          max_d++;
        }

        // Засекаем время перед сортировкой
        double start_time = omp_get_wtime();

        // Сортировка Хоара с использованием OpenMP
        //quickSortParOld(arr, 0, N - 1, max_d);
        //quickSortParOld2(arr, 0, N - 1, max_d);

        addArrays(arr1, arr2, sum, N);

        // Сортировка Хоар последовательно
        //quickSortSeq(arr, 0, N - 1);

        // Засекаем время после сортировки
        double end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        time_par += elapsed_time;

        //printArr(sum, N);
      }
      free(arr1);
      free(arr2);
      free(sum);
      time_par /= ITERS;
      printf("Time taken (%d threads): %f seconds\n", threads_n, time_par);
      threads_n /= 2;
    }
  }
  return 0;
}
