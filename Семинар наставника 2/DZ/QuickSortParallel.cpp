#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>

#define N 100000 // 10000000 1000000 100000 10000
#define ITERS 100 // 10 100

void printArr(int* arr, int n)
{
  for (int i = 0; i < n; i++)
    printf("%d ", arr[i]);
  printf("\n");
}

bool isArraySorted(int* arr, int n)
{
  if (n > 0)
  {
    int prev = arr[0];
    for (int i = 1; i < n; i++)
      if (arr[i] < prev)
        return false;
  }
  return true;
}

void swap(int* a, int* b)
{
  if (*a == *b) return;
  *a = *a + *b;
  *b = *a - *b;
  *a = *a - *b;
}

int partitionLomuto(int* arr, int low, int high)
{
  int pivot = arr[high];
  int i = (low - 1);

  for (int j = low; j < high; j++)
    if (arr[j] <= pivot)
    {
      i++;
      swap(&arr[i], &arr[j]);
    }

  swap(&arr[i + 1], &arr[high]);
  return (i + 1);
}

int partitionHoar(int* arr, int low, int high)
{
  size_t i = low;
  size_t j = high;
  int pivot = arr[(i + j) / 2];
  while (1)
  {
    while (arr[i] < pivot)
      ++i;
    while (arr[j] > pivot)
      --j;

    if (i >= j)
      break;

    swap(&arr[i++], &arr[j--]);
  }
  return j;
}

//Working only with 2 threads - with more WRONG
void quickSortParOld(int* arr, int low, int high, int max_d, int d = 0)
{
  if (low < high)
  {
    int pi = partitionHoar(arr, low, high);

    if (d < max_d)
    {
      #pragma omp parallel sections
      {
        #pragma omp section
        {
          //printf("omp_get_thread_num() = %d\n", omp_get_thread_num());
          // Рекурсивно сортируем элементы, находящиеся до пивота
          quickSortParOld(arr, low, pi, max_d, d + 1);
        }
        #pragma omp section
        {
          //printf("omp_get_thread_num() = %d\n", omp_get_thread_num());
          // Рекурсивно сортируем элементы, находящиеся после пивота
          quickSortParOld(arr, pi + 1, high, max_d, d + 1);
        }
      }
      //#pragma omp taskwait
    }
    else
    {
      {
        //printf("omp_get_thread_num() = %d\n", omp_get_thread_num());
        // Рекурсивно сортируем элементы, находящиеся до пивота
        quickSortParOld(arr, low, pi, max_d, d + 1);
        // Рекурсивно сортируем элементы, находящиеся после пивота
        quickSortParOld(arr, pi + 1, high, max_d, d + 1);
      }
    }
  }
}

void quickSortParOld2(int* arr, int low, int high, int max_d, int d = 0)
{
  if (low < high)
  {
    int pi = partitionHoar(arr, low, high);

    #pragma omp task if (d < max_d)
    {
      //printf("omp_get_thread_num() = %d\n", omp_get_thread_num());
      quickSortParOld2(arr, low, pi, max_d, d + 1);
    }
    #pragma omp task if (d < max_d)
    {
      //printf("omp_get_thread_num() = %d\n", omp_get_thread_num());
      quickSortParOld2(arr, pi + 1, high, max_d, d + 1);
    }
  }
}

void quickSortPar(int* arr, int low, int high, int max_d, int d = 0)
{
  if (low < high)
  {
    int pi = partitionHoar(arr, low, high);

    if (d < max_d)
    {
      #pragma omp task
      {
        //printf("omp_get_thread_num() = %d\n", omp_get_thread_num());
        // Рекурсивно сортируем элементы, находящиеся до пивота
        quickSortPar(arr, low, pi, max_d, d + 1);
      }
      #pragma omp task
      {
        //printf("omp_get_thread_num() = %d\n", omp_get_thread_num());
        // Рекурсивно сортируем элементы, находящиеся после пивота
        quickSortPar(arr, pi + 1, high, max_d, d + 1);
      }
    }
    else
    {
      {
        //printf("omp_get_thread_num() = %d\n", omp_get_thread_num());
        // Рекурсивно сортируем элементы, находящиеся до пивота
        quickSortPar(arr, low, pi, max_d, d + 1);
        // Рекурсивно сортируем элементы, находящиеся после пивота
        quickSortPar(arr, pi + 1, high, max_d, d + 1);
      }
    }
  }
}

void quickSortSeq(int* arr, int low, int high)
{
  if (low < high)
  {
    int pi = partitionHoar(arr, low, high);
    // Рекурсивно сортируем элементы, находящиеся до пивота
    quickSortSeq(arr, low, pi);
    // Рекурсивно сортируем элементы, находящиеся после пивота
    quickSortSeq(arr, pi + 1, high);
  }
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
      int* arr = (int*)malloc(N * sizeof(int));
      double time_par = 0;
      for (int it = 0; it < ITERS; it++)
      {
        // Инициализация массивов случайными значениями
        for (int i = 0; i < N; i++)
        {
          arr[i] = rand() % 100;
          //arr[i] = N - i;
        }
        //printArr(arr, N);

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

        #pragma omp parallel num_threads(threads_n)
        {
          #pragma omp single
          { quickSortPar(arr, 0, N - 1, max_d); }
        }

        // Сортировка Хоар последовательно
        //quickSortSeq(arr, 0, N - 1);

        // Засекаем время после сортировки
        double end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        time_par += elapsed_time;

        //printArr(arr, N);
        if (!isArraySorted(arr, N))
          printf("Massive is not sorted\n");
      }
      free(arr);
      time_par /= ITERS;
      printf("Time taken (%d threads): %f seconds\n", threads_n, time_par);
      threads_n /= 2;
    }
  }
  return 0;
}
