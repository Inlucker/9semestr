#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>

#define N 100000 // 10000000 1000000 100000 10000
#define ITERS 100 // 10 100

void bubbleSortSeq(int arr[], int n)
{
  int i, j;
  for (i = 0; i < n - 1; i++)
  {
    for (j = 0; j < n - i - 1; j++)
    {
      if (arr[j] > arr[j + 1])
      {
        // Swap
        int temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }
}

void bubbleSortPar(int arr[], int n)
{
  int i, j;
#pragma omp parallel for private(i, j) shared(arr, n)
  for (i = 0; i < n - 1; i++)
  {
    for (j = 0; j < n - i - 1; j++)
    {
      if (arr[j] > arr[j + 1])
      {
        // Swap
        int temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }
}

void swap_old(int* a, int* b)
{
  int temp = *a;
  *a = *b;
  *b = temp;
}

void swap(int* a, int* b)
{
  if (*a == *b) return;
  *a = *a + *b;
  *b = *a - *b;
  *a = *a - *b;
}

void bubbleSortPar2(int arr[], int n)
{
  int i = 0, j = 0;
  int first;
  for (i = 0; i < n - 1; i++)
  {
    first = i % 2;
#pragma omp parallel for default(none), shared(arr,first,n)
    for (j = first; j < n - 1; j += 1)
      if (arr[j] > arr[j + 1])
        swap(&arr[j], &arr[j + 1]);
  }
}

void printArr(int* arr, int n)
{
  for (int i = 0; i < n; i++)
    printf("%d ", arr[i]);
  printf("\n");
}

bool isArrayasEqual(int* a, int* b, int n)
{
  for (int i = 0; i < n; i++)
  {
    if (a[i] != b[i])
    {
      printf("arrays doesn't equal at i = %d: %d != %d\n", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

int partition(int* arr, int low, int high)
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

void quickSortPar(int* arr, int low, int high, int max_d, int d = 0)
{
  if (low < high)
  {
    int pi = partition(arr, low, high);

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
    int pi = partition(arr, low, high);
    // Рекурсивно сортируем элементы, находящиеся до пивота
    quickSortSeq(arr, low, pi);
    // Рекурсивно сортируем элементы, находящиеся после пивота
    quickSortSeq(arr, pi + 1, high);
  }
}

int main()
{
  srand(time(NULL));

  int threads_n = -1;
#pragma omp parallel
  {
    if (omp_get_thread_num() == 0)
    {
      threads_n = omp_get_max_threads();
      //threads_n = 8;
      printf("omp_get_max_threads() = %d\n", threads_n);
    }
  }

  while (threads_n > 0)
  {
    omp_set_num_threads(threads_n);
    //int arr[N], arr_seq[N];
    int* arr = (int*)malloc(N * sizeof(int));
    int* arr_seq = (int*)malloc(N * sizeof(int));
    double time_par = 0;
    double time_seq = 0;
    for (int it = 0; it < ITERS; it++)
    {
      // Инициализация массивов случайными значениями
      for (int i = 0; i < N; i++)
      {
        //arr[i] = rand() % 100;
        arr[i] = N - i;
        arr_seq[i] = arr[i];
      }
      //printArr(arr, N);
      //printArr(arr_seq, N);

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
      #pragma omp parallel num_threads(threads_n)
      {
        #pragma omp single
        { quickSortPar(arr, 0, N - 1, max_d); }
      }

      // Засекаем время после сортировки
      double end_time = omp_get_wtime();
      double elapsed_time = end_time - start_time;
      time_par += elapsed_time;


      // Засекаем время перед последовательной сортировкой
      double start_time_seq = omp_get_wtime();

      // Последовательная сортировка Хоара
      quickSortSeq(arr_seq, 0, N - 1);

      // Засекаем время после последовательной сортировки
      double end_time_seq = omp_get_wtime();
      double elapsed_time_seq = end_time_seq - start_time_seq;
      time_seq += elapsed_time_seq;

      //printArr(arr, N);
      //printArr(arr_seq, N);

      //printf("Iteration %d:\nTime taken (%d threads): %f seconds\nTime taken (sequential): %f seconds\n\n", it + 1, threads_n, elapsed_time, elapsed_time_seq);
      if (isArrayasEqual(arr, arr_seq, N))
      {
        //printf("quickSortPar() == quickSortSeq()\n");
      }
      else
        printf("quickSortPar() != quickSortSeq()\n");
    }
    free(arr);
    free(arr_seq);
    time_par /= ITERS;
    time_seq /= ITERS;
    printf("Time taken (%d threads): %f seconds\n", threads_n, time_par);
    printf("Time taken (sequential): %f seconds\n", time_seq);
    threads_n /= 2;
  }

  return 0;
}
