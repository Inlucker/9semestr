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

  for (int gl = 0; gl < 3; gl ++)
  {
    int* arr = (int*)malloc(N * sizeof(int));
    double time_seq = 0;
    for (int it = 0; it < ITERS; it++)
    {
      // Инициализация массивов случайными значениями
      for (int i = 0; i < N; i++)
      {
        arr[i] = rand() % 100;
        //arr[i] = N - i;
      }
      //printArr(arr, N);

      // Засекаем время перед сортировкой
      double start_time = omp_get_wtime();

      // Сортировка Хоар последовательно
      quickSortSeq(arr, 0, N - 1);

      // Засекаем время после сортировки
      double end_time = omp_get_wtime();
      double elapsed_time = end_time - start_time;
      time_seq += elapsed_time;

      //printArr(arr, N);
      if (!isArraySorted(arr, N))
        printf("Massive is not sorted\n");
    }
    free(arr);
    time_seq /= ITERS;
    printf("Time taken (sequential): %f seconds\n", time_seq);
  }
  
  return 0;
}
