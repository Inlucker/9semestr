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