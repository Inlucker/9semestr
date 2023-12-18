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