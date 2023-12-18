import numpy as np
from numba import config, njit, threading_layer, prange
import timeit
def isArraySorted(arr):
    n = len(arr)
    if (n > 0):
        prev = arr[0]
        for i in range(n):
            if (arr[i] < prev):
                return False
    return True;

@njit(parallel=True)
def quick_sort(arr, low, high):
    if low < high:
        #pi = partition(arr, low, high)
        # partitionHoar
        i = low
        j = high
        pivot = arr[(i + j) // 2]
        while (1):
            while (arr[i] < pivot):
                i = i + 1
            while (arr[j] > pivot):
                j = j - 1

            if (i >= j):
                break

            arr[i], arr[j] = arr[j], arr[i]
            i = i + 1
            j = j - 1
        pi = j
        #quick_sort(arr, low, pi)
        #quick_sort(arr, pi + 1, high)
        for id in prange(2):
            if (id == 0):
                quick_sort(arr, low, pi)
            else:
                quick_sort(arr, pi + 1, high)
        '''
        with Parallel(2) as p:
            if p.thread_num == 0:
                quick_sort(arr, low, pi - 1)
            else:
                quick_sort(arr, pi + 1, high)
        '''

@njit
def partition(arr, low, high):
    i = low
    j = high
    pivot = arr[(i + j) // 2]
    while (1):
        while (arr[i] < pivot):
            i = i + 1
        while (arr[j] > pivot):
            j = j - 1

        if (i >= j):
            break

        arr[i], arr[j] = arr[j], arr[i]
        i = i + 1
        j = j - 1
    return j

@njit
def quickSortSeq(arr, low, high):
    if (low < high):
        # partitionHoar
        i = low
        j = high
        pivot = arr[(i + j) // 2]
        while (1):
            while (arr[i] < pivot):
                i = i + 1
            while (arr[j] > pivot):
                j = j - 1

            if (i >= j):
                break

            arr[i], arr[j] = arr[j], arr[i]
            i = i + 1
            j = j - 1
        pi = j

        # recurcive call
        quickSortSeq(arr, low, pi)
        quickSortSeq(arr, pi + 1, high)

@njit(parallel=True)
def quickSortPar(arr, low, high):
    if low < high:
        i = low
        j = high
        pivot = arr[(i + j) // 2]
        while True:
            while arr[i] < pivot:
                i = i + 1
            while arr[j] > pivot:
                j = j - 1

            if i >= j:
                break

            arr[i], arr[j] = arr[j], arr[i]
            i = i + 1
            j = j - 1
        pi = j

        '''
        for i in prange(2):  # распараллеливание рекурсивных вызовов
            quickSortSeq(arr, low, pi)
            quickSortSeq(arr, pi + 1, high)
        '''
        for id in prange(2):
            if (id == 0):
                quickSortSeq(arr, low, pi)
            else:
                quickSortSeq(arr, pi + 1, high)

config.THREADING_LAYER = 'omp'
config.NUMBA_DEFAULT_NUM_THREADS = 2
'''
# Генерация случайного массива
N = 10000 # Размер массива
arr = np.random.randint(1, 100, N)
print(arr)

#quick_sort(arr, 0, N-1)
a = arr.copy()
quickSortPar(a, 0, N-1)
a2 = arr.copy()
quickSortSeq(a2, 0, N-1)

#print(a)
if (not isArraySorted(a)):
    print("a Massive is not sorted\n")
if (not isArraySorted(a2)):
    print("a2 Massive is not sorted\n")
'''

# Замер времени выполнения
'''
arr = np.random.randint(1, 100, N)
num_iterations = 1
for i in range (2):
    total_time = timeit.timeit(lambda: quickSortPar(arr, 0, len(arr)-1), number=num_iterations)

    # Усреднение времени
    average_time = total_time / num_iterations
    print(f"Среднее время выполнения: {average_time} секунд")

arr = np.random.randint(1, 100, N)
for i in range (2):
    total_time = timeit.timeit(lambda: quickSortSeq(arr, 0, len(arr)-1), number=num_iterations)

    # Усреднение времени
    average_time = total_time / num_iterations
    print(f"Среднее время выполнения: {average_time} секунд")
'''
import random
import time

def createRandomArr(n):
    return np.random.randint(0, 100, n)

N = 10000
ITERS = 100


def quickSortSeq2(arr, low, high):
    if (low < high):
        # partitionHoar
        i = low
        j = high
        pivot = arr[(i + j) // 2]
        while (1):
            while (arr[i] < pivot):
                i = i + 1
            while (arr[j] > pivot):
                j = j - 1

            if (i >= j):
                break

            arr[i], arr[j] = arr[j], arr[i]
            i = i + 1
            j = j - 1

        # recurcive call
        pi = j
        quickSortSeq2(arr, low, pi)
        quickSortSeq2(arr, pi + 1, high)

def calcSeqTime():
    full_time = 0
    for it in range(ITERS):
        arr = createRandomArr(N)
        # print(arr)

        start_time = time.time()

        quickSortSeq(arr, 0, len(arr) - 1)

        end_time = time.time()
        elapsed_time = end_time - start_time;
        # print(elapsed_time)
        full_time += elapsed_time

        # print(arr)
        if (not isArraySorted(arr)):
            print("Massive is not sorted\n")

    full_time /= ITERS
    print("Time taken (sequential):", full_time, "seconds")

calcSeqTime()

def calcParTime():
    full_time = 0
    for it in range(ITERS):
        arr = createRandomArr(N)
        # print(arr)

        start_time = time.time()

        quickSortPar(arr, 0, len(arr) - 1)

        end_time = time.time()
        elapsed_time = end_time - start_time;
        # print(elapsed_time)
        full_time += elapsed_time

        # print(arr)
        if (not isArraySorted(arr)):
            print("Massive is not sorted\n")

    full_time /= ITERS
    print("Time taken (sequential):", full_time, "seconds")

calcParTime()