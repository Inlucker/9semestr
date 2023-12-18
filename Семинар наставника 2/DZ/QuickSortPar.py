import numpy as np
from numba import config, njit, prange
import time
import numba

def isArraySorted(arr):
    n = len(arr)
    if (n > 0):
        prev = arr[0]
        for i in range(n):
            if (arr[i] < prev):
                return False
    return True;

def createRandomArr(n):
    return np.random.randint(0, 100, n)

N = 10000
ITERS = 100

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

        for id in prange(2):
            if (id == 0):
                quickSortSeq(arr, low, pi)
            else:
                quickSortSeq(arr, pi + 1, high)

def calcParTime():
    arr = createRandomArr(10)
    quickSortPar(arr, 0, len(arr) - 1)
    threads_n = 8
    while (threads_n > 0):
        numba.set_num_threads(threads_n)
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
        print(f"Time taken ({threads_n} threads):", full_time, "seconds")
        threads_n //= 2

N = 1000000
ITERS = 100

config.THREADING_LAYER = 'omp'
config.NUMBA_DEFAULT_NUM_THREADS = 8

for i in range(3):
    calcParTime()