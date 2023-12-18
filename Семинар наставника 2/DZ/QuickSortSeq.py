import numpy as np
from numba import njit
import time

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

@njit
def quickSortSeqNjit(arr, low, high):
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
        quickSortSeqNjit(arr, low, pi)
        quickSortSeqNjit(arr, pi + 1, high)

@njit
def quickSortIterative(arr):
    stack = []
    stack.append((0, len(arr) - 1))

    while stack:
        low, high = stack.pop()

        if low < high:
            # pivot = partition(arr, low, high)
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
            pivot = j

            # using stack
            stack.append((low, pivot))
            stack.append((pivot + 1, high))

N = 100000
ITERS = 100

def calcSeqTime():
    arr = createRandomArr(10)
    quickSortSeq(arr, 0, len(arr) - 1)
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
    print("Time taken (sequential) without @njit:", full_time, "seconds")

def calcSeqTimeNjit():
    arr = createRandomArr(10)
    quickSortSeqNjit(arr, 0, len(arr) - 1)
    full_time = 0
    for it in range(ITERS):
        arr = createRandomArr(N)
        # print(arr)

        start_time = time.time()

        quickSortSeqNjit(arr, 0, len(arr) - 1)

        end_time = time.time()
        elapsed_time = end_time - start_time;
        # print(elapsed_time)
        full_time += elapsed_time

        # print(arr)
        if (not isArraySorted(arr)):
            print("Massive is not sorted\n")

    full_time /= ITERS
    print("Time taken (sequential) with @njit:", full_time, "seconds")

def calcSeqTimeNjitIterative():
    arr = createRandomArr(10)
    quickSortIterative(arr)
    full_time = 0
    for it in range(ITERS):
        arr = createRandomArr(N)
        # print(arr)

        start_time = time.time()

        quickSortIterative(arr)

        end_time = time.time()
        elapsed_time = end_time - start_time;
        # print(elapsed_time)
        full_time += elapsed_time

        # print(arr)
        if (not isArraySorted(arr)):
            print("Massive is not sorted\n")

    full_time /= ITERS
    print("Time taken (sequential) ITERATIVE with @njit:", full_time, "seconds")

#for i in range(3):
#    calcSeqTime()

for i in range(3):
    calcSeqTimeNjit()

for i in range(3):
    calcSeqTimeNjitIterative()