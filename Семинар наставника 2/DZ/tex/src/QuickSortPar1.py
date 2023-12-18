import numpy as np
from numba import config, njit, prange
import time
import numba

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