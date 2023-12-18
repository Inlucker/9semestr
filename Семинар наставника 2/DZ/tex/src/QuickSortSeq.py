import numpy as np
from numba import njit
import time

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