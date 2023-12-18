from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_thread_num, omp_get_num_threads, omp_set_num_threads, omp_get_max_threads, omp_get_wtime
import numpy as np
from numba import njit
from numba.openmp import openmp_context as openmp
import random

@njit
def quickSortPar(arr, low, high, max_d, d = 0):
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

        if (d < max_d):
            with openmp("task shared(arr)"):
                quickSortPar(arr, low, pi, max_d, d + 1)
            with openmp("task shared(arr)"):
                quickSortPar(arr, pi + 1, high, max_d, d + 1)  
            with openmp("taskwait"):
                return
        else:
            quickSortPar(arr, low, pi, max_d, d + 1)
            quickSortPar(arr, pi + 1, high, max_d, d + 1)

@njit
def quickSortParHelp(arr, max_d):
    with openmp("parallel shared(arr)"):
        with openmp("single"):
            quickSortPar(arr, 0, len(arr)-1, max_d)