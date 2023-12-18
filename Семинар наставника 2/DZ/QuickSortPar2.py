from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_thread_num, omp_get_num_threads, omp_set_num_threads, omp_get_max_threads, omp_get_wtime
import numpy as np
from numba import njit
from numba.openmp import openmp_context as openmp
import random

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
    
@njit
def getOmpTime():
    return omp_get_wtime()

@njit        
def printNumThreads():
    with openmp('parallel'):
        ID = omp_get_thread_num()
        if (ID == 0):
            print("omp_get_max_threads() =", omp_get_max_threads())
            print("omp_get_num_threads() =", omp_get_num_threads())
        print("omp_get_thread_num() =", ID)

@njit
def printThreadNum():
    print("thread_num = ", omp_get_thread_num());

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

        #print("omp_get_thread_num() = ", omp_get_thread_num());
        #print(arr)
        # recurcive call
        if (d < max_d):
            #arr1 = None
            #arr2 = None
            #with openmp("parallel sections"):
            with openmp("task shared(arr)"):
                #print("omp_get_thread_num() = ", omp_get_thread_num());
                #print(arr)
                quickSortPar(arr, low, pi, max_d, d + 1)
                #quickSortSeq(arr, low, pi)
            with openmp("task shared(arr)"):   
                #print("omp_get_thread_num() = ", omp_get_thread_num());
                #print(arr)
                quickSortPar(arr, pi + 1, high, max_d, d + 1)  
                #quickSortSeq(arr, pi + 1, high)
            with openmp("taskwait"):
                return
                #res_arr = arr1 + arr2
        else:
            quickSortPar(arr, low, pi, max_d, d + 1)
            quickSortPar(arr, pi + 1, high, max_d, d + 1)
        #quickSortPar(arr, low, pi)
        #quickSortPar(arr, pi + 1, high)

@njit
def quickSortParHelp(arr, max_d):
    with openmp("parallel shared(arr)"):
        #ID = omp_get_thread_num()
        #print("omp_get_thread_num() =", ID)
        with openmp("single"):
            #print("num_threads = ", omp_get_num_threads())
            #ID2 = omp_get_thread_num()
            #print("single omp_get_thread_num() =", ID2)
            quickSortPar(arr, 0, len(arr)-1, max_d)
            #print("arr after quickSortPar=", arr)
            #with openmp("taskwait"):
                #print("arr after quickSortPar=", arr)
                #return arr

printNumThreads()
        
arr = createRandomArr(6)
print(arr)
quickSortParHelp(arr, 1)
print(arr)
if (isArraySorted(arr)):
    print("Massive", arr, "IS sorted\n")
if (not isArraySorted(arr)):
    print("Massive is not sorted\n")

@njit
def setNumThreads(n):
    omp_set_num_threads(n)


def calcParTime2():
    arr = createRandomArr(10)
    #with openmp("parallel"):
    #    with openmp("single"):
    #        quickSortPar(arr, 0, len(arr)-1, 2)
    quickSortParHelp(arr, 2)
    threads_n = 16
    while (threads_n > 0):
        setNumThreads(threads_n)
        full_time = 0
        for it in range(ITERS):
            arr = createRandomArr(N)
            # print(arr)

            max_d = 0
            sum_th_n = 1
            cur_th_n = 1
            while (sum_th_n < threads_n):
                cur_th_n *= 2
                sum_th_n += cur_th_n
                max_d += 1
    
            start_time = getOmpTime()
    
            #quickSortPar(arr, 0, len(arr) - 1)
            #with openmp("parallel"):
            #    with openmp("single"):
            #        quickSortPar(arr, 0, len(arr)-1, max_d)
            quickSortParHelp(arr, max_d)
    
            end_time = getOmpTime()
            elapsed_time = end_time - start_time;
            # print(elapsed_time)
            full_time += elapsed_time
    
            # print(arr)
            #if (not isArraySorted(arr)):
            #    print("Massive is not sorted\n")
            
        full_time /= ITERS
        print(f"Time taken ({threads_n} threads):", full_time, "seconds")
        threads_n //= 2
        
N = 100000
ITERS = 100

for i in range(3):
    calcParTime2()