import numba
import numpy as np
import time

N = 1000000
ITERS = 100


def createRandomArr(n):
    return np.random.randint(0, 100, N)


@numba.njit(parallel=True)
def add_arrays(a, b):
    result = np.empty_like(a)
    for i in numba.prange(len(a)):
        result[i] = a[i] + b[i]
    return result


def calcSeqTime():
    a = createRandomArr(10)
    b = createRandomArr(10)
    result = add_arrays(a, b)

    threads_n = 8
    while (threads_n > 0):
        numba.set_num_threads(threads_n)
        full_time = 0
        for it in range(ITERS):
            a = createRandomArr(N)
            b = createRandomArr(N)

            start_time = time.time()

            add_arrays(a, b)

            end_time = time.time()
            elapsed_time = end_time - start_time;
            # print(elapsed_time)
            full_time += elapsed_time

        full_time /= ITERS
        print(f"Time taken ({threads_n} threads):", full_time, "seconds")
        threads_n //= 2


def main():
    for i in range(3):
        calcSeqTime()


if __name__ == "__main__":
    main()
