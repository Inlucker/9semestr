#include <algorithm>
#include <chrono>
#include <utility>

#include <cstdio>
#include <cstdlib>

#include <omp.h>

using namespace std;

constexpr size_t n = 1024*1024*128;
//const int n = 16;

static size_t minParallelSize = 128*1024;
static unsigned maxParallelDepth = 4;


void serialQuicksort(int* mas, size_t first, size_t last) {
    int mid;
    size_t f = first, l = last;
    mid = mas[(f + l) / 2];
    while (1) {
        while (mas[f] < mid) { ++f; };
        while (mas[l] > mid) { --l; };

        if (f >= l) { break; }

        std::swap (mas[f++], mas[l--]);
    }

    size_t part = l;

    if (first < part)
        serialQuicksort(mas, first, part);
    if (part+1 < last)
        serialQuicksort(mas, part+1, last);
}

void quicksort(int* mas, size_t first, size_t last, unsigned d = 0) {
    int mid;
    size_t f = first, l = last;
    mid = mas[(f + l) / 2];
    while (1) {
        while (mas[f] < mid) { ++f; };
        while (mas[l] > mid) { --l; };

        if (f >= l) { break; }

        std::swap (mas[f++], mas[l--]);
    }

    size_t part = l;

    #pragma omp task if (d <maxParallelDepth && (last - first) > minParallelSize)
    {
        if (first < part)
            quicksort(mas, first, part, d+1);
    }
    #pragma omp task if (d <maxParallelDepth && (last - first) > minParallelSize)
    {
        if (part+1 < last)
            quicksort(mas, part+1, last, d+1);
    }
    #pragma omp taskwait
}


int main() {
    int rc = EXIT_SUCCESS;

    int* parallelA = new int[n];
    int*   serialA = new int[n];

    for (size_t i=0; i<n; i++) {
        int r = rand();
        parallelA[i] = r;
        serialA[i]   = r;
    }

    [[maybe_unused]] unsigned num_threads = std::min<unsigned> (
                n / minParallelSize / 2,
                1 << (maxParallelDepth-1) ); 

    auto start_time = std::chrono::steady_clock::now();
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp single
        { quicksort(parallelA, 0, n-1); }
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time = end_time - start_time;

    fprintf(stderr, "Complete parallel qsort in %-8.3lfms\n", time.count ());

    start_time = std::chrono::steady_clock::now();
    serialQuicksort(serialA, 0, n-1);
    end_time = std::chrono::steady_clock::now();
    time = end_time - start_time;
    fprintf(stderr, "Complete   serial qsort in %-8.3lfms\n", time.count ());

    for (size_t i=0; i<n-1; i++) {
        if (parallelA[i]>parallelA[i+1]) {
            rc = EXIT_FAILURE;
            fprintf (stderr, "Array is not sorted A[%zd]=%d (>) A[%zd]=%d\n", 
                     i, parallelA[i], i+1, parallelA[i+1] );
            break;
        }
    }

    delete[] parallelA;
    delete[] serialA;

    return rc;
}