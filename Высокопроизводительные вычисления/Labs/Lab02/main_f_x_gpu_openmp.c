#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N (1024)

void get_f_x_cpu (float *hA)
{
  #pragma omp target teams distribute parallel for map(from:hA[0:N])
  for (int idx = 0; idx < N; idx ++)
  {
    float x = 2.0f * 3.1415926f * (float) idx / (float) N;
    hA [idx] = sinf(exp(x));
  }
}

int main (int argc, char *argv[])
{
  float *hA;
  hA = (float*) malloc (N * sizeof(float));
  get_f_x_cpu(hA);
  free(hA);
  return 0;
}