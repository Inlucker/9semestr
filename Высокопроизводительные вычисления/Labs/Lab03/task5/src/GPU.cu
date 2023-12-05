#include <cuda.h>
#include <iostream>
#include <stdio.h>

__global__ void mult(int x, int y, int *res)
{
  *res = x * y;
}

// Вычисление по разностной схеме
__device__ double calc_scheme(double alpha, double prev, double cur, double next)
{
  return cur + alpha * (next - 2 * cur + prev);
}

__global__ void calc(double alpha, double* local_temperature, int chunk_size, double *res, double left, double right)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("res[%d] = %f\n", i, res[i]);
  
  if (i == 0)
  {
    if (left != -1)
      res[i] = calc_scheme(alpha, left, local_temperature[0], local_temperature[1]);
    else
      res[0] = 0;
    
  //printf("res[%d] = %f\n", i, res[i]);
  }
  else if (i == chunk_size - 1)
  {
    if (right != -1)
      res[i] = calc_scheme(alpha, local_temperature[chunk_size - 2], local_temperature[chunk_size - 1], right);
    else
      res[chunk_size - 1] = 0;
    
  //printf("res[%d] = %f\n", i, res[i]);
  }
  else
  {
    res[i] = calc_scheme(alpha, local_temperature[i - 1], local_temperature[i], local_temperature[i + 1]);
    //res[i] = local_temperature[i] + alpha * (local_temperature[i + 1] - 2 * local_temperature[i] + local_temperature[i - 1]);
    //if (i == 1)
    //  printf("%f %f %f %f\n", alpha, local_temperature[i - 1], local_temperature[i], local_temperature[i + 1]);
  }
  //printf("res[%d] = %f\n", i, res[i]);
}

int gpu(int x, int y)
{
  int *dev_res;
  int res = 0;
  cudaMalloc((void**)&dev_res, sizeof(int));
  mult<<<1,1>>>(x, y, dev_res);
  cudaMemcpy(&res, dev_res, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dev_res);

  return res;
}

#include <iostream>

using namespace std;

void gpu2(double alpha, double*& local_temperature, int chunk_size, double left, double right)
{
  double *dev_local_temperature = NULL;
  cudaMalloc((void**)&dev_local_temperature, chunk_size * sizeof(double));
  cudaMemcpy(dev_local_temperature, local_temperature, chunk_size * sizeof(double), cudaMemcpyHostToDevice);
  double *dev_res;
  cudaMalloc((void**)&dev_res, chunk_size * sizeof(double));
  calc<<<1, chunk_size>>>(alpha, dev_local_temperature, chunk_size, dev_res, left, right);
  cudaMemcpy(local_temperature, dev_res, chunk_size * sizeof(double), cudaMemcpyDeviceToHost);
  
  /*for (int i = 0; i < chunk_size; i++)
    cout << local_temperature[i] << " ";
  cout << endl;*/
    
  cudaFree(dev_local_temperature);
  cudaFree(dev_res);
}