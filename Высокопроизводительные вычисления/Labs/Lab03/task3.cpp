#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

#define L 1.0            // Длина стержня
#define K 1.0            // Коэффициент теплопроводности
#define N 10000          // Кол-во точек 10000 25000 50000
#define H (L/N)          // Шаг по пространству
#define TAU 0.0002       // Шаг по времени
#define NX 11            // Количество точек для аналитических вычислений
#define T_FINAL 1e-4     // Время моделирования
#define U0 1.0           // Начальная температура
#define EPS 1e-3         // Точность сравнения
#define ITERS 1e4        // Кол-во итерация для замера времени

enum tags
{
  left,
  right
};

// Аналитическое решение
double analytical_solution(double x, double t)
{
  double sum = 0.0;
  double u0 = U0;

  for (int m = 0; m < 100; m++)
    sum += (1.0 / (2 * m + 1)) * exp(-K * M_PI * M_PI * (2 * m + 1) * (2 * m + 1) * t / (L * L)) * sin(M_PI * (2 * m + 1) * x / L);

  return (4.0 * u0 / M_PI) * sum;
}

// Вычисление по разностной схеме
double calc_scheme(const double& alpha, const double& prev, const double& cur, const double& next)
{
  return cur + alpha * (next - 2 * cur + prev);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  double start, finish, time, time_sum = 0;

  for (int it = 0; it < ITERS; it++)
  {
    if (world_rank == 0)
      start = MPI_Wtime();
    // Общее количество частей
    int world_segments_num = L / H + 1;
    // Кол-во частей для каждого процесса
    int chunk_size = world_segments_num / world_size;
    if (world_rank == world_size - 1)
      chunk_size = world_segments_num - chunk_size * (world_size - 1);
    double dx = H;
    double dt = TAU;
    double alpha = K * dt / (dx * dx);

    // Инициализация локальных данных
    double* local_temperature = (double*)malloc(chunk_size * sizeof(double));
    for (int i = 0; i < chunk_size; i++)
      local_temperature[i] = U0;

    // Применение разностной схемы
    double* temp_buffer = (double*)malloc(chunk_size * sizeof(double));
    for (int t_step = 0; t_step * TAU < T_FINAL; t_step++)
    {
      // Обмен граничными значениями между процессами
      double left = -1, right = -1;
      MPI_Request send_left, send_right;
      MPI_Request recv_left, recv_right;
      if (world_rank > 0)
      {
        MPI_Isend(&local_temperature[0], 1, MPI_DOUBLE, world_rank - 1, tags::right, MPI_COMM_WORLD, &send_left); //send my left to world_rank - 1 right
        MPI_Irecv(&left, 1, MPI_DOUBLE, world_rank - 1, tags::left, MPI_COMM_WORLD, &recv_left); //recv my left from world_rank - 1 right
      }
      if (world_rank < world_size - 1)
      {
        MPI_Isend(&local_temperature[chunk_size - 1], 1, MPI_DOUBLE, world_rank + 1, tags::left, MPI_COMM_WORLD, &send_right); //send my right to world_rank + 1 left
        MPI_Irecv(&right, 1, MPI_DOUBLE, world_rank + 1, tags::right, MPI_COMM_WORLD, &recv_right); //recv my right from world_rank + 1 left
      }

      // Расчет нового распределения температуры внутри локального участка и Обновление значений температуры
      for (int i = 1; i < chunk_size - 1; i++)
        temp_buffer[i] = calc_scheme(alpha, local_temperature[i - 1], local_temperature[i], local_temperature[i + 1]);
      for (int i = 1; i < chunk_size - 1; i++)
        local_temperature[i] = temp_buffer[i];

      if (world_rank > 0)
      {
        MPI_Wait(&recv_left, MPI_STATUS_IGNORE);
        if (left != -1)
          temp_buffer[0] = calc_scheme(alpha, left, local_temperature[0], local_temperature[1]);
        MPI_Wait(&send_left, MPI_STATUS_IGNORE);
        if (left != -1)
          local_temperature[0] = temp_buffer[0];
        else
          local_temperature[0] = 0;
      }
      else
        local_temperature[0] = 0;

      if (world_rank < world_size - 1)
      {
        MPI_Wait(&recv_right, MPI_STATUS_IGNORE);
        if (right != -1)
          temp_buffer[chunk_size - 1] = calc_scheme(alpha, local_temperature[chunk_size - 2], local_temperature[chunk_size - 1], right);
        MPI_Wait(&send_right, MPI_STATUS_IGNORE);
        if (right != -1)
          local_temperature[chunk_size - 1] = temp_buffer[chunk_size - 1];
        else
          local_temperature[chunk_size - 1] = 0;
      }
      else
        local_temperature[chunk_size - 1] = 0;
    }
    free(temp_buffer);

    // Сбор данных на 0 процессе
    // Подготовим массив для принятия данных на процессе с рангом 0
    int* recv_counts = NULL;
    int* displacements = NULL;
    if (world_rank == 0)
    {
      recv_counts = (int*)malloc(world_size * sizeof(int));
      displacements = (int*)malloc(world_size * sizeof(int));

      int displacement = 0;
      for (int i = 0; i < world_size - 1; ++i)
      {
        recv_counts[i] = chunk_size;
        displacements[i] = displacement;
        displacement += recv_counts[i];
      }
      recv_counts[world_size - 1] = world_segments_num - chunk_size * (world_size - 1);
      displacements[world_size - 1] = displacement;
    }
    double* all_temperature = (double*)malloc(world_segments_num * sizeof(double));
    MPI_Gatherv(local_temperature, chunk_size, MPI_DOUBLE, all_temperature, recv_counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
      finish = MPI_Wtime();
      time = finish - start;
      //printf("time %d = %f\n", it + 1, time);
      time_sum += time;
    }

    if (world_rank == 0)
    {
      // Вывод результатов на экран
      /*printf("Numerical Solution:\n");
      for (int i = 0; i < NX; i++)
      {
        double x = i * L / (NX - 1);
        printf("%.2f\t%.4f\n", x, all_temperature[i * world_segments_num / (NX-1)]);
      }*/

      bool eq_flag = true;
      //printf("\nAnalytical Solution:\n");
      for (int i = 0; i < NX; i++)
      {
        double x = i * L / (NX - 1);
        double num_sol = all_temperature[i * world_segments_num / (NX - 1)];
        double anal_sol = analytical_solution(x, T_FINAL);
        //printf("%.2f\t%.4f\n", x, anal_sol);
        double diff = fabs(anal_sol - num_sol);
        if (diff > EPS)
        {
          printf("Numerical Solution != Analytical Solution at x = %.2f and difference = %.4f\n", x, diff);
          eq_flag = false;
          break;
        }
      }
      if (eq_flag)
      {
        //cout << "it = " << it << " Numerical Solution == Analytical Solution" << endl;
      }
      else
      {
        cout << "it = " << it << " Numerical Solution != Analytical Solution" << endl;
        break;
      }
    }
    free(all_temperature);

    free(local_temperature);
  }

  if (world_rank == 0)
  {
    time_sum /= ITERS;
    printf("For %d processes, with N = %d, time = %f secs\n", world_size, N, time_sum);
    printf("(%d, %f)\n", world_size, time_sum);
  }

  MPI_Finalize();

  return 0;
}
