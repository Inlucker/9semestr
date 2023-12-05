#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

#define L 1.0            // Длина стержня
#define K 1.0            // Коэффициент теплопроводности
#define H 0.02           // Шаг по пространству
#define TAU 0.0002       // Шаг по времени
#define NX 11            // Количество точек для аналитических вычислений
#define T_FINAL 0.1      // Время моделирования
#define U0 1.0           // Начальная температура
#define EPS 1e-3         // Точность сравнения

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

  // Общее количество частей
  int world_segments_num = L / H + 1;
  // Кол-во частей для каждого процесса
  int chunk_size = world_segments_num / world_size;
  if (world_rank == world_size - 1)
    chunk_size = world_segments_num - chunk_size * (world_size - 1);
  double dx = H;
  double dt = TAU;
  double alpha = K * dt / (dx * dx);

  int initial_temperature = -1;
  // Рассылка начальной температуры
  if (world_rank == 0)
  {
    initial_temperature = 1;
    MPI_Bcast(&initial_temperature, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  else
  {
    initial_temperature;
    MPI_Bcast(&initial_temperature, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  // Инициализация локальных данных
  double* local_temperature = (double*)malloc(chunk_size * sizeof(double));
  for (int i = 0; i < chunk_size; i++)
    local_temperature[i] = initial_temperature;

  // Применение разностной схемы
  double* temp_buffer = (double*)malloc(chunk_size * sizeof(double));
  for (int t_step = 0; t_step * TAU < T_FINAL; t_step++)
  {
    double left = -1, right = -1;
    // Обмен граничными значениями между процессами
    if (world_rank > 0)
    {
      MPI_Send(&local_temperature[0], 1, MPI_DOUBLE, world_rank - 1, tags::right, MPI_COMM_WORLD); //send my left to world_rank - 1 right
      MPI_Recv(&left, 1, MPI_DOUBLE, world_rank - 1, tags::left, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //recv my left from world_rank - 1 right
    }
    if (world_rank < world_size - 1)
    {
      MPI_Send(&local_temperature[chunk_size - 1], 1, MPI_DOUBLE, world_rank + 1, tags::left, MPI_COMM_WORLD); //send my right to world_rank + 1 left
      MPI_Recv(&right, 1, MPI_DOUBLE, world_rank + 1, tags::right, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //recv my right from world_rank + 1 left
    }

    // Расчет нового распределения температуры внутри локального участка
    if (left != -1)
      temp_buffer[0] = calc_scheme(alpha, left, local_temperature[0], local_temperature[1]);
    for (int i = 1; i < chunk_size - 1; i++)
      temp_buffer[i] = calc_scheme(alpha, local_temperature[i - 1], local_temperature[i], local_temperature[i + 1]);
    if (right != -1)
      temp_buffer[chunk_size - 1] = calc_scheme(alpha, local_temperature[chunk_size - 2], local_temperature[chunk_size - 1], right);

    // Обновление значений температуры
    if (left != -1)
      local_temperature[0] = temp_buffer[0];
    else
      local_temperature[0] = 0;
    for (int i = 1; i < chunk_size - 1; i++)
      local_temperature[i] = temp_buffer[i];
    if (right != -1)
      local_temperature[chunk_size - 1] = temp_buffer[chunk_size - 1];
    else
      local_temperature[chunk_size - 1] = 0;
  }
  free(temp_buffer);

  // Сбор итоговых результатов на 0 процессе
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
    // Вывод результатов на экран
    printf("Numerical Solution:\n");
    for (int i = 0; i < NX; i++)
    {
      double x = i * L / (NX - 1);
      printf("%.2f\t%.4f\n", x, all_temperature[i * world_segments_num / (NX - 1)]);
    }

    bool eq_flag = true;
    printf("\nAnalytical Solution:\n");
    for (int i = 0; i < NX; i++)
    {
      double x = i * L / (NX - 1);
      double num_sol = all_temperature[i * world_segments_num / (NX - 1)];
      double anal_sol = analytical_solution(x, T_FINAL);
      printf("%.2f\t%.4f\n", x, anal_sol);
      double diff = fabs(anal_sol - num_sol);
      if (diff > EPS)
      {
        printf("Numerical Solution != Analytical Solution at x = %.2f and difference = %.4f\n", x, diff);
        eq_flag = false;
      }
    }
    if (eq_flag)
      cout << "\nNumerical Solution == Analytical Solution" << endl;
    else
      cout << "\nNumerical Solution != Analytical Solution" << endl;

  }
  free(all_temperature);

  free(local_temperature);

  MPI_Finalize();

  return 0;
}
