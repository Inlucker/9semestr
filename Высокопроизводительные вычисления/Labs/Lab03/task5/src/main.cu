#include "head.h"

// Аналитическое решение
double analytical_solution(double x, double t)
{
  double sum = 0.0;
  double u0 = U0;

  for (int m = 0; m < 100; m++)
    sum += (1.0 / (2 * m + 1)) * exp(-K * M_PI * M_PI * (2 * m + 1) * (2 * m + 1) * t / (L * L)) * sin(M_PI * (2 * m + 1) * x / L);

  return (4.0 * u0 / M_PI) * sum;
}

int main(int argc, char* argv[])
{
	int world_rank, world_size;	
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);//номер текущего процесса
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);//число процессов
  
  // Общее количество частей
  int world_segments_num = L / H + 1;
  // Кол-во частей для каждого процесса
  int chunk_size = world_segments_num / world_size;
  if (world_rank == world_size - 1)
    chunk_size = world_segments_num - chunk_size * (world_size - 1);
  //std::cout << "This is - " << world_rank << " proccess \n" << "chunk_size = " << chunk_size << "\n";
  double dx = H;
  double dt = TAU;
  double alpha = K * dt / (dx * dx);
  
  // Инициализация локальных данных
  double* local_temperature = (double*)malloc(chunk_size * sizeof(double));
  for (int i = 0; i < chunk_size; i++)
  {
    local_temperature[i] = U0;
  }
  
  //Initial output
  /*{
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
    printf("Initial temps:\n");
    for (int i = 0; i < world_segments_num; i++)
    {
      double x = i * dx;
      printf("%d = %.4f\t%.4f\n", i, x, all_temperature[i]);
    }
    printf("Initial temps LESS:\n");
    for (int i = 0; i < NX; i++)
    {
      double x = i * L / (NX - 1);
      printf("%.2f\t%.4f\n", x, all_temperature[i * (world_segments_num-1) / (NX - 1)]);
    }

  }
  free(all_temperature);
  }*/
  
  // Применение разностной схемы на GPU
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
    
    /*cout << "Rank " << world_rank << " left = " << left << " right = " << right << endl;
    for (int i = 0; i < chunk_size; i++)
      cout << local_temperature[i] << " ";
    cout << endl;*/
    
    gpu2(alpha, local_temperature, chunk_size, left, right);
  }
  
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
      printf("%d = %.2f\t%.4f\n", i * (world_segments_num-1) / (NX - 1), x, all_temperature[i * (world_segments_num-1) / (NX - 1)]);
    }

    bool eq_flag = true;
    printf("\nAnalytical Solution:\n");
    for (int i = 0; i < NX; i++)
    {
      double x = i * L / (NX - 1);
      double num_sol = all_temperature[i * (world_segments_num-1) / (NX - 1)];
      double anal_sol = analytical_solution(x, T_FINAL);
      printf("%d = %.2f\t%.4f\n", i * (world_segments_num-1) / (NX - 1), x, anal_sol);
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