#include <iostream>
#include <mkl.h>

using namespace std;

// Размер сетки
const int N = 100;
const int M = 100;

// Начальная температура
const double initialTemp = 1.0;
// Граничные условия
const double boundaryValue = 0.0;

// Шаги по времени и пространству
const double dt = 0.001;
const double dx = 0.1;
const double t_final = 0.01;

// Функция для инициализации начальных условий (заполнение сетки единицами)
void initialize(double** grid)
{
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      grid[i][j] = initialTemp;
}

// Функция для численного решения уравнения теплопроводности (пример явной схемы)
void solveHeatEquation(double** grid)
{
  // Численное решение с использованием явной схемы
  int steps = t_final / dt;
  cout << steps << endl;
  for (int k = 0; k < 1000; ++k)
  {
    // Внутренние узлы (исключая границы)
    for (int i = 1; i < N - 1; ++i)
    {
      for (int j = 1; j < M - 1; ++j)
      {
        grid[i][j] += dt * (grid[i + 1][j] - 2 * grid[i][j] + grid[i - 1][j]) / (dx * dx)
          + dt * (grid[i][j + 1] - 2 * grid[i][j] + grid[i][j - 1]) / (dx * dx);
      }
    }

    // Обновление граничных условий (пример: Дирихле)
    for (int i = 0; i < N; ++i)
    {
      grid[i][0] = boundaryValue;  // Левая граница
      grid[i][M - 1] = boundaryValue;  // Правая граница
    }

    for (int j = 0; j < M; ++j)
    {
      grid[0][j] = boundaryValue;  // Верхняя граница
      grid[N - 1][j] = boundaryValue;  // Нижняя граница
    }
  }
}

// Функция для вывода результатов и сравнения с аналитическим решением
void printAndCompareResults(double** grid)
{
  // Вывод значений температуры во всех точках
  std::cout << "Temperature values at all points:\n";
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      printf("Grid[%d][%d]: %.4f\n", i, j, grid[i][j]);
      //std::cout << "Grid[" << i << "][" << j << "]: " << grid[i][j] << "\n";
}

int main()
{
  // Выделение памяти под сетку
  double** grid = new double* [N];
  for (int i = 0; i < N; ++i)
    grid[i] = new double[M];

  // Инициализация начальных условий
  initialize(grid);
  printAndCompareResults(grid);

  // Решение задачи теплопроводности
  solveHeatEquation(grid);

  // Вывод результатов и сравнение с аналитическим решением
  printAndCompareResults(grid);

  // Освобождение памяти
  for (int i = 0; i < N; ++i)
    delete[] grid[i];
  delete[] grid;

  return 0;
}
