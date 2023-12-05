#include <iostream>
#include <mpi.h>
#include <cuda.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "CPU.cpp"
#include "GPU.cu"

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