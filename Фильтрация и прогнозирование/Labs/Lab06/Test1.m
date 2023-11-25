clear;
clc;
close all;

% Параметры системы
T = 300;            % Время моделирования в секундах
f_c = 365/433;      % Частота вращения Земли в сутках^(-1)
Q = 100;            % Коэффициент затухания

% Определение матриц системы
A = -1/T;           % Матрица состояния
B = 1/T;            % Матрица входа
C = 1;              % Матрица выхода
D = 0;              % Матрица прямой передачи

% Создание объекта пространственно-состояний
sys = ss(A, B, C, D);

% Задание начального угла поворота (в радианах)
initial_angle = 0;

% Задание времени моделирования
t = 0:1:300;

% Генерация входного сигнала (например, угловая скорость)
input_signal = Q * cos(2*pi*f_c*t);

figure
plot(input_signal)

% Симуляция системы
figure
[y, t_sim, x] = lsim(sys, input_signal, t, initial_angle);

% Построение графика угла поворота от времени
figure
plot(t_sim, y);
xlabel('Время (сек)');
ylabel('Угол поворота (радиан)');
title('Моделирование вращения Земли');
