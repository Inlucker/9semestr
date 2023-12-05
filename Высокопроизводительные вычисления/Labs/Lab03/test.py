import numpy as np
import matplotlib.pyplot as plt

# Параметры
k = 1
h = 0.02
tau = 0.0002
T = 0.1

# Шаги по времени и пространству
num_steps_t = int(T / tau)
num_steps_x = int(1 / h)

# Инициализация массива температур
u = np.zeros((num_steps_x + 1, num_steps_t + 1))

# Установка начальных условий
u[:, 0] = 1.0

# Явная конечно-разностная схема
for n in range(num_steps_t):
    for i in range(1, num_steps_x):
        test1 = u[i, n]
        alpha = (k * tau / h**2)
        test2 = u[i + 1, n]
        test3 = 2 * u[i, n]
        test4 = u[i - 1, n]
        u[i, n + 1] = u[i, n] + (k * tau / h**2) * (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n])

# Отобразим результат
x_values = np.linspace(0, 1, num_steps_x + 1)
for i in range(num_steps_x):
    print(x_values[i], u[i, -1])
plt.plot(x_values, u[:, -1])
plt.title('Распределение температуры на момент времени T = 0.1')
plt.xlabel('Пространственная координата')
plt.ylabel('Температура')
plt.show()
