import timeit
import random

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)

# Генерация случайного массива
random_array = [random.randint(1, 1000) for _ in range(10000)]

# Замер времени выполнения
num_iterations = 100
for i in range (3):
    total_time = timeit.timeit(lambda: quicksort(random_array), number=num_iterations)

    # Усреднение времени
    average_time = total_time / num_iterations
    print(f"Среднее время выполнения: {average_time} секунд")
