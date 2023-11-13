module load nvidia_sdk/nvhpc/23.5
salloc -n 1 --gpus=1 -A proj_1447
cd lab02

module avail
module list
module load nvidia_sdk/nvhpc/23.5

srun -G 2 -A proj_1447 --time=1 -N 1 nvaccelinfo
(Выделение 2(-G) видеокарт на 1 (-N) узле, с максимальным временем задачи 1 минута (--time) в рамках проект 1447 (-A). Запуск nvaccelinfo)

nvcc hello.cu -o hello
srun -N 1 -G 1 -A proj_1447 --time=1 hello
nvcc kernel.cu -o kernel
srun -N 1 -G 1 -A proj_1447 --time=1 kernel
nvcc kernelAsync.cu -o kernelAsync
srun -c 32 -N 1 -G 1 -A proj_1447 --time=1 kernelAsync
nvc -mp=gpu main_f_x_gpu_openmp.c
srun -c 32 -N 1 -G 1 -A proj_1447 --time=1 a.out


nvcc task1.cu -O3 -o task1
srun -N 1 -G 1 -A proj_1447 --time=1 task1
nvcc task2a.cu -O3 -o task2a
srun -N 1 -G 1 -A proj_1447 --time=1 task2a
nvcc task2b.cu -O3 -o task2b
srun -N 1 -G 1 -A proj_1447 --time=1 task2b
nvcc task2c.cu -O3 -o task2c
srun -N 1 -G 1 -A proj_1447 --time=1 task2c
nvcc task2d.cu -O3 -o task2d
srun -N 1 -G 1 -A proj_1447 --time=1 task2d
nvcc task4.cu -O3 -o task4 -lcublas
srun -N 1 -G 1 -A proj_1447 --time=1 task4
nvc -mp=gpu task5.c -O3 -o task5
srun -N 1 -G 1 -A proj_1447 --time=1 task5

