module load nvidia_sdk/nvhpc/23.5
cd lab02
salloc -n 1 --gpus=1 -A proj_1447

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
nvcc AddArrTestStreams.cu -o AddArr
srun -N 1 -G 1 -A proj_1447 --time=1 AddArr


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
nvcc task3.cu -O3 -o task3
srun -N 1 -G 1 -A proj_1447 --time=1 task3
nvcc task4.cu -O3 -o task4 -lcublas
srun -N 1 -G 1 -A proj_1447 --time=1 task4
nvc -mp=gpu task5.c -O3 -o task5
srun -N 1 -G 1 -A proj_1447 --time=1 task5

PROFILE:
nvcc -lineinfo -arch=sm_70 -o task6 task6.cu
srun -N 1 -G 1 -A proj_1447 --time=1 task6
nsys profile -o results_task6 ./task6

nsys profile --trace=cuda,nvtx --stats=true --output results3_task6 ./task6

nvcc -lineinfo -arch=sm_70 -o task1 task1.cu
nsys profile --trace=cuda,nvtx --stats=true --output results1_task1 ./task1

nsys profile --trace=cuda,nvtx --stats=true --trace-fork-before-exec=true --output results4_task6  ./task6

ncu --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared ./task6


srun nsys nvprof ./cuda 1000 10
srun nsys profile -t cuda ./cuda 1000 10