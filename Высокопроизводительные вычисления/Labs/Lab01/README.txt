g++ -fopenmp -o task1 task1.cpp
srun -c 16 --time 1 ./task1
sbatch -c 16 --time 1 --wrap="./task1"
sbatch ./task1.batch

g++ -fopenmp -o task2 task2.cpp
sbatch ./task2.batch

g++ -fopenmp -lopenblas -o task3 task3.cpp
sbatch ./task3.batch

g++ -fopenmp -o task4 task4.cpp
sbatch ./task4.batch


g++ -fopenmp -o lab1 gemm_Пронин_МСМТ231.cpp
srun -c 16 --time 1 ./lab1
export OMP_NUM_THREADS=8
srun -c 16 --time 1 ./lab1
sbatch ./lab1.batch