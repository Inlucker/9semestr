g++ -fopenmp -o task1 task1.cpp
srun -c 16 --time 1 ./task1
sbatch -c 16 --time 1 --wrap="./task1"
sbatch ./task1.batch

g++ -fopenmp -o task2 task2.cpp
sbatch ./task2.batch

module load OpenBlas/v0.3.18
g++ -fopenmp -lopenblas -o task3OpenBlas task3OpenBlas.cpp
sbatch ./task3OpenBlas.batch

module load QuantumEspresso/v6.38_pgi_mkl
g++ -fopenmp -lmkl_rt -o task3MKL task3MKL.cpp
sbatch ./task3MKL.batch

g++ -fopenmp -o task4 task4.cpp
sbatch ./task4.batch


g++ -fopenmp -o lab1 gemm_Пронин_МСМТ231.cpp
srun -c 16 --time 1 ./lab1
export OMP_NUM_THREADS=8
srun -c 16 --time 1 ./lab1
sbatch ./lab1.batch