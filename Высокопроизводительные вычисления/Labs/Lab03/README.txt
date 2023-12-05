which mpirun

mpic++ -o MPI_Test MPI_Test.cpp

srun -n 4 ./MPI_Test
mpiexec -np 4 ./MPI_Test


mpic++ -o task1 task1.cpp
mpiexec ./task1


mpic++ -o task2 task2.cpp
mpiexec -np 1 ./task2
mpiexec -np 2 ./task2
mpiexec -np 4 ./task2
mpiexec -np 8 ./task2
mpiexec -np 16 ./task2
mpiexec -np 24 ./task2


mpic++ -o task3 task3.cpp
mpiexec -np 1 ./task3
mpiexec -np 2 ./task3
mpiexec -np 4 ./task3
mpiexec -np 8 ./task3
mpiexec -np 16 ./task3
mpiexec -np 24 ./task3

mpic++ -o compare compare.cpp
mpiexec -np 1 ./compare
mpiexec -np 2 ./compare
mpiexec -np 4 ./compare
mpiexec -np 8 ./compare
mpiexec -np 16 ./compare
mpiexec -np 24 ./compare


mpic++ -o task4 task4.cpp
mpiexec ./task4


module load QuantumEspresso/v6.38_pgi_mkl
g++ -lmkl_rt -o task6 task6.cpp
srun ./task6