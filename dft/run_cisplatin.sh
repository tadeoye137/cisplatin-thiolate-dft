#!/bin/bash
#$ -N cisplatin
#$ -pe mpi 16
#$ -cwd
#$ -j y
#$ -S /bin/bash

module purge
module load gnu/7.3.1
module load openmpi/4.1.4_gnu7.3.1
module load fftw/fftw-3.3.10

mpirun -np 16 /home/tadeoye/q-e-qe-7.4.1/bin/pw.x < cisplatin.in > cisplatin.out
