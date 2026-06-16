#!/bin/sh
set -e

mpic++ main_mpi.cpp train.cpp guessing.cpp md5.cpp -o main_mpi -O2 -fopenmp -pthread
mpic++ main_mpi_omp.cpp train.cpp guessing.cpp md5.cpp -o main_mpi_omp -O2 -fopenmp -pthread -DGEN_THREADS=4
mpic++ correctness_mpi.cpp train.cpp guessing.cpp md5.cpp -o correctness_mpi -O2 -fopenmp -pthread
