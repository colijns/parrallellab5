#!/bin/sh
set -e

g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o correctness_omp -O2 -fopenmp -pthread -DVERIFY_MODE=1 -DGEN_MODE=1 -DGEN_THREADS=4 -DGEN_THRESHOLD=2500
g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o correctness_pthread -O2 -fopenmp -pthread -DVERIFY_MODE=2 -DGEN_MODE=2 -DGEN_THREADS=4 -DGEN_THRESHOLD=2500
mpic++ correctness_mpi.cpp train.cpp guessing.cpp md5.cpp -o correctness_mpi -O2 -fopenmp -pthread
