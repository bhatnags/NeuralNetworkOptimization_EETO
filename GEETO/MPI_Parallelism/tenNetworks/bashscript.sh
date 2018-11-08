#!/bin/bash

#SBATCH -n 10
#SBATCH -p compute
#SBATCH -t 4:15:00
#SBATCH -J par


export OMP_NUM_THREADS=1
mpiexec python dnnt.py >> res.txt
