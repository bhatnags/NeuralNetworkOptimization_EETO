#!/bin/bash

#SBATCH -n 06
#SBATCH -p compute
#SBATCH -t 3:15:00
#SBATCH -J par


export OMP_NUM_THREADS=1
mpiexec python dnnt.py >> res.txt
