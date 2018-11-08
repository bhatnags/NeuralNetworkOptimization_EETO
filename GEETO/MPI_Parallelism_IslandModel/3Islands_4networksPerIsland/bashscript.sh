#!/bin/bash

#SBATCH -n 12
#SBATCH -p compute
#SBATCH -t 10:20:00
#SBATCH -J ipar


export OMP_NUM_THREADS=1
mpiexec python innt.py >> res.txt
