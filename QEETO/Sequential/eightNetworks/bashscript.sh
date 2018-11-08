#!/bin/bash

#SBATCH -n 1
#SBATCH -p compute
#SBATCH -t 10:00:00
#SBATCH -J qtest


export OMP_NUM_THREADS=1
python qnnt.py


