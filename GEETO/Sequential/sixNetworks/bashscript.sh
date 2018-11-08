#!/bin/bash

#SBATCH -n 1
#SBATCH -p compute
#SBATCH -t 13:00:00
#SBATCH -J test


export OMP_NUM_THREADS=1
python snnt.py


