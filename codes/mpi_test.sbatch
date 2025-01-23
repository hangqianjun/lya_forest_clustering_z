#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --job-name=test

conda activate pymaster

module load evp-patch

srun -n 3 python mpi_test.py
