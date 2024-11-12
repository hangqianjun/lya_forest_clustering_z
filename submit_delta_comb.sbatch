#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=treecorr

conda activate pymaster

module load evp-patch

srun -n 32 python comb-delta-laura.py
