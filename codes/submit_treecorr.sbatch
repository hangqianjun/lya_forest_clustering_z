#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=treecorr

conda activate pymaster

module load evp-patch

srun -n 32 python compute_wtheta_pixels.py
