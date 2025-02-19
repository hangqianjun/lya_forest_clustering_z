#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=yaw
#SBATCH -o lya-yaw.out


conda activate pymaster

module load evp-patch

python measure_yaw.py > lya-yaw.log
