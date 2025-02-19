#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=test
#SBATCH -o deltaf.out

conda activate pymaster

module load evp-patch

srun -n 16 python make_lya_catalogue.py -zbins_file /pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt -outroot /pscratch/sd/q/qhang/desi-lya/delta_F-noise/ -nchunks 16 > deltaf.log
