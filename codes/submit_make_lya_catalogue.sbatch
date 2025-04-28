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

srun -n 16 python make_lya_catalogue.py -sim_num 0 -sim_mode 1 -zbins_file /pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 16 -run_mode 0 > deltaf.log

#srun -n 16 python make_lya_catalogue.py -sim_num 0 -sim_mode 2 -zbins_file /pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 16 -run_mode 0 > deltaf.log