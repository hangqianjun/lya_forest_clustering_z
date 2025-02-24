#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=yaw
#SBATCH -o lya-yaw.out

conda activate pymaster

module load evp-patch

python measure_yaw-w-random.py -sim_num 0 -sim_mode 0 -source 2 -zcut 1.8 3.0 -zbins_file /pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 1> lya-yaw.log
