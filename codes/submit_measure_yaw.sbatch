#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=lya
#SBATCH -o lya-yaw-baseline-9.out

echo 'This job started on: ' `date`

conda activate yaw_env

module load evp-patch

python measure_yaw-w-random.py -sim_num 9 -sim_mode 2 -source 2 -zcut 1.8 3.0 -zbins_file /pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 1 > lya-yaw-baseline-9.log

echo 'This job ended on: ' `date`
