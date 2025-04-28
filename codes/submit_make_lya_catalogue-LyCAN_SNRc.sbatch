#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=lya
#SBATCH -o deltaf-LyCAN_SNRc-0.out

echo 'This job started on: ' `date`

conda activate pymaster

module load evp-patch

srun -n 64 python make_lya_catalogue-LyCAN.py -sim_num 0 -SNRcut 1 -zbins 2 3 40 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 64 -run_mode 0 > deltaf-LyCAN_SNRc-0.log 

echo 'This job ended on: ' `date`
