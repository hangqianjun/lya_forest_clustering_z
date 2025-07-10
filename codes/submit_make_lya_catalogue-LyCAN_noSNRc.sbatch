#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=lya
#SBATCH -o deltaf-LyCAN_noSNRc-0.out

echo 'This job started on: ' `date`

conda activate pymaster

srun -n 64 python make_lya_catalogue-LyCAN.py -sim_num 0 -SNRcut 0 -zbins 2 3 10 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 64 -run_mode 0 -cat_tag 10bin > deltaf-LyCAN_noSNRc-0.log 

echo 'This job ended on: ' `date`
