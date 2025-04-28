#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=lya
#SBATCH -o deltaf-LyCAN_noSNRc-9.out

echo 'This job started on: ' `date`

conda activate pymaster

srun -n 64 python make_lya_catalogue-LyCAN.py -SNRcut 0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 64 -run_mode 1 -cat_tag 20bin -sim_num 1 > deltaf-LyCAN_noSNRc-9.log 

echo 'This job ended on: ' `date`
