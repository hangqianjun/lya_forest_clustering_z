#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=lya
#SBATCH -o deltaf-true_cont-9.out

echo 'This job started on: ' `date`

conda activate pymaster

srun -n 16 python make_lya_catalogue.py -sim_mode 1 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 16 -run_mode 1 -cat_tag 20bin -sim_num 9 > deltaf-true_cont-9.log 

echo 'This job ended on: ' `date`
