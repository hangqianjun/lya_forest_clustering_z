#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=lya
#SBATCH -o deltaf-uncontaminated-0.out

echo 'This job started on: ' `date`

conda activate pymaster

srun -n 16 python make_lya_catalogue.py -sim_num 0 -sim_mode 2 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 16 -run_mode 0 -cat_tag 20bin > deltaf-uncontaminated-0.log 

echo 'This job ended on: ' `date`
