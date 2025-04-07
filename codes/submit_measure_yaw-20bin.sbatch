#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=lya
#SBATCH -o lya-yaw-LyCAN_noSNRc-0.out

echo 'This job started on: ' `date`

conda activate yaw_env

python measure_yaw-w-random.py -sim_num 0 -sim_mode 0 -source 2 -deltaf_weight 1 -unk_zcut 0 3 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 0 -ref_tag 20bin -yaw_tag 20bin-SRD_nz -unk_tag SRD_nz > lya-yaw-LyCAN_noSNRc-0.log 

echo 'This job ended on: ' `date`
