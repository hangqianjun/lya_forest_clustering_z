#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=lya
#SBATCH -o lya-yaw-raw-thetasplit-9.out

echo 'This job started on: ' `date`

conda activate yaw_env

python measure_yaw-w-random-theta.py -sim_num 9 -sim_mode 0 -source 2 -deltaf_weight 1 -unk_zcut 0 3 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results-newbias/ -theta 1 50 15 -unk_tag SRD_nz -ref_tag 20bin -yaw_tag 20bin-SRD_nz -combtheta 0 > lya-yaw-raw-thetasplit-9.log 

echo 'This job ended on: ' `date`
