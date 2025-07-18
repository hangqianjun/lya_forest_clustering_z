#!/bin/bash

#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=lya
#SBATCH -o lya-make-gal-9.out

echo 'This job started on: ' `date`

conda activate pymaster

srun -n 32 python make_gal_catalogue-newbias.py -source 2 -zcut 1.8 3.0 -outroot /pscratch/sd/q/qhang/desi-lya/results-newbias/ -nchunks 32 -run_mode 1 -sim_num 1 > lya-make-gal-9.log 

echo 'This job ended on: ' `date`
