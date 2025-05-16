import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Collect galaxy catalogues for yaw.')
parser.add_argument('-sub', type=str, default="sub_lya_raw", help='What to submit. Valid arguements: sub_lya_raw, sub_lya_true_cont, sub_lya_uncontaminated, sub_lya_baseline (depricated), sub_lya_LyCAN_noSNRc, sub_lya_LyCAN_SNRc, sub_yaw_raw, sub_yaw_true_cont, sub_yaw_uncontaminated, sub_yaw_baseline (depricated), sub_yaw_LyCAN_noSNRc, sub_yaw_LyCAN_SNRc')
args = parser.parse_args()

def write_sbatch(sbatchfile,comm,time='00:15:00',nodes=1,ntasks=1,
                 job_name="test",output="test.out",env="pymaster",run=0):
    fsb=open(sbatchfile,'w')

    fsb.write('#!/bin/bash\n\n')
    
    fsb.write('#SBATCH -C cpu\n')
    fsb.write('#SBATCH --qos=regular\n')
    fsb.write(f'#SBATCH --time={time}\n')
    fsb.write(f'#SBATCH --nodes={nodes}\n')
    fsb.write(f'#SBATCH --ntasks-per-node={ntasks}\n') #node node=1 is intentional
    fsb.write(f'#SBATCH --job-name={job_name}\n')
    fsb.write(f'#SBATCH -o {output}\n\n') #stdout/err
    
    fsb.write('echo \'This job started on: \' `date`\n\n')

    fsb.write(f'conda activate {env}\n\n')
    
    #fsb.write('module load evp-patch\n\n')

    fsb.write(comm)

    fsb.write('echo \'This job ended on: \' `date`\n')
    fsb.close()
    
    sub='sbatch %s'%(sbatchfile)
    print(sub)
    if(run==1):
        os.system(sub)
    return 0


# submit raw lya catalogue:
if args.sub == "sub_lya_raw":
    sbatchfile = "submit_make_lya_catalogue-raw.sbatch"
    time='00:40:00'
    nodes=1
    ntasks=64
    job_name="lya"
    env="pymaster"
    
    for ii in range(1,10):
        comm=f'srun -n 64 python make_lya_catalogue-raw.py -sim_num {ii} -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 64 -run_mode 0 -cat_tag 20bin > deltaf-raw-{ii}.log \n\n'
        output=f"deltaf-raw-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)

if args.sub == "sub_lya_true_cont":
    sbatchfile = "submit_make_lya_catalogue-true_cont.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="pymaster"
    
    for ii in range(1,10):
        comm=f'srun -n 16 python make_lya_catalogue.py -sim_num {ii} -sim_mode 1 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 16 -run_mode 0 -cat_tag 20bin > deltaf-true_cont-{ii}.log \n\n'
        output=f"deltaf-true_cont-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)

if args.sub == "sub_lya_uncontaminated":
    sbatchfile = "submit_make_lya_catalogue-uncontaminated.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="pymaster"
    
    for ii in range(1,10):
        comm=f'srun -n 16 python make_lya_catalogue.py -sim_num {ii} -sim_mode 2 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 16 -run_mode 0 -cat_tag 20bin > deltaf-uncontaminated-{ii}.log \n\n'
        output=f"deltaf-uncontaminated-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)
"""
if args.sub == "sub_lya_baseline":
    sbatchfile = "submit_make_lya_catalogue-baseline.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="pymaster"
    
    for ii in range(1,10):
        comm=f'srun -n 16 python make_lya_catalogue.py -sim_num {ii} -sim_mode 3 -zbins_file /pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 16 -run_mode 0 > deltaf-baseline-{ii}.log \n\n'
        output=f"deltaf-baseline-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)
"""

if args.sub == "sub_lya_LyCAN_noSNRc":
    sbatchfile = "submit_make_lya_catalogue-LyCAN_noSNRc.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=64
    job_name="lya"
    env="pymaster"
    
    for ii in range(1,10):
        comm=f'srun -n 64 python make_lya_catalogue-LyCAN.py -sim_num {ii} -SNRcut 0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 64 -run_mode 0 -cat_tag 20bin > deltaf-LyCAN_noSNRc-{ii}.log \n\n'
        output=f"deltaf-LyCAN_noSNRc-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)

"""
if args.sub == "sub_lya_LyCAN_SNRc":
    sbatchfile = "submit_make_lya_catalogue-LyCAN_SNRc.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=64
    job_name="lya"
    env="pymaster"
    
    for ii in range(1):
        comm=f'srun -n 64 python make_lya_catalogue-LyCAN.py -sim_num {ii} -SNRcut 1 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 64 -run_mode 0 -cat_tag 20bin > deltaf-LyCAN_SNRc-{ii}.log \n\n'
        output=f"deltaf-LyCAN_SNRc-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)
"""

if args.sub == "sub_lya_LyCAN_cont_noSNRc":
    # contaminated LyCAN
    sbatchfile = "submit_make_lya_catalogue-LyCAN_cont_noSNRc.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=64
    job_name="lya"
    env="pymaster"
    
    for ii in range(10):
        comm=f'srun -n 64 python make_lya_catalogue-LyCAN.py -sim_num {ii} -sim_mode_tag LyCAN_cont -SNRcut 0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -nchunks 64 -run_mode 0 -cat_tag 20bin > deltaf-LyCAN_noSNRc-{ii}.log \n\n'
        output=f"deltaf-LyCAN_noSNRc-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)


# ++++++ YAW ++++++

# ++++++ --- YAW baseline --- ++++++

# submit cross-correlation - raw
if args.sub == "sub_yaw_raw":
    sbatchfile = "submit_measure_yaw-20bin.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(1,10):
        comm=f'python measure_yaw-w-random.py -sim_num {ii} -sim_mode 0 -source 2 -deltaf_weight 1 -unk_zcut 1.8 3.0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 0 -ref_tag 20bin -yaw_tag 20bin > lya-yaw-raw-{ii}.log \n\n'
        output=f"lya-yaw-raw-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)

# submit cross-correlation - true_cont
if args.sub == "sub_yaw_true_cont":
    sbatchfile = "submit_measure_yaw-20bin.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(1,10):
        comm=f'python measure_yaw-w-random.py -sim_num {ii} -sim_mode 1 -source 2 -deltaf_weight 2 -unk_zcut 1.8 3.0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 0 -ref_tag 20bin -yaw_tag 20bin > lya-yaw-true_cont-{ii}.log \n\n'
        output=f"lya-yaw-true_cont-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)

# submit cross-correlation - uncontaminated
if args.sub == "sub_yaw_uncontaminated":
    sbatchfile = "submit_measure_yaw-20bin.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(1,10):
        comm=f'python measure_yaw-w-random.py -sim_num {ii} -sim_mode 2 -source 2 -deltaf_weight 2 -unk_zcut 1.8 3.0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 0 -ref_tag 20bin -yaw_tag 20bin > lya-yaw-uncontaminated-{ii}.log \n\n'
        output=f"lya-yaw-uncontaminated-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)
"""     
# submit cross-correlation - baseline
if args.sub == "sub_yaw_baseline":
    print("Depricated")
    sbatchfile = "submit_measure_yaw-20bin.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(0):
        comm=f'python measure_yaw-w-random.py -sim_num {ii} -sim_mode 3 -source 2 -unk_zcut 1.8 3.0 -zbins_file /pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 1 > lya-yaw-baseline-{ii}.log \n\n'
        output=f"lya-yaw-baseline-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=0)
"""
# submit cross-correlation - noSNRcut
if args.sub == "sub_yaw_LyCAN_noSNRc":
    sbatchfile = "submit_measure_yaw-20bin.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(1,10):
        comm=f'python measure_yaw-w-random.py -sim_num {ii} -sim_mode 4 -source 2 -deltaf_weight 2 -unk_zcut 1.8 3.0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 0 -ref_tag 20bin -yaw_tag 20bin > lya-yaw-LyCAN_noSNRc-{ii}.log \n\n'
        output=f"lya-yaw-LyCAN_noSNRc-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)

"""
# submit cross-correlation - SNRcut
if args.sub == "sub_yaw_LyCAN_SNRc":
    sbatchfile = "submit_measure_yaw-20bin.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(1):
        comm=f'python measure_yaw-w-random.py -sim_num {ii} -sim_mode 5 -source 2 -deltaf_weight 2 -unk_zcut 1.8 3.0 -zbins 2 3 40 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 0 > lya-yaw-LyCAN_SNRc-{ii}.log \n\n'
        output=f"lya-yaw-LyCAN_SNRc-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)
"""

# submit cross-correlation - noSNRcut contaminated
if args.sub == "sub_yaw_LyCAN_cont_noSNRc":
    sbatchfile = "submit_measure_yaw.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(1,10):
        comm=f'python measure_yaw-w-random.py -sim_num {ii} -sim_mode 6 -source 2 -deltaf_weight 2 -unk_zcut 1.8 3.0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 0 -ref_tag 20bin -yaw_tag 20bin > lya-yaw-LyCAN_noSNRc-{ii}.log \n\n'
        output=f"lya-yaw-LyCAN_noSNRc-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)

# ++++++ --- YAW SRD --- ++++++

# submit cross-correlation - noSNRcut
if args.sub == "sub_yaw_LyCAN_noSNRc_srd":
    sbatchfile = "submit_measure_yaw-20bin.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(1,10):
        comm=f'python measure_yaw-w-random.py -sim_num {ii} -sim_mode 4 -source 2 -deltaf_weight 2 -unk_zcut 0 3 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -plot 0 -unk_tag SRD_nz -ref_tag 20bin -yaw_tag 20bin-SRD_nz > lya-yaw-LyCAN_noSNRc-{ii}.log \n\n'
        output=f"lya-yaw-LyCAN_noSNRc-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)



# ++++++ --- YAW norand --- ++++++
if args.sub == "sub_yaw_LyCAN_noSNRc_norand":
    sbatchfile = "submit_measure_yaw-20bin.sbatch"
    time='00:30:00'
    nodes=1
    ntasks=16
    job_name="lya"
    env="yaw_env"
    
    for ii in range(1,10):
        comm=f'python measure_yaw-norand.py -sim_num {ii} -sim_mode 4 -source 2 -deltaf_weight 2 -unk_zcut 1.8 3.0 -zbins 2 3 20 -outroot /pscratch/sd/q/qhang/desi-lya/results/ -ref_tag 20bin -yaw_tag 20bin > lya-yaw-LyCAN_noSNRc-{ii}.log \n\n'
        output=f"lya-yaw-LyCAN_noSNRc-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)




