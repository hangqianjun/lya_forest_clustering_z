# submit queues for 10 mocks, temporary


import numpy as np
import os

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

if(0):
    sbatchfile = "submit_corr_coeff.sbatch"
    time='00:40:00'
    nodes=1
    ntasks=64
    job_name="lya"
    env="yaw_env"
    
    for ii in range(2,10):
        comm=f'python measure_corr_coeff.py -sim_num {ii} > corr_coeff-{ii}.log \n\n'
        output=f"corr_coeff-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)

if(1):
    sbatchfile = "submit_gal_cross.sbatch"
    time='01:00:00'
    nodes=1
    ntasks=64
    job_name="lya"
    env="yaw_env"
    
    for ii in range(10):
        comm=f'python measure_yaw-gal-cross.py -sim_num {ii} > gal_cross-{ii}.log \n\n'
        output=f"gal_cross-{ii}.out"
        print('Running: ',comm)
        write_sbatch(sbatchfile,comm,time=time,nodes=nodes,ntasks=ntasks,job_name=job_name,output=output,env=env,
                    run=1)
