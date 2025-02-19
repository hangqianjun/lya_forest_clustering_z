import os
import numpy as np
import healpy as hp
from astropy.io import fits
from orphics import mpi,stats
import argparse
import healpy
from pixell import utils

# here call mpi
comm,rank,my_tasks = mpi.distribute(5)

print(rank, my_tasks)

s = stats.Stats(comm)

for task in my_tasks:
    if task  == my_tasks[0]:
        ones = np.ones(task + 1)*task
    else:
        ones=np.append(ones, np.ones(task + 1)*task)
    
s.get_stacks()
allones = utils.allgather(ones,comm)

fname="mpi_test.txt"
allones = np.array(allones)
np.savetxt(fname, allones)