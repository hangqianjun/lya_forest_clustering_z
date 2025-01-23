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

ones = []
for task in my_tasks:
    ones.append(np.ones(5)*task)

#s.get_stacks()
#allones = utils.allgatherv(ones,comm)
fname="mpi_test.txt"
ones = np.asarray(ones)
np.savetxt(fname, ones)