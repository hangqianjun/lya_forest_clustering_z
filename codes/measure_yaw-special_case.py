"""
This is the special case for testing:
e.g. - number of bins, scales
This is running on one mock. 
Code is meant to be flexible and easy to change, but not good reproducibility checks.
If reasonable, implement into the main code as an option for reproducibility.
"""
import yaw
import os
import shutil
# check
import numpy as np
from astropy.io import fits
import pylab as pl

import healpy as hp

from yaw.correlation import autocorrelate_scalar, crosscorrelate_scalar
from yaw import autocorrelate, crosscorrelate
from yaw.utils import parallel
from yaw.correlation.corrfunc import ScalarCorrFunc

import pandas as pd
import lya_utils as lu

def delete_and_recreate_cache_directory(cache_dir):
    if parallel.on_root():  # if running with MPI, this is only executed on rank 0
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)
        


njn=64
# here can test a range of scales
theta_min=[10]
theta_max=[30]
theta_scaled=None
resolution=None
unit='arcmin'
Nbins=5

# let's meausre the wpp first
outroot = "/pscratch/sd/q/qhang/desi-lya/results-newbias/"
sim_num = 0
type_tag = "unknown"
unk_tag = ""
unk_zcut=[1.8,3.0]
#ref_tag = ""
sim_mode_tag = "raw"
cache_tag = ""
yaw_tag = f"-{Nbins}bin"
#yaw_tag = ""
rand_z_name = "Z"

saveroot = outroot + f"run-{sim_num}/"
path_unknown = saveroot + f"catalogue/{type_tag}{unk_tag}-zmin-{unk_zcut[0]}-zmax-{unk_zcut[1]}.fits"
#path_reference = saveroot + f"catalogue{ref_tag}/delta-{sim_mode_tag}.fits"
path_unk_rand = "/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-w-z.fits"

zbins = [2,3,Nbins]
edges = np.linspace(float(zbins[0]), float(zbins[1]), int(zbins[2])+1)
zsamp = (edges[1:] + edges[:-1])/2.

# turn on logging to terminal (can change level to "info" or remove this line entirely)
#get_logger(level="info", pretty=True, capture_warnings=True)
PROGRESS = True  # if you want to see a progress bar

# CONFIGURATION
patch_num = njn # code will generate this number of patch centers from the reference randoms
config = yaw.Configuration.create(
    rmin=theta_min,  # scalar or list of lower scale cuts
    rmax=theta_max,
    unit=unit,
    rweight=theta_scaled,
    resolution=resolution,
    edges=edges,
)

# LOADING CATALOGS
CACHE_DIR = saveroot + f"yaw{cache_tag}/cache_{sim_mode_tag}/"
print("cache: ", CACHE_DIR)

delete_and_recreate_cache_directory(CACHE_DIR)

if os.path.isdir(os.path.join(CACHE_DIR, "unknown")):
    print("Loading unknown from cache.")
    cat_unknown = yaw.Catalog(cache_directory=os.path.join(CACHE_DIR, "unknown"))
else:
    # set up the catalogues:
    cat_unknown = yaw.Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "unknown"),
        path=path_unknown,
        ra_name="RA",
        dec_name="DEC",
        redshift_name="Z_COSMO",
        #weight_name="weight_column",  # optional
        patch_num=patch_num,
        progress=PROGRESS,
        degrees=True,
    )
patch_centers = cat_unknown.get_centers()

"""
cat_reference = yaw.Catalog.from_file(
    cache_directory=os.path.join(CACHE_DIR, "reference"),
    path=path_reference,
    ra_name="RA",
    dec_name="DEC",
    redshift_name="Z",
    #weight_name=ref_weight_name,
    #kappa_name=ref_name,
    patch_centers=patch_centers,
    progress=PROGRESS,
    degrees=True,
)

cat_ref_rand = None 
"""

if os.path.isdir(os.path.join(CACHE_DIR, "unk_rand")):
    print("Loading random from cache.")
    cat_unk_rand = yaw.Catalog(cache_directory=os.path.join(CACHE_DIR, "unk_rand"))
else:
    cat_unk_rand = yaw.Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "unk_rand"),
        path=path_unk_rand,
        ra_name="RA",
        dec_name="DEC",
        redshift_name=rand_z_name,
        patch_centers=patch_centers,
        progress=PROGRESS,
        degrees=True,
    )

print("Done loading catalogues")

print("Computing w_pp")
w_pp = autocorrelate(
    config,
    cat_unknown,
    cat_unk_rand,
    progress=PROGRESS
)

# save
print("Saving w_pp")
# save them (if different scales, need to save each file!):
for ii in range(len(theta_min)):
    cts_pp = w_pp[ii]
    fname = saveroot + f"yaw{yaw_tag}/w_pp-norsd-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.hdf5"
    cts_pp.to_file(fname)
    print("Saved: ", fname)
