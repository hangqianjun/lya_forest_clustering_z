import pylab as pl
import healpy as hp
from astropy.io import fits
import numpy as np

import os
import shutil

from yaw import Catalog, Configuration, RedshiftData, autocorrelate, crosscorrelate
from yaw.correlation.measurements import autocorrelate_scalar,  crosscorrelate_scalar
# but need to add it in the __init__
from yaw.utils.logging import get_logger
from yaw.utils import parallel

def delete_and_recreate_cache_directory(cache_dir):
    if parallel.on_root():  # if running with MPI, this is only executed on rank 0
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)
        
        
from yaw.options import Unit

set_params=3
savedir = "/pscratch/sd/q/qhang/desi-lya/yaw/"


if set_params==0:
    njn=64
    theta_min=1
    theta_max=20
    theta_scaled=None
    resolution=None
    unit='arcmin'
    folder = "test-njn-64-noscale-1-20-arcm/"
    
elif set_params==1:
    njn=128
    theta_min=1
    theta_max=20
    theta_scaled=None
    resolution=None
    unit='arcmin'
    folder = "test-njn-128-noscale-1-20-arcm/"
    
        
elif set_params==2:
    njn=64
    theta_min=1
    theta_max=20
    theta_scaled=-1.0
    resolution=10
    unit='arcmin'
    folder = "test-njn-128-noscale-1-20-arcm/"
    
elif set_params==3:
    njn=64
    theta_min=1
    theta_max=10
    theta_scaled=None
    resolution=None
    unit='arcmin'
    folder = "test-njn-64-noscale-1-10-arcm/"

zsampf = np.loadtxt('/pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt')
edges = zsampf[:,0]
zsamp = zsampf[:-1,1]

# turn on logging to terminal (can change level to "info" or remove this line entirely)
get_logger(level="debug", pretty=True, capture_warnings=True)
PROGRESS = True  # if you want to see a progress bar

# CONFIGURATION
patch_num = njn # code will generate this number of patch centers from the reference randoms
config = Configuration.create(
    rmin=theta_min,  # scalar or list of lower scale cuts
    rmax=theta_max,
    unit=unit,
    rweight=theta_scaled,
    resolution=resolution,
    edges=edges,
)

# LOADING CATALOGS
CACHE_DIR = savedir + folder + "cache/"
delete_and_recreate_cache_directory(CACHE_DIR)

# set up the catalogues:

cat_unknown = Catalog.from_file(
    cache_directory=os.path.join(CACHE_DIR, "unknown"),
    path="/pscratch/sd/q/qhang/desi-lya/photometry-catalogue-overlap-zmin-1.8.fits",
    ra_name="RA",
    dec_name="DEC",
    #weight_name="weight_column",  # optional
    patch_num=patch_num,
    progress=PROGRESS,
)
patch_centers = cat_unknown.get_centers()


cat_reference = Catalog.from_file(
    cache_directory=os.path.join(CACHE_DIR, "reference"),
    path="/pscratch/sd/q/qhang/desi-lya/delta-laura-comb-overlap.fits",
    ra_name="RA",
    dec_name="DEC",
    redshift_name="Z",
    #weight_name="weight_column",  # optional
    kappa_name="DELTAF",
    patch_centers=patch_centers,
    progress=PROGRESS,
)

cat_ref_rand = None 
cat_unk_rand = None  # assuming you don't have this


# measurements:
w_ss= autocorrelate_scalar(
    config,
    cat_reference,
    progress=PROGRESS
) # returns a list, one for each scale, just pick the first here
#   w_ss.to_file("...") -> store correlation pair counts as HDF5 file


w_sp = crosscorrelate_scalar(
    config,
    cat_reference,
    cat_unknown,
    unk_rand=cat_unk_rand,
    progress=PROGRESS
) # returns a list, one for each scale, just pick the first here
#   w_sp.to_file("...") -> store correlation pair counts as HDF5 file


# save them:
cts_ss = w_ss[0]
cts_ss.to_file(savedir + folder + "w_ss.hdf5")

cts_sp = w_sp[0]
cts_sp.to_file(savedir + folder + "w_sp.hdf5")
# restored = yaw.CorrFunc.from_file("w_sp.hdf5")


# COMPUTE REDSHIFTS
ncc = RedshiftData.from_corrfuncs(cross_corr=w_sp[0], ref_corr=w_ss[0])  # unk_corr=w_pp
ncc.to_files(savedir + folder + "nz_estimate")  # store as ASCII files with extensions .dat, .smp and .cov