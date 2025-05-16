# measure n(z)
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

def delete_and_recreate_cache_directory(cache_dir):
    if parallel.on_root():  # if running with MPI, this is only executed on rank 0
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)

njn=64
# here can test a range of scales
theta_min=[5,10,15]
theta_max=[15,30,50]
theta_scaled=None
resolution=None
unit='arcmin'

Nbins=40

outroot = "/pscratch/sd/q/qhang/desi-lya/results/"
sim_num = 0
type_tag = "unknown"
unk_tag = ""
unk_zcut=[1.8,3.0]

if Nbins == 40:
    ref_tag = ""
    yaw_tag = ""
else:
    ref_tag = f"-{Nbins}bin"
    yaw_tag = f"-{Nbins}bin"

sim_mode_tag = "raw"
rand_z_name = "Z"
ref_weight_name = "NPIX"
ref_name = "DELTA_F"

saveroot = outroot + f"run-{sim_num}/"
path_unknown = saveroot + f"catalogue/{type_tag}{unk_tag}-zmin-{unk_zcut[0]}-zmax-{unk_zcut[1]}.fits"
path_reference = saveroot + f"catalogue{ref_tag}/delta-{sim_mode_tag}.fits"
path_unk_rand = "/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-w-z.fits"

zbins = [2,3,Nbins]
edges = np.linspace(float(zbins[0]), float(zbins[1]), int(zbins[2])+1)
zsamp = (edges[1:] + edges[:-1])/2.

# turn on logging to terminal (can change level to "info" or remove this line entirely)
#get_logger(level="info", pretty=True, capture_warnings=True)
PROGRESS = True  # if you want to see a progress bar

# CONFIGURATION
patch_num = njn

# LOADING CATALOGS
CACHE_DIR = saveroot + f"yaw{yaw_tag}/cache_{sim_mode_tag}-zbins/"
print("cache: ", CACHE_DIR)

delete_and_recreate_cache_directory(CACHE_DIR)

# set up the catalogues:
cat_ref_rand = None 

cat_unk_rand = yaw.Catalog.from_file(
    cache_directory=os.path.join(CACHE_DIR, "unk_rand"),
    path=path_unk_rand,
    ra_name="RA",
    dec_name="DEC",
    redshift_name=rand_z_name,
    patch_num=patch_num,
    progress=PROGRESS,
    degrees=True,
)

patch_centers = cat_unk_rand.get_centers()

cat_reference = yaw.Catalog.from_file(
    cache_directory=os.path.join(CACHE_DIR, "reference"),
    path=path_reference,
    ra_name="RA",
    dec_name="DEC",
    redshift_name="Z",
    weight_name=ref_weight_name,
    kappa_name=ref_name,
    patch_centers=patch_centers,
    progress=PROGRESS,
    degrees=True,
)
print("Done loading catalogues")

# load unknown data
fin = fits.open(path_unknown)
unknownz = fin[1].data['Z']
unknownz_bin = np.digitize(unknownz, edges)

# here define the unknown sample for each redshift slice: and measure the cross-correlation, save them:
W_SP = {}
#for ii in range(len(edges)-1):
for ii in range(38):
    print(f"Working on bin {ii}...")
    # here select the catalog:
    ind = unknownz_bin == ii + 1

    dataframe_unknown = {
        'RA': fin[1].data['RA'][ind],
        'DEC': fin[1].data['DEC'][ind],
        'Z': fin[1].data['Z'][ind],
    }
    # turn into pandas dataframe:
    dataframe_unknown = pd.DataFrame.from_dict(dataframe_unknown)
    
    cat_unknown = yaw.Catalog.from_dataframe(
        cache_directory=os.path.join(CACHE_DIR, f"unknown-bin{ii}"),
        dataframe=dataframe_unknown,
        ra_name="RA",
        dec_name="DEC",
        redshift_name="Z",
        #weight_name="weight_column",  # optional
        patch_centers=patch_centers,
        progress=PROGRESS,
        degrees=True,
    )

    # code will generate this number of patch centers from the reference randoms
    config = yaw.Configuration.create(
        rmin=theta_min,  # scalar or list of lower scale cuts
        rmax=theta_max,
        unit=unit,
        rweight=theta_scaled,
        resolution=resolution,
        edges=edges[ii:ii+2],
    )
    
    print("Computing w_sp")
    w_sp = crosscorrelate_scalar(
        config,
        cat_reference,
        cat_unknown,
        unk_rand=cat_unk_rand,
        progress=PROGRESS
    )

    W_SP[ii] = w_sp

w_sp_theta = []
for ii in range(len(theta_min)):
    vec = np.zeros((len(edges)-1, 2))
    #for jj in range(len(edges)-1):
    for jj in range(38):
        vec[jj,0] = W_SP[jj][ii].sample().data
        vec[jj,1] = W_SP[jj][ii].sample().error
    w_sp_theta.append(vec)

for ii in range(len(theta_min)):
    fname = saveroot + f"yaw{yaw_tag}/w_sp-zbins-{sim_mode_tag}-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.txt"
    np.savetxt(fname, np.c_[zsamp, w_sp_theta[ii]])