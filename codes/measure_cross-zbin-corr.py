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
import argparse

parser = argparse.ArgumentParser(description='Corr Coeff')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
args = parser.parse_args()

def delete_and_recreate_cache_directory(cache_dir):
    if parallel.on_root():  # if running with MPI, this is only executed on rank 0
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)

njn=64
# here can test a range of scales
Ntheta=15
theta_range=[[1,50]]
theta_min = []
theta_max = []
for ii in range(len(theta_range)):
    thetas = np.logspace(np.log10(theta_range[ii][0]),np.log10(theta_range[ii][1]),Ntheta+1)
    for jj in range(Ntheta):
        theta_min.append(thetas[jj])
        theta_max.append(thetas[jj+1])
theta_scaled=None
resolution=None
unit='arcmin'
Nbins=20

outroot = "/pscratch/sd/q/qhang/desi-lya/results-newbias/"
sim_num = args.sim_num
type_tag = "unknown"
unk_tag = "-SRD_nz"
unk_zcut=[0,3]

ref_tag = f"-{Nbins}bin"
yaw_tag = f"-{Nbins}bin-SRD_nz"

cache_tag = ""
sim_mode_tag = "raw"
rand_z_name = "Z"
ref_weight_name = "NPIX"
ref_name = "DELTA_F"

saveroot = outroot + f"run-{sim_num}/"
path_unknown = saveroot + f"catalogue/{type_tag}{unk_tag}-zmin-{unk_zcut[0]}-zmax-{unk_zcut[1]}.fits"
path_reference = f"/pscratch/sd/q/qhang/desi-lya/results/run-{sim_num}/catalogue{ref_tag}/delta-{sim_mode_tag}.fits"
path_unk_rand = "/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-w-z.fits"

#zbins = [2,3,Nbins]
zbins = [1,2,Nbins]
edges = np.linspace(float(zbins[0]), float(zbins[1]), int(zbins[2])+1)
edges = edges[4:]
zsamp = (edges[1:] + edges[:-1])/2.
print(edges)

# turn on logging to terminal (can change level to "info" or remove this line entirely)
#get_logger(level="info", pretty=True, capture_warnings=True)
PROGRESS = True  # if you want to see a progress bar

# CONFIGURATION
patch_num = njn

# LOADING CATALOGS
CACHE_DIR = saveroot + "cache/"
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
    path = path_reference,
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

# code will generate this number of patch centers from the reference randoms
# let's only pick one zbin in cat reference - cross-correlate it with all other unknown bins
config = yaw.Configuration.create(
    rmin=theta_min,  # scalar or list of lower scale cuts
    rmax=theta_max,
    unit=unit,
    rweight=theta_scaled,
    resolution=resolution,
    #edges=edges[:2], # fix the redshift bin
    edges = [2,2.05], # first bin 
)

# load unknown data
fin = fits.open(path_unknown)
unknownz = fin[1].data['Z']
unknownz_bin = np.digitize(unknownz, edges)

# here define the unknown sample for each redshift slice: and measure the cross-correlation, save them:
W_SP = {}

for ii in range(len(edges)-1):
    print(f"Working on bin {ii}...")
    # here select the catalog:
    ind = unknownz_bin == ii + 1

    dataframe_unknown = {
        'RA': fin[1].data['RA'][ind],
        'DEC': fin[1].data['DEC'][ind],
    }
    # turn into pandas dataframe:
    dataframe_unknown = pd.DataFrame.from_dict(dataframe_unknown)
    
    cat_unknown = yaw.Catalog.from_dataframe(
        cache_directory=os.path.join(CACHE_DIR, "unknown"),
        dataframe=dataframe_unknown,
        ra_name="RA",
        dec_name="DEC",
        #weight_name="weight_column",  # optional
        patch_centers=patch_centers,
        progress=PROGRESS,
        degrees=True,
        overwrite=True,
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

# change save method:
for jj in range(len(theta_range)):
    out = np.zeros((njn, len(zsamp)*Ntheta))
    for zz in range(len(zsamp)): 
        for ii in range(Ntheta):
            cts_pp = W_SP[zz][ii]
            wpp_jk = cts_pp.sample().samples
            ind = zz*Ntheta + ii
            out[:,ind] = wpp_jk[:,0]
    #fname = saveroot + f"yaw{yaw_tag}/w_sp-cross-zbin-lya-z1-{sim_mode_tag}-thetasplit-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt"
    fname = saveroot + f"yaw{yaw_tag}/w_sp-cross-zbin-lya-z1-{sim_mode_tag}-thetasplit-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}-z12.txt"
    np.savetxt(fname, out)
    print("Saved: ", fname)

"""
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
"""