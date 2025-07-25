"""
Sanity checks:
galaxy top-hat x galaxy overall.
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
import argparse

parser = argparse.ArgumentParser(description='Gal cross')
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
#print("theta_min: ", theta_min)
#print("theta_max: ", theta_max)
#print(Ntheta)
#exit()
theta_scaled=None
resolution=None
unit='arcmin'
Nbins=20
alpha=0

# let's meausre the wpp first
# need to update with new bias, for now use old bias as the 
# cross-correlation is off for new bias
outroot = "/pscratch/sd/q/qhang/desi-lya/results-newbias/"
sim_num = args.sim_num
type_tag = "unknown"
unk_tag = "-SRD_nz"
unk_zcut=[0,3]
#ref_tag = ""
ref_tag = f"-{Nbins}bin"
sim_mode_tag = "raw"
cache_tag = ""
yaw_tag = f"-{Nbins}bin-SRD_nz"
#yaw_tag = ""
rand_z_name = "Z_SRD"
ref_name = 'DELTA_F'
ref_weight_name = 'NPIX'

saveroot = outroot + f"run-{sim_num}/"
path_unknown = saveroot + f"catalogue/{type_tag}{unk_tag}-zmin-{unk_zcut[0]}-zmax-{unk_zcut[1]}.fits"
# access the delta files from the old folder
#path_reference = f"/pscratch/sd/q/qhang/desi-lya/results/run-{sim_num}/catalogue{ref_tag}/delta-{sim_mode_tag}.fits"
path_unk_rand = "/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-w-z.fits"

print(path_unknown)
#print(path_reference)
print(path_unk_rand)


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
CACHE_DIR = saveroot + "cache/"
print("cache: ", CACHE_DIR)

delete_and_recreate_cache_directory(CACHE_DIR)

# let's set up a reference cataogue that is a trimed 
# version of the actual catalog, to avoid taking 
# too long to compute

fin = fits.open(path_unknown)
dataframe = {"RA": fin[1].data["RA"][::10],
             "DEC": fin[1].data["DEC"][::10],
             "Z":fin[1].data["Z"][::10],}
dataframe = pd.DataFrame.from_dict(dataframe)


# set up the catalogues:
cat_unknown = yaw.Catalog.from_file(
    cache_directory=os.path.join(CACHE_DIR, "unknown"),
    path=path_unknown,
    ra_name="RA",
    dec_name="DEC",
    #redshift_name="Z",
    #weight_name="weight_column",  # optional
    patch_num=patch_num,
    progress=PROGRESS,
    degrees=True,
    overwrite=True,
)
patch_centers = cat_unknown.get_centers()

cat_unk_rand = yaw.Catalog.from_file(
    cache_directory=os.path.join(CACHE_DIR, "unk_rand"),
    path=path_unk_rand,
    ra_name="RA",
    dec_name="DEC",
    redshift_name=rand_z_name,
    patch_centers=patch_centers,
    progress=PROGRESS,
    degrees=True,
    overwrite=True,
)

# this is a 5% trimmed version
cat_reference = yaw.Catalog.from_dataframe(
    cache_directory=os.path.join(CACHE_DIR, "reference"),
    dataframe=dataframe,
    ra_name="RA",
    dec_name="DEC",
    redshift_name="Z",    
    patch_centers=patch_centers,
    progress=PROGRESS,
    degrees=True,
    overwrite=True,
)

cat_ref_rand = None

print("Done loading catalogues")


print("Computing galaxy w_sp")
# auto is ok, cross gives error on redshift binning
w_sp = crosscorrelate(
    config,
    cat_reference,
    cat_unknown,
    ref_rand = cat_unk_rand,
    unk_rand = None,
    progress=PROGRESS
)

print("Saving Jackknife realizations...")
# let's save the jackknife realizations here, so we can compute other quantities ourselves if we need them
#w_sp
for jj in range(len(theta_range)):
    for ii in range(Ntheta):
        cts_pp = w_sp[ii]
        wpp_jk = cts_pp.sample().samples
        if ii == 0:
            out = np.copy(wpp_jk)
        else:
            out = np.c_[out,wpp_jk]
    fname = saveroot + f"yaw{yaw_tag}/w_sp-gal-{sim_mode_tag}-thetasplit-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt"
    np.savetxt(fname, out)
    print("Saved: ", fname)

"""
#now combine the angles and save the combined version (mean, std, Jackknife):
# let's try three different alpha's to test things: 
def get_w_comb(w_thetasplit, njn, Ntheta, Nbins, thetas, alpha):
    Theta_bincen = (thetas[1:] + thetas[:-1])/2.
    dTheta=np.array([thetas[i+1]-thetas[i] for i in range(Ntheta)])
    
    w_comb_jk = np.zeros((njn,Nbins))
    for jk in range(njn):
        data_to_get = w_thetasplit[jk, :].reshape((Ntheta,Nbins))
        for ii in range(Nbins):
            denom = sum(Theta_bincen**alpha*dTheta)
            w_comb_jk[jk, ii] = sum(data_to_get[:,ii]*Theta_bincen**alpha*dTheta)/denom
    mean = np.mean(w_comb_jk, axis=0)
    std = np.std(w_comb_jk, axis=0)*np.sqrt(njn)
    w_comb = np.c_[mean,std]
    w_comb = np.c_[w_comb, w_comb_jk.T]
    return w_comb

print(f"Combining theta bins with alpha={alpha}...")

for jj in range(len(theta_range)):
    thetas = np.logspace(np.log10(theta_range[jj][0]),np.log10(theta_range[jj][1]),Ntheta+1)
    
    # w_pp
    w_thetasplit = np.loadtxt(saveroot + f"yaw{yaw_tag}/w_pp-thetasplit-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt")
    wpp_comb = get_w_comb(w_thetasplit, njn, Ntheta, Nbins, thetas, alpha)
    fname = saveroot + f"yaw{yaw_tag}/w_pp-thetacomb-alpha-{alpha}-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt"
    np.savetxt(fname, wpp_comb)
  
    # w_ss
    #w_thetasplit = np.loadtxt(saveroot + f"yaw{yaw_tag}/w_ss-{sim_mode_tag}-thetasplit-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt")
    #wss_comb = get_w_comb(w_thetasplit, njn, Ntheta, Nbins, thetas, alpha)
    #fname = saveroot + f"yaw{yaw_tag}/w_ss-{sim_mode_tag}-thetacomb-alpha-{alpha}-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt"
    #np.savetxt(fname, wss_comb)
   
    # load w_ss from old directory
    wss_comb = np.loadtxt(f"/pscratch/sd/q/qhang/desi-lya/results/run-{sim_num}/yaw{yaw_tag}/w_ss-{sim_mode_tag}-thetacomb-alpha-{alpha}-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt")
    
    # w_sp
    w_thetasplit = np.loadtxt(saveroot + f"yaw{yaw_tag}/w_sp-{sim_mode_tag}-thetasplit-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt")
    wsp_comb = get_w_comb(w_thetasplit, njn, Ntheta, Nbins, thetas, alpha)
    fname = saveroot + f"yaw{yaw_tag}/w_sp-{sim_mode_tag}-thetacomb-alpha-{alpha}-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}.txt"
    np.savetxt(fname, wsp_comb)

    # nz estimator:
    deltaz = edges[1]-edges[0]
    zz = (edges[1:] + edges[:-1])/2.
    
    fname = saveroot + f"yaw{yaw_tag}/nz_estimate-{sim_mode_tag}-wpp-thetacomb-alpha-{alpha}-theta-min-{theta_range[jj][0]}-max-{theta_range[jj][1]}" # .dat, .smp, .cov
    # get errorbar:
    samps = wsp_comb[:,2:]/np.sqrt(wss_comb[:,2:]*wpp_comb[:,2:])/deltaz
    np.savetxt(fname+".smp", np.c_[zz, samps])
    
    cov = np.cov(samps)*njn
    np.savetxt(fname+".cov", cov)
    
    std = np.sqrt(np.diag(cov))
    mean = np.mean(samps,axis=1)
    out=np.c_[zz, -mean, std]
    np.savetxt(fname+".dat", out)
    print("saved nz files.")
"""