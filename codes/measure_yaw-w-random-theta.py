"""
Theta split runs, can choose to combine or not, and can choose the weights ~ theta^alpha.
"""

import pylab as pl
import healpy as hp
from astropy.io import fits
import numpy as np

import os
import shutil

import yaw
from yaw.utils import parallel
from yaw.correlation import autocorrelate_scalar, crosscorrelate_scalar
from yaw import autocorrelate
# but need to add it in the __init__
import argparse

parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')

parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
parser.add_argument('-sim_mode', type=int, default=0, help='0=raw, 1=true continuum, 2=uncontaminated (picca), 3=baseline (depricated), 4=LyCAN_noSNRcut, 5=LyCAN_SNRcut (depricated), 6=LyCAN_cont_noSNRcut')
parser.add_argument('-source', type=int, default=2, help='1=QSO; 2=galaxies')
parser.add_argument('-deltaf_weight', type=int, default=2, help='0=no weight (uniform weight), 1=NPIX, 2=TOTWEIGHTS')
parser.add_argument('-unk_zcut', nargs='+', default=[1.8,3.0], help='Redshift cuts in the unknown sample redshift.')
parser.add_argument('-zbins', nargs='+', default=[2,3,40], help='Zmin, Zmax, Nbin')
parser.add_argument('-zbins_file', type=str, default="", help='Redshift bin edges file, if provided will overwrite zbins. It is encouraged to use this option for reproducibility.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
#parser.add_argument('-plot', type=int, default=0, help='0=no plot, 1=produce and save a plot for the results, 2=skip correlation, just plot.')
parser.add_argument('-theta', nargs='+', default=[20,50,15], help="theta min, max, nbins -> log bins.")
parser.add_argument('-run_unknown_corr', type=int, default=0, help='Notice this only needs to be run once for all delta_f modes. 0=automode, searches for the w_pp file in the results directory; if not found, will run unknown auto-correlation; 1=run and overwrite the existing file if any.')
parser.add_argument('-ref_tag', type=str, default="", help="tag for the ref folders; valid tags: None, 20bin")
parser.add_argument('-unk_tag', type=str, default="", help="tag for the unk folders (same tag is used for randoms zcolumn); valid tags: None, low, mid, SRD_nz.")
parser.add_argument('-yaw_tag', type=str, default="", help="tag for naming the yaw folders; used for different yaw settings such as number of redshift bins. Default is given by the default arguments above.")
parser.add_argument('-combtheta', type=int, default=0, help="0=do not combine theta measurements, 1=combine theta measurements")
parser.add_argument('-theta_mask', nargs='+', default=[20,50], help="additional mask applied on the theta bins when combine, min and max values. Ignored if combtheta=0.") 
parser.add_argument('-alpha', nargs='+', default=0, help="theta scaling applied to combine the theta bins. Ignored if combtheta=0")
args = parser.parse_args()

def delete_and_recreate_cache_directory(cache_dir):
    if parallel.on_root():  # if running with MPI, this is only executed on rank 0
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)

njn=64
# here can test a range of scales
#theta_min=[5,10,15] # will last bin changed from 15 to 30
#theta_max=[15,30,50]
thetas_edges = np.logspace(np.log10(float(args.theta[0])),np.log10(float(args.theta[1])),int(args.theta[2])+1)
theta_min = thetas_edges[:-1]
theta_max = thetas_edges[1:]
theta_scaled=None
resolution=None
unit='arcmin'
#cache_tag = ""

if args.sim_mode == 0:
    sim_mode_tag = "raw"
elif args.sim_mode == 1:
    sim_mode_tag = "true_cont"
elif args.sim_mode == 2:
    sim_mode_tag = "uncontaminated"
elif args.sim_mode == 3:
    sim_mode_tag = "baseline"
elif args.sim_mode == 4:
    sim_mode_tag = "LyCAN_noSNRcut"
elif args.sim_mode == 5:
    sim_mode_tag = "LyCAN_SNRcut"
elif args.sim_mode == 6:
    sim_mode_tag = "LyCAN_cont_noSNRcut"
    # contaminated case

if args.source == 1:
    type_tag = "QSO"
elif args.source == 2:
    type_tag = "unknown"

if args.deltaf_weight == 0:
    ref_name = 'DELTA_F'
    ref_weight_name = None
elif args.deltaf_weight == 1:
    ref_name = 'DELTA_F'
    ref_weight_name = 'NPIX'
elif args.deltaf_weight == 2:
    ref_name = 'DELTA_F_WEIGHTED'
    ref_weight_name = 'TOTWEIGHTS'


# make a z_column dictionary for unk_tag:
rand_z_dict={
    "low": "Z_LOW",
    "mid": "Z_MID",
    "SRD_nz": "Z_SRD",
    "LBG_nz": "Z",
    # this one zlim has to be [0, 3]
    # need a rand_z for LBG, maybe just use Z
}

# Update all tags
if args.unk_tag == "":
    unk_tag = args.unk_tag 
    rand_z_name = "Z"
else:
    unk_tag = "-" + args.unk_tag
    rand_z_name =rand_z_dict[args.unk_tag]

if args.ref_tag == "":
    ref_tag = args.ref_tag 
else:
    ref_tag = "-" + args.ref_tag 

if args.yaw_tag == "":
    yaw_tag = args.yaw_tag
else:
    yaw_tag = "-" + args.yaw_tag


saveroot = args.outroot + f"run-{args.sim_num}/"
ref_root = "/pscratch/sd/q/qhang/desi-lya/results/" + f"run-{args.sim_num}/"

path_unknown = saveroot + f"catalogue/{type_tag}{unk_tag}-zmin-{args.unk_zcut[0]}-zmax-{args.unk_zcut[1]}.fits"
path_reference = ref_root + f"catalogue{ref_tag}/delta-{sim_mode_tag}.fits"
path_unk_rand = "/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-w-z.fits"

print("Unknown: ", path_unknown)
print("Reference: ", path_reference)
print("Random: ", path_unk_rand)


if args.zbins_file != "":
    zsampf = np.loadtxt(args.zbins_file)
    edges = zsampf[:,0]
    zsamp = zsampf[:-1,1]
else:
    edges = np.linspace(float(args.zbins[0]), float(args.zbins[1]), int(args.zbins[2])+1)
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

# set up the catalogues:
cat_unknown = yaw.Catalog.from_file(
    cache_directory=os.path.join(CACHE_DIR, "unknown"),
    path=path_unknown,
    ra_name="RA",
    dec_name="DEC",
    redshift_name="Z",
    #weight_name="weight_column",  # optional
    patch_num=patch_num,
    progress=PROGRESS,
    degrees=True,
)
patch_centers = cat_unknown.get_centers()


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

cat_ref_rand = None 
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

# measurements:

print("Computing w_ss")
w_ss= autocorrelate_scalar(
    config,
    cat_reference,
    progress=PROGRESS
) # returns a list, one for each scale, just pick the first here
#   w_ss.to_file("...") -> store correlation pair counts as HDF5 file

#w_ss
for ii in range(int(args.theta[2])):
    cts_pp = w_ss[ii]
    wpp_jk = cts_pp.sample().samples
    if ii == 0:
        out = np.copy(wpp_jk)
    else:
        out = np.c_[out,wpp_jk]
fname = saveroot + f"yaw{yaw_tag}/w_ss-{sim_mode_tag}-thetasplit-min-{round(thetas_edges[0])}-max-{round(thetas_edges[-1])}.txt"
np.savetxt(fname, out)
print("Saved: ", fname)


print("Computing w_sp")
w_sp = crosscorrelate_scalar(
    config,
    cat_reference,
    cat_unknown,
    unk_rand=cat_unk_rand,
    progress=PROGRESS
) # returns a list, one for each scale, just pick the first here
#   w_sp.to_file("...") -> store correlation pair counts as HDF5 file

#w_sp
for ii in range(int(args.theta[2])):
    cts_pp = w_sp[ii]
    wpp_jk = cts_pp.sample().samples
    if ii == 0:
        out = np.copy(wpp_jk)
    else:
        out = np.c_[out,wpp_jk]
fname = saveroot + f"yaw{yaw_tag}/w_sp-{sim_mode_tag}-thetasplit-min-{round(thetas_edges[0])}-max-{round(thetas_edges[-1])}.txt"
np.savetxt(fname, out)
print("Saved: ", fname)

# wpp
wppfname = saveroot + f"yaw{yaw_tag}/w_pp-thetasplit-min-{round(thetas_edges[0])}-max-{round(thetas_edges[-1])}.txt"
if os.path.isfile(wppfname)!=True:
    print("Computing w_pp")
    # also run unknown case:
    w_pp = autocorrelate(
        config,
        cat_unknown,
        random=cat_unk_rand,
        progress=PROGRESS
    )

#w_pp
for ii in range(int(args.theta[2])):
    cts_pp = w_pp[ii]
    wpp_jk = cts_pp.sample().samples
    if ii == 0:
        out = np.copy(wpp_jk)
    else:
        out = np.c_[out,wpp_jk]
np.savetxt(wppfname, out)
print("Saved: ", fname)


# combine the theta measurements if needed:
if args.combtheta == 1:
    # load in the stuff and combine theta:
    # apply theta_mask, loop over alpha to combine the files
    # save with thetacomb tag and respective min and max
    
    def get_w_comb(w_thetasplit, njn, Ntheta, Nbins, thetas, alpha, theta_mask):
        # do the selection:
        mask_min = theta_mask[0]
        mask_max = theta_mask[1]
        selind = np.where((thetas>=mask_min)&(thetas<=mask_max))[0][:-1]
            
        Theta_bincen = (thetas[1:] + thetas[:-1])/2.
        dTheta=np.array([thetas[i+1]-thetas[i] for i in range(Ntheta)])
        
        w_comb_jk = np.zeros((njn,Nbins))
        for jk in range(njn):
            data_to_get = w_thetasplit[jk, :].reshape((Ntheta,Nbins))
            for ii in range(Nbins):
                denom = sum(Theta_bincen**alpha*dTheta)
                w_comb_jk[jk, ii] = sum(data_to_get[selind,ii]*Theta_bincen[selind]**alpha*dTheta[selind])/denom
        mean = np.mean(w_comb_jk, axis=0)
        std = np.std(w_comb_jk, axis=0)*np.sqrt(njn)
        w_comb = np.c_[mean,std]
        w_comb = np.c_[w_comb, w_comb_jk.T]
        
        return w_comb

    mask_min = float(args.theta_mask[0])
    mask_max = float(args.theta_mask[1])
    thetas_edges_masked = thetas[(thetas>=mask_min)&(thetas<=mask_max)]
    
    for alpha in args.alpha:
        print(f"Combining theta bins with alpha={alpha}...")
        # w_pp
        w_thetasplit = np.loadtxt(saveroot + f"yaw{yaw_tag}/w_pp-thetasplit-min-{args.thetas[0]}-max-{args.thetas[1]}.txt")
        wpp_comb = get_w_comb(w_thetasplit, njn, int(args.theta[2]), len(zsamp), thetas_edges, float(alpha), [mask_min,mask_max])
        fname = saveroot + f"yaw{yaw_tag}/w_pp-thetacomb-alpha-{alpha}-min-{thetas_edges_masked[0]}-max-{thetas_edges_masked[-1]}.txt"
        np.savetxt(fname, wpp_comb)
      
        # w_ss
        w_thetasplit = np.loadtxt(saveroot + f"yaw{yaw_tag}/w_ss-{sim_mode_tag}-thetasplit-min-{args.thetas[0]}-max-{args.thetas[1]}.txt")
        wss_comb = get_w_comb(w_thetasplit, njn, int(args.theta[2]), len(zsamp), thetas_edges, float(alpha), [mask_min,mask_max])
        fname = saveroot + f"yaw{yaw_tag}/w_ss-{sim_mode_tag}-thetacomb-alpha-{alpha}-min-{thetas_edges_masked[0]}-max-{thetas_edges_masked[-1]}.txt"
        np.savetxt(fname, wss_comb)

        # w_sp
        w_thetasplit = np.loadtxt(saveroot + f"yaw{yaw_tag}/w_sp-{sim_mode_tag}-thetasplit-min-{args.thetas[0]}-max-{args.thetas[1]}.txt")
        wsp_comb = get_w_comb(w_thetasplit, njn, int(args.theta[2]), len(zsamp), thetas_edges, float(alpha), [mask_min,mask_max])
        fname = saveroot + f"yaw{yaw_tag}/w_sp-{sim_mode_tag}-thetacomb-alpha-{alpha}-min-{thetas_edges_masked[0]}-max-{thetas_edges_masked[-1]}.txt"
        np.savetxt(fname, wsp_comb)
            






    