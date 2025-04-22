"""
This runs some limited cases for yaw without using randoms.
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
parser.add_argument('-sim_mode', type=int, default=0, help='0=raw, 1=true continuum deltas, 2=uncontaminated mocks (baseline)')
parser.add_argument('-source', type=int, default=2, help='1=QSO; 2=galaxies')
parser.add_argument('-deltaf_weight', type=int, default=2, help='0=no weight (uniform weight), 1=NPIX, 2=TOTWEIGHTS')
parser.add_argument('-unk_zcut', nargs='+', default=[1.8,3.0], help='Redshift cuts in the unknown sample redshift.')
parser.add_argument('-zbins', nargs='+', default=[2,3,40], help='Zmin, Zmax, Nbin')
parser.add_argument('-zbins_file', type=str, default="", help='Redshift bin edges file, if provided will overwrite zbins. It is encouraged to use this option for reproducibility.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
#parser.add_argument('-plot', type=int, default=0, help='0=no plot, 1=produce and save a plot for the results, 2=skip correlation, just plot.')
#parser.add_argument('-run_unknown_corr', type=int, default=0, help='Notice this only needs to be run once for all delta_f modes. 0=automode, searches for the w_pp file in the results directory; if not found, will run unknown auto-correlation; 1=run and overwrite the existing file if any.')
parser.add_argument('-ref_tag', type=str, default="", help="tag for the ref folders; valid tags: None, 20bin")
parser.add_argument('-unk_tag', type=str, default="", help="tag for the unk folders (same tag is used for randoms zcolumn); valid tags: None, low, mid, SRD_nz.")
parser.add_argument('-yaw_tag', type=str, default="", help="tag for naming the yaw folders; used for different yaw settings such as number of redshift bins. Default is given by the default arguments above.")
args = parser.parse_args()

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
    "SRD_nz": "Z_SRD", # this one zlim has to be [0, 3]
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

path_unknown = saveroot + f"catalogue/{type_tag}{unk_tag}-zmin-{args.unk_zcut[0]}-zmax-{args.unk_zcut[1]}.fits"
path_reference = saveroot + f"catalogue{ref_tag}/delta-{sim_mode_tag}.fits"
#path_unk_rand = "/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-w-z.fits"

print("Unknown: ", path_unknown)
print("Reference: ", path_reference)
#print("Random: ", path_unk_rand)


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
CACHE_DIR = saveroot + f"yaw{yaw_tag}/cache_{sim_mode_tag}/"
print("cache: ", CACHE_DIR)

#delete_and_recreate_cache_directory(CACHE_DIR)

# set up the catalogues:
"""
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
"""
cat_unknown = yaw.Catalog(cache_directory=os.path.join(CACHE_DIR, "unknown"))
patch_centers = cat_unknown.get_centers()

"""
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
"""
cat_reference = yaw.Catalog(cache_directory=os.path.join(CACHE_DIR, "reference"))

cat_ref_rand = None 

"""
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
"""
cat_unk_rand = None

print("Done loading catalogues")

# measurements:
print("Computing w_sp")
w_sp = crosscorrelate_scalar(
    config,
    cat_reference,
    cat_unknown,
    unk_rand=cat_unk_rand,
    progress=PROGRESS
) # returns a list, one for each scale, just pick the first here
#   w_sp.to_file("...") -> store correlation pair counts as HDF5 file

# save them (if different scales, need to save each file!):
for ii in range(len(theta_min)):
    cts_sp = w_sp[ii]
    cts_sp.to_file(saveroot + f"yaw{yaw_tag}/w_sp-{sim_mode_tag}-norand-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.hdf5")