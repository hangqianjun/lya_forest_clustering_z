"""
This is the script for making the random catalogue used for 
cross-correlation. Output will be ra, dec for the randoms.

! Need to upgrade to use mpi if needed.

Run with ENV = pymaster.
"""
import os
import numpy as np
import healpy as hp
from astropy.io import fits
#from orphics import mpi,stats
import argparse
import healpy
#from pixell import utils
import lya_utils as lu

parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask. If empty, will generate a cut-out defined by ralim, declim.')
parser.add_argument('-outroot', type=str, default="/pscratch/sd/q/qhang/desi-lya/", help='Where to save the catalogues.')
parser.add_argument('-Nrandom', type=float, default=1.7e8, help="How many randoms to produce.")
parser.add_argument('-ralim', nargs='+', default=[0, 360], help="RA limit in degrees to generate randoms.")
parser.add_argument('-declim', nargs='+', default=[-25, 20], help="DEC limit in degrees to generate randoms.")
parser.add_argument('-zdist', nargs='+', default=["zmin-1.8", "zmin-1.8-low", "zmin-1.8-mid", "srd"], help="file characterising redshift distribution, to assign random redshifts. If empty, will not assign redshifts.")
args = parser.parse_args()


zdist_file_dict={
    "zmin-1.8": "/pscratch/sd/q/qhang/desi-lya/nz-gal-z1.8-3.0-all-boxes.txt", 
    "zmin-1.8-low": "/pscratch/sd/q/qhang/desi-lya/nz-gal-low-z1.8-3.0-box-0.txt", 
    "zmin-1.8-mid": "/pscratch/sd/q/qhang/desi-lya/nz-gal-mid-z1.8-3.0-box-0.txt", 
    "srd": "/pscratch/sd/q/qhang/desi-lya/nz-gal-SRD_nz-z0-3-box-0.txt",
}

zdist_tag_dict={
    "zmin-1.8": "Z", 
    "zmin-1.8-low": "Z_LOW", 
    "zmin-1.8-mid": "Z_MID", 
    "srd": "Z_SRD",
}

# load things, set up directories:
if args.mask != "":
    mask = hp.read_map(args.mask)
    nside=hp.get_nside(mask)
    npix = int(12*nside**2)
    usepix = np.arange(npix)[mask==1]
else:
    mask = None

# generate a bigger random sample:
# make random catalogue
ra_min = (args.ralim[0])/180.*np.pi
ra_max = (args.ralim[1])/180.*np.pi
dec_min = (args.declim[0])/180.*np.pi
dec_max = (args.declim[1])/180.*np.pi
print('ra range = %f .. %f' % (ra_min, ra_max))
print('dec range = %f .. %f' % (dec_min, dec_max))

# compute random samples needed:
if args.mask != "":
    area_lim = -(np.cos(dec_max + np.pi/2.) - np.cos(dec_min + np.pi/2.)) * (ra_max - ra_min)
    area_mask = np.mean(mask)*(np.pi*4)
    N_to_generate = args.Nrandom * area_lim / area_mask
else:
    N_to_generate = args.Nrandom
N_to_generate = int(N_to_generate)

print("N random: ", N_to_generate)

if len(args.zdist) != 0:
     # only limit to z = [2,3] for cross-correlation bins
    # grab the files:
    Z_holder = {}
    zdist = []
    for ii, zd in enumerate(args.zdist):
        fin = np.loadtxt(zdist_file_dict[zd])
        # normalize zdist to probability:
        zdist.append(np.c_[fin[:,0], fin[:,1]/sum(fin[:,1])])
        Z_holder[zdist_tag_dict[zd]]=np.array([])

print("Will generate these redshifts: ", list(Z_holder.keys()))

# split into chunks:
nchunk = 5
N_per_chunk = int(N_to_generate/nchunk)

RA = np.array([])
DEC = np.array([])

for ii in range(nchunk):
    
    tmp_ra = np.random.uniform(ra_min, ra_max, N_per_chunk)
    tmp_sindec = np.random.uniform(np.sin(dec_min), np.sin(dec_max), N_per_chunk)
    tmp_dec = np.arcsin(tmp_sindec)

    # select inside the pixel:
    if args.mask != "":
        pix = hp.ang2pix(nside,
                         np.radians(90 - tmp_dec*180/np.pi),
                         np.radians(tmp_ra*180/np.pi))
        sel = np.isin(pix, usepix)
    
        rand_ra = tmp_ra[sel]*180/np.pi
        rand_dec = tmp_dec[sel]*180/np.pi
    else:
        rand_ra = tmp_ra*180/np.pi
        rand_dec = tmp_dec*180/np.pi

    RA = np.append(RA, rand_ra)
    DEC = np.append(DEC, rand_dec)

    if len(args.zdist) != 0:
        for ii, zd in enumerate(args.zdist):
            rand_z = np.random.choice(zdist[ii][:,0], size=len(rand_ra), replace=True, p=zdist[ii][:,1])
            Z_holder[zdist_tag_dict[zd]]=np.append(Z_holder[zdist_tag_dict[zd]],rand_z)

print("Number of randoms generated: ", len(RA))

data_holder = {
    'RA': RA,
    'DEC': DEC,
}

if len(args.zdist) != 0:
    for zd in args.zdist:
        data_holder[zdist_tag_dict[zd]] = Z_holder[zdist_tag_dict[zd]]
        
fname = args.outroot + "random-catalogue-overlap-w-z.fits"
lu.save_catalog_to_fits(fname, data_holder)