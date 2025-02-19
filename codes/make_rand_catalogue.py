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


parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask. If empty, will generate full sky random.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-Nrandom', type=float, default=0, help="How many randoms to produce.")
parser.add_argument('-ralim', nargs='+', default=[0, 360], help="RA limit in degrees to generate randoms.")
parser.add_argument('-declim', nargs='+', default=[-90, 90], help="DEC limit in degrees to generate randoms.")
args = parser.parse_args()

# functions
def save_catalog_to_fits(fname, data_matrix):
    c=[]
    dtype_to_fits_type = {'int64': 'K',
                          'float64': 'D',
                          'float32': 'E',
                          '<U6': '20A',
                          'bool': 'bool',
                          '>f8': 'D',
                          '>f4': 'E',
                         }
    
    for ii, keys in enumerate(data_matrix.keys()):
        col=fits.Column(name=keys, array=data_matrix[keys],
                        format=dtype_to_fits_type[str(data_matrix[keys].dtype)])
        c.append(col)
    t = fits.BinTableHDU.from_columns(c)
    t.writeto(fname)


# load things, set up directories:
if args.mask != "":
    mask = hp.read_map(args.mask)
    nside=hp.get_nside(mask)
    npix = int(12*nside**2)
    usepix = np.arange(npix)[mask==1]

    # generate a bigger random sample:
    # make random catalogue
    ra_min = (args.ralim[0] - 1)/180.*np.pi
    ra_max = (args.ralim[1] + 1)/180.*np.pi
    dec_min = (args.declim[0] - 1)/180.*np.pi
    dec_max = (args.declim[1] - 1)/180.*np.pi
    print('ra range = %f .. %f' % (ra_min, ra_max))
    print('dec range = %f .. %f' % (dec_min, dec_max))

else:
    mask = None
    ra_min = 0
    ra_max = 360
    dec_min = -90
    dec_max = 90
    print("Generating random for the full sky.")

# compute random samples needed:
if mask != None:
    area_lim = (np.cos(dec_min + np.pi/2) - np.cos(dec_max + np.pi/2)) * (ra_max - ra_min)
    area_mask = np.mean(mask)/np.pi/4.
    N_to_generate = args.Nrandom * area_lim / area_mask
else:
    N_to_generate = args.Nrandom

tmp_ra = np.random.uniform(ra_min, ra_max, N_to_generate)
tmp_sindec = np.random.uniform(np.sin(dec_min), np.sin(dec_max), N_to_generate)
tmp_dec = np.arcsin(tmp_sindec)

# select inside the pixel:
if mask != None: 
    pix = hp.ang2pix(nside,
                     np.radians(90 - tmp_dec*180/np.pi),
                     np.radians(tmp_ra*180/np.pi))
    sel = np.in1d(pix, sepix)

    rand_ra = tmp_ra[sel]*180/np.pi
    rand_dec = tmp_dec[sel]*180/np.pi
else:
    rand_ra = tmp_ra*180/np.pi
    rand_dec = tmp_dec*180/np.pi

print("Number of randoms generated: ", len(rand_ra))

# save the catalogue:
data_holder = {
    'RA': rand_ra,
    'DEC': rand_dec,
}
fname = args.outroot + "randoms.fits"
save_catalog_to_fits(fname, data_holder)