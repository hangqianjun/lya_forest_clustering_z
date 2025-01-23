"""
This is the script for making the photometric galaxy catalogue (and QSO catalogue)
This will grab simulation results, select a sample of photometric galaxies
either in redshift, or by matching a given redshift distribution.
Output will be a fits file with RA, DEC, true redshift, and weights. 
A n(z) file will also be saved for the catalogue.

Run with ENV = pymaster.
"""
import os
import numpy as np
import healpy as hp
from astropy.io import fits
from orphics import mpi,stats
import argparse
import healpy
from pixell import utils


parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
parser.add_argument('-sim_root', type=str, default="", help='If provided overwrites the sim_num, load sim from this directory. File structure has to be consistent.')
parser.add_argument('-source', type=int, default=1, help='1=galaxies; 2=QSO')
parser.add_argument('-zbins', nargs='+', default=[1.8,3], help='Cuts in redshift. Provide bin edges')
parser.add_argument('-target_nz', type=str, default="", help='Directory to target the nz file. If provided, will try to match the n(z) distribution.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
#parser.add_argument('-make_random', type=int, default=0, help="1=make randoms for the catalogue on the same footprint. 0=no randoms will ge generated.")
#parser.add_argument('-Nxrandom', type=float, default=0, help="How many times random to produce, ignored if make_random=0. If multiple bins, will take the larges bin to take ")
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


print("Initializing...")
# load things, set up directories:
mask = hp.read_map(args.mask)
nside=hp.get_nside(mask)
npix = int(12*nside**2)
usepix = np.arange(npix)[mask==1]

if args.sim_root == "":
    simroot = f"/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/test_box_{args.sim_num}/gal_box/results/"
else:
    simroot = args.sim_root

zbins = args.zbins
nzbins = len(zbins)-1
if nzbins == 0:
    print("No cuts on redshift.")
elif nzbins == 1:
    print(f"Selecting sample with redshift range: {zbins[0]},{zbins[1]}.")
    ### for now we will use this mode
elif nzbins > 1:
    print(f"Splitting samples into {nzbins} redshift bins.")
    print("Not implemented!")
    exit()

if args.target_nz != "":
    print("Not implemented")
    exit()

if args.source == 1:
    type_tag = "unknown"
elif args.source == 2:
    type_tag = "QSO"

# here call mpi
comm,rank,my_tasks = mpi.distribute(128)
s = stats.Stats(comm)

galmap = 0
nz = 0
RA = []
DEC = []
Z = []

for ii in my_tasks:
    
    fname = simroot + f"out_srcs_s{args.source}_{ii}.fits"
    f=fits.open(fname)
    
    redshift = f[1].data['Z_COSMO'] + f[1].data['DZ_RSD']
    ra = f[1].data['RA']
    dec = f[1].data['DEC']
    pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    sel1 = np.in1d(pix,usepix)
    
    #for kk in range(nzbins):
    sel = sel1 * (redshift > zbins[kk])&(redshift <= zbins[kk+1])

    n = np.bincount(pix[sel], minlength=npix)
    cc = np.histogram(redshift[sel], bins=80, range=[0,3])
    
    galmap += n
    nz += cc[0]

    RA.append(ra[sel])
    DEC.append(dec[sel])
    Z.append(redshift[sel])
    
# collect:
s.get_stacks()

galmap_all = utils.allgatherv(galmap,comm)
nz_all = utils.allgatherv(nz,comm)

galmap = np.sum(galmap_all)
nz = np.sum(nz_all)

data_holder = {
'RA': utils.allgatherv(RA,comm),
'DEC': utils.allgatherv(DEC,comm),
'Z': utils.allgatherv(Z,comm),
}

# now save:
fname = args.outroot + f"catalogue-{type_tag}-zmin-{zbins[kk]}-zmax-{zbins[kk+1]}.fits"
save_catalog_to_fits(fname, data_holder)

nzout = np.c_[(cc[1][1:] + cc[1][:-1])*0.5, nz]
np.savetxt(saveroot + f"nz-{type_tag}-zmin-{zbins[kk]}-zmax-{zbins[kk+1]}.txt",nzout)

hp.write_map(saveroot + f"map-{type_tag}-zmin-{zbins[kk]}-zmax-{zbins[kk+1]}.fits", galmap, overwrite=True, dtype='int')