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


parser = argparse.ArgumentParser(description='Collect galaxy catalogues for yaw.')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
#parser.add_argument('-sim_root', type=str, default="", help='If provided overwrites the sim_num, load sim from this directory. File structure has to be consistent.')
parser.add_argument('-source', type=int, default=1, help='1=QSO; 2=galaxies')
parser.add_argument('-zcut', nargs='+', default=[1.8,3], help='Cuts in redshift. Provide bin edges')
parser.add_argument('-target_nz', type=str, default="", help='Directory to target the nz file. If provided, will try to match the n(z) distribution.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-nchunks', type=int, default=1, help='How many chunks to split the data')
#parser.add_argument('-make_random', type=int, default=0, help="1=make randoms for the catalogue on the same footprint. 0=no randoms will ge generated.")
#parser.add_argument('-Nxrandom', type=float, default=0, help="How many times random to produce, ignored if make_random=0. If multiple bins, will take the larges bin to take ")
parser.add_argument('-run_mode', type=int, default=2, help='0=run chunks, 1=process chunks, 2=debug, runs 0 with 1 chunk.')  
args = parser.parse_args()

# functions
def save_catalog_to_fits(fname, data_matrix, overwrite=True):
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
    t.writeto(fname, overwrite=True)


print("Initializing...")
# load things, set up directories:
if args.sim_num == 0:
    simroot = f"/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/test_box_{args.sim_num}/gal_box/results/"
else:
    simroot = f"/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/test_box-{args.sim_num}/results/"
zbins = args.zcut
saveroot = args.outroot + f"run-{args.sim_num}/catalogue/"

if args.target_nz != "":
    print("Not implemented")
    exit()

if args.source == 1:
    type_tag = "QSO"
elif args.source == 2:
    type_tag = "unknown"

if args.run_mode == 0 or args.run_mode == 2:
    
    nzbins = len(zbins)-1
    print(f"Selecting sample with redshift range: {zbins[0]},{zbins[1]}.")

    mask = hp.read_map(args.mask)
    nside=hp.get_nside(mask)
    npix = int(12*nside**2)
    usepix = np.arange(npix)[mask==1]

    Nfiles = int(128/args.nchunks)
    fname_list = np.arange(128)
    fname_chunks = []
    for kk in range(args.nchunks):
        if kk < args.nchunks - 1 :
            fname_chunks.append(fname_list[(kk*Nfiles):((kk+1)*Nfiles)])
        elif kk == args.nchunks - 1 :
            fname_chunks.append(fname_list[(kk*Nfiles):])
        print(f"Chunk {kk} contains {len(fname_chunks[kk])} files.")
        
    # here call mpi
    if args.run_mode == 0:
        comm,rank,my_tasks = mpi.distribute(args.nchunks)
        s = stats.Stats(comm)
    elif args.run_mode == 2:
        my_tasks = [0]
    
    # save the different parts separately
    for task in my_tasks:
    
        galmap = 0
        nz = 0
        RA = np.array([])
        DEC = np.array([])
        Z = np.array([])
    
        for mm in fname_chunks[task]:
        
            fname = simroot + f"out_srcs_s{args.source}_{mm}.fits"
            f=fits.open(fname)
            
            redshift = f[1].data['Z_COSMO'] + f[1].data['DZ_RSD']
            ra = f[1].data['RA']
            dec = f[1].data['DEC']

            pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
            sel1 = np.in1d(pix,usepix)
            
            #for kk in range(nzbins):
            sel = sel1 * ((redshift > float(zbins[0]))&(redshift <= float(zbins[1])))
        
            n = np.bincount(pix[sel], minlength=npix)
            cc = np.histogram(redshift[sel], bins=80, range=[0,3])
            
            galmap += n
            nz += cc[0]
        
            RA=np.append(RA, ra[sel])
            DEC=np.append(DEC, dec[sel])
            Z=np.append(Z, redshift[sel])
        
        data_holder = {
        'RA': RA,
        'DEC': DEC,
        'Z': Z,
        }

        nzout = np.c_[(cc[1][1:] + cc[1][:-1])*0.5, nz]
        
        print("Number of objects in chunk: ", len(data_holder["RA"]))
        
        # now save:
        fname = saveroot + f"{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.fits"
        save_catalog_to_fits(fname, data_holder, overwrite=True)
        np.savetxt(saveroot + f"nz-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.txt",nzout)
        hp.write_map(saveroot + f"galmap-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.fits", galmap, overwrite=True, dtype='int')

elif args.run_mode == 1:
    
    print("Combining chunks...")
    
    keys = ['RA', 'DEC', 'Z']
    
    data_holder = {}
    for key in keys:
        data_holder[key] = np.array([])
    nz = 0
    galmap = 0

    for task in range(args.nchunks):
        # data holder
        fname = saveroot +  f"{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.fits"
        fin = fits.open(fname)
        for key in keys:
            data_holder[key] = np.append(data_holder[key], fin[1].data[key])

        # nz:
        fname = saveroot + f"nz-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.txt"
        fin = np.loadtxt(fname)
        nz += fin[:,1]
        if task == 0:
            zz = fin[:,0]

        # galmap:
        fname = saveroot + f"galmap-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.fits"
        fin = hp.read_map(fname)
        galmap += fin
        
    # save    
    savename = saveroot +  f"{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.fits"
    save_catalog_to_fits(savename, data_holder)
    print(f"saved: {savename}")

    savename = saveroot + f"nz-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.txt"
    np.savetxt(savename, np.c_[zz, nz])
    print(f"saved: {savename}")

    savename = saveroot + f"galmap-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.fits"
    hp.write_map(savename, galmap, overwrite=True, dtype='int')
    print(f"saved: {savename}")

    delete_names = [
        saveroot + f"{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-*.fits",
        saveroot + f"nz-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-*.txt",
        saveroot + f"galmap-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-*.fits",
    ]
    print(f"Files: {delete_names} can now be safely deleted.")