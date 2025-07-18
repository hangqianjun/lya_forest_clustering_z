"""
This is the script for making the Lya flux catalogue from the Lya skewers.
This will grab simulation results, normalize the fluxes, and split the transmission
into user-defined redshift bins. 
Output will be a fits file with RA, DEC, delta_F, flux redshift, flux redshift bin, QSO redshift, and weights.

For noisy mocks:
"""
import os
import numpy as np
import healpy as hp
from astropy.io import fits
from orphics import mpi,stats
import argparse
import healpy
#from pixell import utils
from glob import glob


parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
parser.add_argument('-sim_mode', type=int, default=0, help='1=true continuum deltas, 2=uncontaminated, 3=baseline (depricated)')
#parser.add_argument('-sim_root', type=str, default="", help='If provided overwrites the sim_num, load sim from this directory. File structure has to be consistent.')
#parser.add_argument('-zbins', nargs='+', default=[2,3,40], help='Zmin, Zmax, Nbin')
parser.add_argument('-nchunks', type=int, default=1, help='How many chunks to split the data')
#parser.add_argument('-zbins_file', type=str, default="", help='Redshift bin edges file, if provided will overwrite zbins.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-run_mode', type=int, default=0, help='0=run chunks, 1=process chunks, 2=debug, runs 0 with 1 chunk.')
parser.add_argument('-cat_tag', type=str, default="", help="Custom tag added to the catalogue folder to distinguish different settings, such as number of zbins. If using all above default setting, the default 'catalogue/' folder is assumed.")
args = parser.parse_args()

# functions
def save_catalog_to_fits(fname, data_matrix,overwrite=True):
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
    t.writeto(fname,overwrite=overwrite)

if args.sim_mode == 1:
    sim_mode_tag = "true_cont"
elif args.sim_mode == 2:
    sim_mode_tag = "uncontaminated"
elif args.sim_mode == 3:
    sim_mode_tag = "baseline"
    print("Warning: depricated.")

# read in redshift bin somewhere, or save it with the results;
simroot = "/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/lya_mocks/mock_analysis/qq_desi_y5/skewers_desi_footprint.5/"
simroot += f"analysis-{args.sim_num}/jura-0/{sim_mode_tag}/deltas_lya/Delta/"

if args.cat_tag == "":
    saveroot = args.outroot + f"run-{args.sim_num}/catalogue/"
else:
    saveroot = args.outroot + f"run-{args.sim_num}/catalogue-{args.cat_tag}/"

mask = hp.read_map(args.mask)
nside=hp.get_nside(mask)
npix = round(12*nside**2,0)
usepix = np.arange(npix)[mask==1]

# Determine list of fnames to use within the mask:
mask_degrade = hp.ud_grade(mask, 16)
hp.mollview(mask_degrade)
pixels_in_mask = np.arange(12*16**2)[mask_degrade.astype(bool)]

fname_list = glob(simroot + "*.fits.gz", recursive = True)
fname_pix = []
for i in range(len(fname_list)):
    fname_pix.append(int(fname_list[i][(len(simroot) + 6):-8]))
fname_pix = np.array(fname_pix)
fname_ind = np.in1d(fname_pix, pixels_in_mask)
fname_list = np.array(fname_list)[fname_ind]
print("Total files to go through: ", len(fname_list))

emit = 1215.67
lambda_rf_min=1040
lambda_rf_max=1200
dodgy_lowz_cut=3600

wave_grid = np.loadtxt("wave.txt")

if args.run_mode == 0 or args.run_mode == 2:
    
    #fname_list = glob(simroot + "*.fits.gz", recursive = True)
    #print("Total files to go through: ", len(fname_list))
    print(f"Splitting into {args.nchunks} chunks...")
    # here, split the file list to chunks and send to different nodes:
    Nfiles = int(len(fname_list)/args.nchunks)+1
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

        use_fname_list = fname_chunks[task]
    
        data_holder={
            'tot_deltaf': np.zeros(len(wave_grid)),
            'tot_weighted_deltaf': np.zeros(len(wave_grid)),
            'tot_pixel': np.zeros(len(wave_grid)),
            'tot_weights': np.zeros(len(wave_grid)),
        }

        if args.run_mode == 2:
            use_fname_list = [use_fname_list[0]]
            print(use_fname_list)
            
        for mm in range(len(use_fname_list)):
        
            hdu = fits.open(fname_chunks[task][mm])
            wave = hdu[1].data # this is the same for all runs / objects
            ra = hdu[2].data['RA']*180/np.pi
            dec = hdu[2].data['DEC']*180/np.pi
            zqso = hdu[2].data['Z']
            qid= hdu[2].data['LOS_ID']
            delta = hdu[3].data
            
            #cont = hdu[5].data
            objred = (wave-emit)/emit
        
            # grab all objects in this file:
            nobj = len(delta)
            print(f"number of los: {nobj}")

            # select pixels in forest
            lambda_obs_min=lambda_rf_min*(1+zqso)
            lambda_obs_max=lambda_rf_max*(1+zqso)
            in_forest=np.logical_and(wave[None,:] > lambda_obs_min[:,None], wave[None,:] < lambda_obs_max[:,None])
            in_forest *= wave[None,:] > dodgy_lowz_cut
            
            # select objects that lie inside the mask
            pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
            sel1 = np.in1d(pix,usepix)

            #useind = in_forest*sel1[:,None]
            useind = np.ones((hdu[3].data).shape)
            useind = (useind.astype(bool))
            
            delta_0 = np.nan_to_num(hdu[3].data, 0)
            weights_0 = np.nan_to_num(hdu[4].data, 0)

            unit_weights = np.zeros(weights_0.shape)
            unit_weights[weights_0>0] = 1
            
            data_holder['tot_deltaf'] += np.sum(((delta_0*unit_weights)*useind.astype(int)),axis=0)
            data_holder['tot_weighted_deltaf'] += np.sum(((delta_0*weights_0)*useind.astype(int)),axis=0)
            data_holder['tot_pixel'] += np.sum((unit_weights*useind.astype(int)),axis=0)
            data_holder['tot_weights'] += np.sum((weights_0*useind.astype(int)),axis=0)
        
        # save
        savename = saveroot + f"norm-delta-{sim_mode_tag}-chunk-{task}.txt"
        out = np.c_[data_holder['tot_deltaf'], data_holder['tot_pixel'], data_holder['tot_weighted_deltaf'], data_holder['tot_weights']]
        np.savetxt(savename, out)
        print(f"saved: {savename}")

elif args.run_mode == 1:

    print("Combining chunks...")

    deltaF = 0
    npix = 0
    deltaFw=0
    nw = 0
    
    for task in range(args.nchunks):
        # data holder
        fname = saveroot + f"norm-delta-{sim_mode_tag}-chunk-{task}.txt"
        fin = np.loadtxt(fname)
        deltaF += fin[:,0]
        npix += fin[:,1]
        deltaFw += fin[:,2]
        nw += fin[:,3]
        
    norm_no_weight = deltaF/npix
    norm_weight = deltaFw/nw
    
    # save    
    savename = saveroot +  f"norm-delta-{sim_mode_tag}.txt"
    out = np.c_[norm_weight,norm_no_weight]
    np.savetxt(savename,out)
    print(f"saved: {savename}")

    delete_names = [
        saveroot + f"norm-delta-{sim_mode_tag}-chunk-*.txt",
    ]
    for f in delete_names:
        os.system(f"rm {f}")
    print(f"Files: {delete_names} have now been safely deleted.")