"""
This file computes the weighted mean flux for raw delta f (Eq. 7 on the paper)
"""

from pixell import utils
import numpy as np
import healpy as hp
from orphics import mpi,stats
from astropy.io import fits
#import treecorr
from glob import glob
import argparse
import os


parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
#parser.add_argument('-zbins', nargs='+', default=[2,3,40], help='Zmin, Zmax, Nbin')
parser.add_argument('-nchunks', type=int, default=1, help='How many chunks to split the data')
#parser.add_argument('-zbins_file', type=str, default="", help='Redshift bin edges file, if provided will overwrite zbins.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-run_mode', type=int, default=0, help='0=run chunks, 1=process chunks, 2=debug, runs 0 with 1 chunk, 3=get n(z) for Lya.')
parser.add_argument('-cat_tag', type=str, default="", help="Custom tag added to the catalogue folder to distinguish different settings, such as number of zbins. If using all above default setting, the default 'catalogue/' folder is assumed.")
args = parser.parse_args()

sim_mode_tag = 'raw'

print("Running under catalogue tag: ", args.cat_tag)

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

# define a wavelength / redshift grid
wave_grid = np.loadtxt("wave.txt")

if args.run_mode == 0 or args.run_mode == 2:
    
    print(f"Splitting into {args.nchunks} chunks...")
    # here, split the file list to chunks and send to different nodes:
    Nfiles = int(round(len(fname_list)/args.nchunks,0))

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

    for task in my_tasks:
        
        use_fname_list = fname_chunks[task]

        if args.run_mode == 2:
            use_fname_list = [use_fname_list[0]]
        
        data_holder={
            'tot_deltaf': np.zeros(len(wave_grid)),
            'tot_weighted_deltaf': np.zeros(len(wave_grid)),
            'tot_pixel': np.zeros(len(wave_grid)),
            'tot_weights': np.zeros(len(wave_grid)),
        }
        
        for fname in use_fname_list:
        
            delta_F = fits.open(fname)

            # grab all objects in this file:
            nobj = len(delta_F)-1
            print(f"{nobj} objects to go through...")

            #if args.run_mode == 2:
            #    nobj=100
            
            for jj in range(nobj):
    
                wavelength_log = delta_F[jj+1].data['LOGLAM']
                #delta_l = delta_F[jj+1].data['DELTA']
                #weight_l = delta_F[jj+1].data['WEIGHT']
                #cont_l = delta_F[1].data['CONT']
                
                # for each, bin in redshift: 
                wave = 10**wavelength_log
                objred = (wave-emit)/emit
                
                # group into coarse redshift bins
                #bin_tag = np.digitize(objred, bin_edges)
    
                hduh = delta_F[jj+1].header
                ra = hduh['RA']*180/np.pi
                dec = hduh['DEC']*180/np.pi
                zqso = hduh['Z']

                # select objects that lie inside the mask
                pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
                sel1 = np.in1d(pix,usepix)

                # select pixels in forest
                lambda_obs_min=lambda_rf_min*(1+zqso)
                lambda_obs_max=lambda_rf_max*(1+zqso)
                in_forest=np.logical_and(wave > lambda_obs_min, wave < lambda_obs_max)
                in_forest *= wave > dodgy_lowz_cut
    
                # now bin:
                useind = in_forest*sel1
                if len(objred[useind])>0:
                    
                    # need to match wavelength:
                    grid_ind = (wave_grid>(wave[useind][0]-0.4))&(wave_grid<(wave[useind][-1]+0.4))

                    #print(wave_grid[grid_ind][0], wave_grid[grid_ind][-1])
                    #print(wave[useind][0],wave[useind][-1])
                    
                    data_holder['tot_deltaf'][grid_ind] += delta_F[jj+1].data['DELTA'][useind]
                    data_holder['tot_weighted_deltaf'][grid_ind] += delta_F[jj+1].data['DELTA'][useind]*delta_F[jj+1].data['WEIGHT'][useind]
                    data_holder['tot_pixel'][grid_ind] += 1
                    data_holder['tot_weights'][grid_ind] += delta_F[jj+1].data['WEIGHT'][useind]
        
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