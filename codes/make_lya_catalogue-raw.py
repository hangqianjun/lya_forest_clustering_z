"""
This is the script for making the Lya flux catalogue from the Lya skewers (RAW only).
This will grab simulation results, normalize the fluxes, and split the transmission
into user-defined redshift bins. 
Output will be a fits file with RA, DEC, delta_F, flux redshift, flux redshift bin, QSO redshift, and weights.

RAW files have different structure, hence this script. For true_cont and baseline cases, use make_lya_catalogue.py

UPDATE JUL 2025: normalization of mean delta_f over the joint survey footprint
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
parser.add_argument('-zbins', nargs='+', default=[2,3,40], help='Zmin, Zmax, Nbin')
parser.add_argument('-nchunks', type=int, default=64, help='How many chunks to split the data')
parser.add_argument('-zbins_file', type=str, default="", help='Redshift bin edges file, if provided will overwrite zbins.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask.')
parser.add_argument('-outroot', type=str, default="/pscratch/sd/q/qhang/desi-lya/results/", help='Where to save the catalogues.')
parser.add_argument('-run_mode', type=int, default=2, help='0=run chunks, 1=process chunks, 2=debug, runs 0 with 1 chunk, 3=get n(z) for Lya.')
parser.add_argument('-cat_tag', type=str, default="", help="Custom tag added to the catalogue folder to distinguish different settings, such as number of zbins. If using all above default setting, the default 'catalogue/' folder is assumed.")
args = parser.parse_args()

# def save fits file:
def save_catalog_to_fits(fname, data_matrix, overwrite=True):
    c=[]
    dtype_to_fits_type = {'int64': 'K',
                          'float32': 'D',
                          'float64': 'E',
                          '<U6': '20A',
                          'bool': 'bool',
                          '>f4': 'D',
                          '>f8': 'E',
                         }
    
    for ii, keys in enumerate(data_matrix.keys()):
        col=fits.Column(name=keys, array=data_matrix[keys],
                        format=dtype_to_fits_type[str(data_matrix[keys].dtype)])
        c.append(col)
    t = fits.BinTableHDU.from_columns(c)
    t.writeto(fname, overwrite=overwrite)

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

print("Loading pre-computed <delta_F>(lambda) over the mask...")
wave_grid = np.loadtxt("wave.txt")
meandF = np.loadtxt(f"/pscratch/sd/q/qhang/desi-lya/results-newbias/run-{args.sim_num}/catalogue/norm-delta-{sim_mode_tag}.txt")
meandF = np.nan_to_num(meandF)

emit = 1215.67
lambda_rf_min=1040
lambda_rf_max=1200
dodgy_lowz_cut=3600

redshift_grid = (wave_grid-emit)/emit

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
    
    
    if args.zbins_file == "":
        nbin = int(args.zbins[2])
        bin_edges = np.linspace(float(args.zbins[0]), float(args.zbins[1]), nbin+1)
        zbin_centre = (bin_edges[1:] + bin_edges[:-1])/2.
    elif args.zbins_file != "":
        fin = np.loadtxt(args.zbins_file)
        bin_edges = fin[:,0]
        zbin_centre = fin[:-1,1]
        nbin = len(bin_edges)-1

    # compute the meandF in each bin:
    meandF_binned = np.zeros((nbin,2))
    for zz in range(nbin):
        ind = (redshift_grid>=bin_edges[zz])&(redshift_grid<bin_edges[zz+1])
        meandF_binned[zz,:] = np.mean(meandF[ind,:],axis=0)
    
    # here call mpi
    if args.run_mode == 0:
        comm,rank,my_tasks = mpi.distribute(args.nchunks)
        s = stats.Stats(comm)
    elif args.run_mode == 2:
        my_tasks = [37]

    for task in my_tasks:
        
        use_fname_list = fname_chunks[task]

        if args.run_mode == 2:
            use_fname_list = [use_fname_list[0]]
        
        data_holder={
            'RA': np.array([]),
            'DEC': np.array([]),
            'Z': np.array([]),
            'ZQSO': np.array([]),
            'DELTA_F': np.array([]),
            'DELTA_F_WEIGHTED': np.array([]),
            'NPIX': np.array([]),
            'TOTWEIGHTS': np.array([]),
        }
        
        for fname in use_fname_list:
        
            delta_F = fits.open(fname)
            # grab all objects in this file:
            nobj = len(delta_F)-1
            
            if args.run_mode == 2:
                nobj=5

            print(f"{nobj} objects to go through...")
    
            for jj in range(nobj):
    
                wavelength_log = delta_F[jj+1].data['LOGLAM']
                #delta_l = delta_F[jj+1].data['DELTA']
                #weight_l = delta_F[jj+1].data['WEIGHT']
                #cont_l = delta_F[1].data['CONT']
                
                # for each, bin in redshift: 
                wave = 10**wavelength_log
                objred = (wave-emit)/emit
                
                # group into coarse redshift bins
                bin_tag = np.digitize(objred, bin_edges)
    
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
                for kk in range(nbin):
                    useind = (bin_tag == kk+1)*in_forest*sel1
                    if len(objred[useind])>0:

                        num_pix = len(objred[useind])
                        totweights = np.sum(delta_F[jj+1].data['WEIGHT'][useind])

                        # Apply the mean correction
                        # direct average of deltaF
                        deltaF = np.sum(delta_F[jj+1].data['DELTA'][useind])/num_pix
                        deltaF = (1+deltaF)/(1 + meandF_binned[kk,1])-1
 
                        # weighted average of deltaF
                        deltaF_w = np.sum(delta_F[jj+1].data['DELTA'][useind]*delta_F[jj+1].data['WEIGHT'][useind])/totweights
                        deltaF_w = (1+deltaF_w)/(1 + meandF_binned[kk,0])-1
                        
                        data_holder['RA'] = np.append(data_holder['RA'],ra)
                        data_holder['DEC'] = np.append(data_holder['DEC'],dec)
                        data_holder['ZQSO'] = np.append(data_holder['ZQSO'],zqso)
                        data_holder['Z'] = np.append(data_holder['Z'],zbin_centre[kk])
                        data_holder['DELTA_F'] = np.append(data_holder['DELTA_F'],deltaF)
                        data_holder['DELTA_F_WEIGHTED'] = np.append(data_holder['DELTA_F_WEIGHTED'],deltaF_w)
                        data_holder['NPIX'] = np.append(data_holder['NPIX'],num_pix)
                        data_holder['TOTWEIGHTS'] = np.append(data_holder['TOTWEIGHTS'],totweights)
        
        print("Number of objects: ", len(data_holder['RA']))
        if len(data_holder['RA'])>0:
            savename = saveroot + f"delta-{sim_mode_tag}-chunk-{task}.fits"
            save_catalog_to_fits(savename, data_holder, overwrite=True)
            print(f"saved: {savename}")

elif args.run_mode == 1:

    print("Combining chunks...")
    
    keys = ['RA', 'DEC', 'Z', 'ZQSO', 'DELTA_F', 'DELTA_F_WEIGHTED', 'NPIX', 'TOTWEIGHTS']
    
    data_holder = {}
    for key in keys:
        data_holder[key] = np.array([])

    for task in range(args.nchunks):
        # data holder
        fname = saveroot + f"delta-{sim_mode_tag}-chunk-{task}.fits"
        fin = fits.open(fname)
        for key in keys:
            data_holder[key] = np.append(data_holder[key], fin[1].data[key])
        
    print(f"Total number of objects: {len(data_holder[key])}")
    # save    
    savename = saveroot +  f"delta-{sim_mode_tag}.fits"
    save_catalog_to_fits(savename, data_holder)
    print(f"saved: {savename}")

    delete_names = [
        saveroot + f"delta-{sim_mode_tag}-chunk-*.fits",
    ]
    
    for f in delete_names:
        os.system(f"rm {f}")
    
    print(f"Files: {delete_names} have been safely deleted.")


elif args.run_mode == 3:
    """
    Grab the n(z) for the sample in the redshift slices:
    """
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
    
    
    if args.zbins_file == "":
        nbin = int(args.zbins[2])
        bin_edges = np.linspace(float(args.zbins[0]), float(args.zbins[1]), nbin+1)
        zbin_centre = (bin_edges[1:] + bin_edges[:-1])/2.
    elif args.zbins_file != "":
        fin = np.loadtxt(args.zbins_file)
        bin_edges = fin[:,0]
        zbin_centre = fin[:-1,1]
        nbin = len(bin_edges)-1

    # define a wavelength / redshift grid
    wave_grid = np.loadtxt("wave.txt")
    
    # here call mpi
    comm,rank,my_tasks = mpi.distribute(args.nchunks)
    s = stats.Stats(comm)

    for task in my_tasks:
        
        data_holder={'tot_weights':{}, 'tot_nobj':{}}
        for ii in range(nbin):
            data_holder['tot_weights'][ii] = np.zeros(len(wave_grid))
            data_holder['tot_nobj'][ii] = np.zeros(len(wave_grid))

        use_fname_list = fname_chunks[task]
        for fname in use_fname_list:
        
            delta_F = fits.open(fname)
            # grab all objects in this file:
            nobj = len(delta_F)-1
            print(f"{nobj} objects to go through...")
    
            for jj in range(nobj):
    
                wavelength_log = delta_F[jj+1].data['LOGLAM']
                    
                #delta_l = delta_F[jj+1].data['DELTA']
                weight_l = delta_F[jj+1].data['WEIGHT']
                #cont_l = delta_F[1].data['CONT']
                
                # for each, bin in redshift: 
                wave = 10**wavelength_log
                objred = (wave-emit)/emit
                
                # group into coarse redshift bins
                bin_tag = np.digitize(objred, bin_edges)
    
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
                for kk in range(nbin):
                    useind = (bin_tag == kk+1)*in_forest*sel1
                    
                    if len(objred[useind])>0:
                        
                        # turn this to number, as float digits are not the same for both arrays...
                        useind_num = np.arange(len(bin_tag))[useind]
                        ind1 = useind_num[0]
                        ind2 = useind_num[-1]
                        
                        grid_ind = (wave_grid>(wave[ind1]-0.4))&(wave_grid<(wave[ind2]+0.4))

                        data_holder['tot_nobj'][kk][grid_ind] += 1
                        data_holder['tot_weights'][kk][grid_ind] += weight_l[useind]
                 

        out1 = np.copy(wave_grid)
        out2 = np.copy(wave_grid)
        for ii in range(nbin):
            out1 = np.c_[out1, data_holder['tot_nobj'][ii]]
            out2 = np.c_[out2, data_holder['tot_weights'][ii]]
            
        # save
        savename = saveroot + f"nz-tot-nobj-delta-{sim_mode_tag}-chunk-{task}.txt"
        np.savetxt(savename, out1)
        print(f"saved: {savename}")

        savename = saveroot + f"nz-tot-weights-delta-{sim_mode_tag}-chunk-{task}.txt"
        np.savetxt(savename, out2)
        print(f"saved: {savename}")

elif args.run_mode == 4:

    print("Combining chunks...")
    
    nz_tot_nobj = 0
    nz_tot_weights = 0 

    for task in range(args.nchunks):
        # data holder
        fname = saveroot + f"nz-tot-nobj-delta-{sim_mode_tag}-chunk-{task}.txt"
        fin = np.loadtxt(fname)
        if task ==0:
            nz_tot_nobj = fin
        else:
            nz_tot_nobj[:,1:] += fin[:,1:]


        # data holder
        fname = saveroot + f"nz-tot-weights-delta-{sim_mode_tag}-chunk-{task}.txt"
        fin = np.loadtxt(fname)
        if task ==0:
            nz_tot_weights = fin
        else:
            nz_tot_weights[:,1:] += fin[:,1:]
        
    # save    
    savename = saveroot +  f"nz-tot-nobj-delta-{sim_mode_tag}.txt"
    np.savetxt(savename, nz_tot_nobj)
    print(f"saved: {savename}")

    savename = saveroot +  f"nz-tot-weights-delta-{sim_mode_tag}.txt"
    np.savetxt(savename, nz_tot_weights)
    print(f"saved: {savename}")

    delete_names = [
        saveroot + f"nz-tot-nobj-delta-{sim_mode_tag}-chunk-*.txt",
        saveroot + f"nz-tot-weights-delta-{sim_mode_tag}-chunk-*.txt"
    ]

    for f in delete_names:
        os.system(f"rm {f}")
    
    print(f"Files: {delete_names} have been safely deleted.")