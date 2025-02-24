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
parser.add_argument('-sim_mode', type=int, default=0, help='1=true continuum deltas, 2=uncontaminated mocks (baseline)')
#parser.add_argument('-sim_root', type=str, default="", help='If provided overwrites the sim_num, load sim from this directory. File structure has to be consistent.')
parser.add_argument('-zbins', nargs='+', default=[1.8,3], help='Redshift bin edges to compute delta F')
parser.add_argument('-nchunks', type=int, default=1, help='How many chunks to split the data')
parser.add_argument('-zbins_file', type=str, default="/pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt", help='Redshift bin edges file, if provided will overwrite zbins.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-run_mode', type=int, default=0, help='0=run chunks, 1=process chunks, 2=debug, runs 0 with 1 chunk.') 
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

#if args.sim_mode == 0:
#    sim_mode_tag = "raw"
if args.sim_mode == 1:
    sim_mode_tag = "true_cont"
elif args.sim_mode == 2:
    sim_mode_tag = "baseline"

# read in redshift bin somewhere, or save it with the results;
simroot = "/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/lya_mocks/mock_analysis/qq_desi_y5/skewers_desi_footprint.5/"
simroot += f"analysis-{args.sim_num}/jura-0/{sim_mode_tag}/deltas_lya/Delta/"

saveroot = args.outroot + f"run-{args.sim_num}/catalogue/"

mask = hp.read_map(args.mask)
nside=hp.get_nside(mask)
npix = round(12*nside**2,0)
usepix = np.arange(npix)[mask==1]

if args.run_mode == 0 or args.run_mode == 2:
    
    fname_list = glob(simroot + "*.fits.gz", recursive = True)
    print("Total files to go through: ", len(fname_list))
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
    
    emit = 1215.67
    lambda_rf_min=1040
    lambda_rf_max=1200
    
    if args.zbins_file == "":
        bin_edges = args.zbins
        zbin_centre = []
    elif args.zbins_file != "":
        fin = np.loadtxt(args.zbins_file)
        bin_edges = fin[:,0]
        zbin_centre = fin[:-1,1]
    nbin = len(bin_edges)-1
    
    
    # here call mpi
    if args.run_mode == 0:
        comm,rank,my_tasks = mpi.distribute(args.nchunks)
        s = stats.Stats(comm)
    elif args.run_mode == 2:
        my_tasks = [0]
    
    # save the different parts separately
    for task in my_tasks:
    
        RA = np.array([])
        DEC = np.array([])
        Z = np.array([])
        ZQSO = np.array([])
        DELTA_F = np.array([])
        NPIX = np.array([])
        TOTWEIGHTS = np.array([])
    
        for mm in range(len(fname_chunks[task])):
        
            hdu = fits.open(fname_chunks[task][mm])
            wave = hdu[1].data
            ra = hdu[2].data['RA']*180/np.pi
            dec = hdu[2].data['DEC']*180/np.pi
            zqso = hdu[2].data['Z']
            qid= hdu[2].data['LOS_ID']
            delta = hdu[3].data
            weights = hdu[4].data
            #cont = hdu[5].data
            objred = (wave-emit)/emit
        
            # grab all objects in this file:
            nobj = len(delta)
            print(f"number of los: {nobj}")
        
            bin_tag = np.digitize(objred, bin_edges)

            # select pixels in forest
            lambda_obs_min=lambda_rf_min*(1+zqso)
            lambda_obs_max=lambda_rf_max*(1+zqso)
            in_forest=np.logical_and(wave[None,:] > lambda_obs_min[:,None], wave[None,:] < lambda_obs_max[:,None])
            
            # select objects that lie inside the mask
            pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
            sel1 = np.in1d(pix,usepix)
            
            # now bin:
            npix_file = np.array([])
            deltaf_file = np.array([])
            totweights_file = np.array([])
            ra_file = np.array([])
            dec_file = np.array([])
            z_file = np.array([])
            zq_file = np.array([])
            
            for kk in range(nbin):
                if (kk + 1) in bin_tag:
                    useind = (bin_tag == kk+1)[None,:]*in_forest*sel1[:,None]
                    num_pix = np.sum(useind,axis=1)
                
                    # make a mask for objects without any pixel
                    obj_mask = num_pix>0
                    # just in case, should get rid of any nan or inf
                    delta_0 = np.nan_to_num(delta, 0)
                
                    # compute delta_F
                    avg_deltaf = np.sum((delta_0*useind.astype(int))[obj_mask,:],axis=1)
                    avg_deltaf = avg_deltaf/num_pix[obj_mask]
                
                    # compute tot weight
                    weights_0 = np.nan_to_num(weights, 0)
                    tot_weights = np.sum(weights*useind.astype(int),axis=1)
        
                    npix_file = np.append(npix_file, num_pix[obj_mask])
                    deltaf_file = np.append(deltaf_file, avg_deltaf)
                    totweights_file = np.append(totweights_file, tot_weights[obj_mask])
                    ra_file = np.append(ra_file, ra[obj_mask])
                    dec_file = np.append(dec_file, dec[obj_mask])
                    z_file = np.append(z_file, np.ones(len(num_pix[obj_mask]))*zbin_centre[kk])
                    zq_file = np.append(zq_file, zqso[obj_mask])
        
            print(f"Number of delta F in this file is {len(ra_file)}")
            
            RA=np.append(RA, ra_file)
            DEC=np.append(DEC, dec_file)
            Z=np.append(Z, z_file)
            ZQSO=np.append(ZQSO, zq_file)
            DELTA_F=np.append(DELTA_F, deltaf_file)
            NPIX=np.append(NPIX, npix_file)
            TOTWEIGHTS=np.append(TOTWEIGHTS, totweights_file)
        
        data_holder = {
        'RA': RA,
        'DEC': DEC,
        'Z': Z,
        'ZQSO': ZQSO,
        'DELTA_F': DELTA_F,
        'NPIX': NPIX,
        'TOTWEIGHTS': TOTWEIGHTS,
        }
        
        print("Number of objects in chunk: ", len(data_holder["DELTA_F"]))
        # save
        savename = saveroot + f"delta-{sim_mode_tag}-chunk-{task}.fits"
        save_catalog_to_fits(savename, data_holder, overwrite=True)
        print(f"saved: {savename}")
    
elif args.run_mode == 1:

    print("Combining chunks...")
    
    keys = ['RA', 'DEC', 'Z', 'ZQSO', 'DELTA_F', 'NPIX', 'TOTWEIGHTS']
    
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
    savename = saveroot + f"delta-{sim_mode_tag}.fits"
    save_catalog_to_fits(savename, data_holder)
    print(f"saved: {savename}")

    delete_names = [
        saveroot + f"delta-{sim_mode_tag}-chunk-*.fits",
    ]
    print(f"Files: {delete_names} can now be safely deleted.")
    
    
