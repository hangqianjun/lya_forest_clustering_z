"""
This is the script for making the Lya flux catalogue from the Lya skewers (RAW only).
This will grab simulation results, normalize the fluxes, and split the transmission
into user-defined redshift bins. 
Output will be a fits file with RA, DEC, delta_F, flux redshift, flux redshift bin, QSO redshift, and weights.

RAW files have different structure, hence this script. For true_cont and baseline cases, use make_lya_catalogue.py
"""

from pixell import utils
import numpy as np
import healpy as hp
from orphics import mpi,stats
from astropy.io import fits
#import treecorr
from glob import glob
import argparse


parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
parser.add_argument('-sim_mode_tag', type=str, default="LyCAN", help="LyCAN = uncontaminated, for type LyCAN_cont")
parser.add_argument('-SNRcut', type=int, default=0, help='0=noSNRc, 1=SNRc')
parser.add_argument('-zbins', nargs='+', default=[2,3,40], help='Zmin, Zmax, Nbin')
parser.add_argument('-nchunks', type=int, default=1, help='How many chunks to split the data')
parser.add_argument('-zbins_file', type=str, default="", help='Redshift bin edges file, if provided will overwrite zbins.')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-run_mode', type=int, default=0, help='0=run chunks, 1=process chunks, 2=debug, runs 0 with 1 chunk.') 
parser.add_argument('-cat_tag', type=str, default="", help="Custom tag added to the catalogue folder to distinguish different settings, such as number of zbins. If using all above default setting, the default 'catalogue/' folder is assumed.")
args = parser.parse_args()

# def save fits file:
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
    t.writeto(fname, overwrite=overwrite)

#sim_mode_tag = "LyCAN"
raw_sim_mode_tag = args.sim_mode_tag
if args.SNRcut == 0:
    lycan_mode_tag = 'noSNRcut'
elif args.SNRcut == 1:
    lycan_mode_tag = 'SNRcut'

sim_mode_tag = raw_sim_mode_tag + "_" + lycan_mode_tag

# read in redshift bin somewhere, or save it with the results;
simroot = "/global/cfs/cdirs/desi/users/wmturner/photo-z/deltas/"
if raw_sim_mode_tag == "LyCAN":
    simroot += f"mock-{args.sim_num}/jura-0/{lycan_mode_tag}/Delta/"
elif raw_sim_mode_tag == "LyCAN_cont":
    simroot += f"mock-{args.sim_num}/jura-124/{lycan_mode_tag}/Delta/"
else:
    print("sim_mode_tag not accepted.")
    exit()

#print(simroot)
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

pixels_in_mask = hp.ring2nest(16,pixels_in_mask)

fname_list = glob(simroot + "*.fits", recursive = True)

fname_pix = []
for i in range(len(fname_list)):
    fname_pix.append(int(fname_list[i][(len(simroot) + 6):-5]))

fname_pix = np.array(fname_pix)
fname_ind = np.in1d(fname_pix, pixels_in_mask)
fname_list = np.array(fname_list)[fname_ind]
print("Total files to go through: ", len(fname_list))

emit = 1215.67
lambda_rf_min=1040
lambda_rf_max=1200
dodgy_lowz_cut=3600

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
    
    # here call mpi
    if args.run_mode == 0:
        comm,rank,my_tasks = mpi.distribute(args.nchunks)
        s = stats.Stats(comm)
    elif args.run_mode == 2:
        my_tasks = [37]

    #for task in my_tasks:
    for task in my_tasks:
        
        use_fname_list = fname_chunks[task]

        #if args.run_mode == 2:
        #    use_fname_list = [use_fname_list[0]]
        
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
            print(f"{nobj} objects to go through...")
    
            for jj in range(nobj):
    
                wave = delta_F[jj+1].data['LAMBDA']
                delta_l = delta_F[jj+1].data['DELTA']
                weight_l = delta_F[jj+1].data['WEIGHT']
                #cont_l = delta_F[1].data['CONT']
                
                # for each, bin in redshift: 
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

                sel2 = ~np.isnan(delta_l)
    
                # now bin:
                for kk in range(nbin):
                    useind = (bin_tag == kk+1)*in_forest*sel1*sel2
                    
                    if len(objred[useind])>0:
                        num_pix = len(objred[useind])
                        totweights = np.sum(weight_l[useind])

                        # here we save both weighted and non-weighted deltas
                        deltaF = np.sum(delta_l[useind])/num_pix
                        
                        # weighted average of deltaF
                        deltaFw = np.sum(delta_l[useind]*weight_l[useind])/totweights

                        data_holder['RA'] = np.append(data_holder['RA'],ra)
                        data_holder['DEC'] = np.append(data_holder['DEC'],dec)
                        data_holder['ZQSO'] = np.append(data_holder['ZQSO'],zqso)
                        data_holder['Z'] = np.append(data_holder['Z'],zbin_centre[kk])
                        data_holder['DELTA_F'] = np.append(data_holder['DELTA_F'],deltaF)
                        data_holder['DELTA_F_WEIGHTED'] = np.append(data_holder['DELTA_F_WEIGHTED'],deltaFw)
                        data_holder['NPIX'] = np.append(data_holder['NPIX'],num_pix)
                        data_holder['TOTWEIGHTS'] = np.append(data_holder['TOTWEIGHTS'],totweights)
        
        print("Number of objects: ", len(data_holder['RA']))
        if len(data_holder['RA'])>0:
            # save
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
    print(f"Files: {delete_names} can now be safely deleted.")