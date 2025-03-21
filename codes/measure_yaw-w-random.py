import pylab as pl
import healpy as hp
from astropy.io import fits
import numpy as np

import os
import shutil

import yaw
from yaw.utils import parallel
from yaw.correlation import autocorrelate_scalar, crosscorrelate_scalar
# but need to add it in the __init__
import argparse

parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')

parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
parser.add_argument('-sim_mode', type=int, default=0, help='0=raw, 1=true continuum deltas, 2=uncontaminated mocks (baseline)')
parser.add_argument('-source', type=int, default=2, help='1=QSO; 2=galaxies')
parser.add_argument('-deltaf_weight', type=int, default=2, help='0=no weight (uniform weight), 1=NPIX, 2=TOTWEIGHTS')
parser.add_argument('-zcut', nargs='+', default=[1.8,3.0], help='Cuts in redshift. Provide bin edges')
parser.add_argument('-zbins_file', type=str, default="/pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt", help='Redshift bin edges file, if provided will overwrite zbins.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-plot', type=int, default=0, help='0=no plot, 1=produce and save a plot for the results, 2=skip correlation, just plot.')
args = parser.parse_args()

def delete_and_recreate_cache_directory(cache_dir):
    if parallel.on_root():  # if running with MPI, this is only executed on rank 0
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)

#set_params=3
"""
if set_params==0:
    njn=64
    theta_min=1
    theta_max=20
    theta_scaled=None
    resolution=None
    unit='arcmin'
    folder = "test-njn-64-noscale-1-20-arcm/"
    

elif set_params==1:
    njn=128
    theta_min=1
    theta_max=20
    theta_scaled=None
    resolution=None
    unit='arcmin'
    folder = "test-njn-128-noscale-1-20-arcm/"
    
        
elif set_params==2:
    njn=64
    theta_min=1
    theta_max=20
    theta_scaled=-1.0
    resolution=10
    unit='arcmin'
    folder = "test-njn-128-noscale-1-20-arcm/"
"""

#elif set_params==3:
njn=64
theta_min=1
theta_max=10
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
    sim_mode_tag = "LyCAN_noSNRc"
elif args.sim_mode == 5:
    sim_mode_tag = "LyCAN_SNRc"

if args.source == 1:
    type_tag = "QSO"
elif args.source == 2:
    type_tag = "unknown"

if args.deltaf_weight == 0:
    ref_weight_name = None
elif args.deltaf_weight == 1:
    ref_weight_name = 'NPIX'
elif args.deltaf_weight == 2:
    ref_weight_name = 'TOTWIEGHTS'
    
zbins = args.zcut

saveroot = args.outroot + f"run-{args.sim_num}/"

path_unknown = saveroot + f"catalogue/{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.fits"
path_reference = saveroot + f"catalogue/delta-{sim_mode_tag}.fits"
path_unk_rand = "/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-zmin-1.8.fits"

zsampf = np.loadtxt(args.zbins_file)
edges = zsampf[:,0]
zsamp = zsampf[:-1,1]

if args.plot != 2:

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
    CACHE_DIR = saveroot + f"yaw/cache_{sim_mode_tag}/"
    delete_and_recreate_cache_directory(CACHE_DIR)
    
    # set up the catalogues:
    cat_unknown = yaw.Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "unknown"),
        path=path_unknown,
        ra_name="RA",
        dec_name="DEC",
        #weight_name="weight_column",  # optional
        patch_num=patch_num,
        progress=PROGRESS,
        degrees=True,
    )
    patch_centers = cat_unknown.get_centers()
    
    
    cat_reference = yaw.Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "reference"),
        path=path_reference,
        ra_name="RA",
        dec_name="DEC",
        redshift_name="Z",
        weight_name=ref_weight_name,
        kappa_name="DELTA_F",
        patch_centers=patch_centers,
        progress=PROGRESS,
        degrees=True,
    )
    
    cat_ref_rand = None 
    cat_unk_rand = yaw.Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "unk_rand"),
        path=path_unk_rand,
        ra_name="RA",
        dec_name="DEC",
        patch_centers=patch_centers,
        progress=PROGRESS,
        degrees=True,
    )
    
    # measurements:
    w_ss= autocorrelate_scalar(
        config,
        cat_reference,
        progress=PROGRESS
    ) # returns a list, one for each scale, just pick the first here
    #   w_ss.to_file("...") -> store correlation pair counts as HDF5 file
    
    
    w_sp = crosscorrelate_scalar(
        config,
        cat_reference,
        cat_unknown,
        unk_rand=cat_unk_rand,
        progress=PROGRESS
    ) # returns a list, one for each scale, just pick the first here
    #   w_sp.to_file("...") -> store correlation pair counts as HDF5 file
    
    # save them:
    cts_ss = w_ss[0]
    cts_ss.to_file(saveroot + f"yaw/w_ss-{sim_mode_tag}.hdf5")
    
    cts_sp = w_sp[0]
    cts_sp.to_file(saveroot + f"yaw/w_sp-{sim_mode_tag}.hdf5")
    # restored = yaw.CorrFunc.from_file("w_sp.hdf5")
    
    # COMPUTE REDSHIFTS
    ncc = yaw.RedshiftData.from_corrfuncs(cross_corr=w_sp[0], ref_corr=w_ss[0])  # unk_corr=w_pp
    ncc.to_files(saveroot + f"yaw/nz_estimate-{sim_mode_tag}")  # store as ASCII files with extensions .dat, .smp and .cov

if args.plot != 0: 

    # load results:
    from yaw.correlation.corrfunc import ScalarCorrFunc
    
    w_sp = ScalarCorrFunc.from_file(saveroot + f"yaw/w_sp-{sim_mode_tag}.hdf5")
    w_ss = ScalarCorrFunc.from_file(saveroot + f"yaw/w_ss-{sim_mode_tag}.hdf5")
    ncc = yaw.RedshiftData.from_files(saveroot + f"yaw/nz_estimate-{sim_mode_tag}.hdf5")
    x = ncc.binning.mids

    # load nz_true:
    #nz_true = np.loadtxt("/global/homes/q/qhang/desi/lya/nz_phot_samp-zbin-match.txt")
    nz_true = np.loadtxt(saveroot + f"catalogue/nz-{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.txt")
    dz_true = nz_true[1,0] - nz_true[0,0]
    nz_true[:,1] = nz_true[:,1]/sum(nz_true[:,1])/dz_true # normalize
    # interpolate nz_true to the grid: 
    nz_true_interp = np.interp(x, nz_true[:,0], nz_true[:,1])
    
    # load true unknown bias evolution and interpolate:
    bias_z = np.loadtxt("/global/homes/q/qhang/desi/lya/bias_dc2.txt")
    bias_z_int = np.interp(zsamp, bias_z[:,0], bias_z[:,1])
    
    # save plots:
    fig,axarr=pl.subplots(2,2, figsize=[10,10])
    fig.suptitle(f"NJN={njn}, [{theta_min},{theta_max}]{unit}, random, run-{args.sim_num}")
    
    pl.sca(axarr[0,0])
    pl.errorbar(x, -ncc.data, yerr=ncc.error,fmt='.',label="ncc * (-1) (yaw measurement)")
    pl.ylabel("n(z) with arbitrary amplitude")
    pl.xlabel("redshift")
    pl.title("w_x / sqrt(w_auto)")
    pl.errorbar(x, -ncc.data/bias_z_int*bias_z_int[1], 
                yerr=ncc.error/bias_z_int*bias_z_int[1],fmt='.',label="phot. bias evol removed")
    pl.ylim([-0.2,1.0])
    # fit the factor
    yfit = (nz_true_interp/np.sum(nz_true_interp)/(x[1]-x[0]))[2:]
    ytemp = (-ncc.data/bias_z_int*bias_z_int[1])[2:]
    w = (1/(ncc.error/bias_z_int*bias_z_int[1])**2)[2:]
    ind = np.isnan(ytemp)
    factor = sum((ytemp/yfit*w)[~ind])/sum(w[~ind])

    plot_ind = nz_true[:,0]>1.8
    pl.plot(nz_true[plot_ind,0], nz_true[plot_ind,1]*factor,label="truth")
    pl.legend()
    
    pl.sca(axarr[0,1])
    pl.imshow(ncc.correlation,vmax=1,vmin=-1,cmap='RdBu',extent=[edges[0],edges[-1], edges[0],edges[-1]])
    pl.colorbar()
    pl.xlabel("redshift")
    pl.ylabel("redshift")
    pl.title("correlation matrix")
    
    pl.sca(axarr[1,0])
    w_sp_samp = w_sp.sample()  # creates a CorrFunc object
    pl.errorbar(x, w_sp_samp.data, yerr=w_sp_samp.error,fmt='.')
    pl.ylabel("w(theta)")
    pl.xlabel("redshift")
    pl.title("w_x")
    pl.ylim([-0.001,0.001])
    
    pl.sca(axarr[1,1])
    w_ss_samp = w_ss.sample()  # creates a CorrFunc object
    pl.errorbar(x, w_ss_samp.data, yerr=w_ss_samp.error,fmt='.')
    pl.ylabel("w(theta)")
    pl.xlabel("redshift")
    pl.title("w_auto")
    pl.ylim([0,0.004])
    pl.tight_layout()
    
    pl.show()
    pl.savefig(saveroot + f"plots/w_ss_sp_cov_nz-{sim_mode_tag}.pdf", bbox_inches="tight")
    pl.close()
