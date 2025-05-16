import pylab as pl
import healpy as hp
from astropy.io import fits
import numpy as np

import os
import shutil

import yaw
from yaw.utils import parallel
from yaw.correlation import autocorrelate_scalar, crosscorrelate_scalar
from yaw import autocorrelate
# but need to add it in the __init__
import argparse

parser = argparse.ArgumentParser(description='Compute stacked kappa profile for Dirac mocks.')

parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
parser.add_argument('-sim_mode', type=int, default=0, help='0=raw, 1=true continuum, 2=uncontaminated (picca), 3=baseline (depricated), 4=LyCAN_noSNRcut, 5=LyCAN_SNRcut (depricated), 6=LyCAN_cont_noSNRcut')
parser.add_argument('-source', type=int, default=2, help='1=QSO; 2=galaxies')
parser.add_argument('-deltaf_weight', type=int, default=2, help='0=no weight (uniform weight), 1=NPIX, 2=TOTWEIGHTS')
parser.add_argument('-unk_zcut', nargs='+', default=[1.8,3.0], help='Redshift cuts in the unknown sample redshift.')
parser.add_argument('-zbins', nargs='+', default=[2,3,40], help='Zmin, Zmax, Nbin')
parser.add_argument('-zbins_file', type=str, default="", help='Redshift bin edges file, if provided will overwrite zbins. It is encouraged to use this option for reproducibility.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-plot', type=int, default=0, help='0=no plot, 1=produce and save a plot for the results, 2=skip correlation, just plot.')
parser.add_argument('-run_unknown_corr', type=int, default=0, help='Notice this only needs to be run once for all delta_f modes. 0=automode, searches for the w_pp file in the results directory; if not found, will run unknown auto-correlation; 1=run and overwrite the existing file if any.')
parser.add_argument('-ref_tag', type=str, default="", help="tag for the ref folders; valid tags: None, 20bin")
parser.add_argument('-unk_tag', type=str, default="", help="tag for the unk folders (same tag is used for randoms zcolumn); valid tags: None, low, mid, SRD_nz.")
parser.add_argument('-yaw_tag', type=str, default="", help="tag for naming the yaw folders; used for different yaw settings such as number of redshift bins. Default is given by the default arguments above.")
args = parser.parse_args()

def delete_and_recreate_cache_directory(cache_dir):
    if parallel.on_root():  # if running with MPI, this is only executed on rank 0
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.mkdir(cache_dir)

njn=64
# here can test a range of scales
theta_min=[5,10,15] # will last bin changed from 15 to 30
theta_max=[15,30,50]
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
    sim_mode_tag = "LyCAN_noSNRcut"
elif args.sim_mode == 5:
    sim_mode_tag = "LyCAN_SNRcut"
elif args.sim_mode == 6:
    sim_mode_tag = "LyCAN_cont_noSNRcut"
    # contaminated case

if args.source == 1:
    type_tag = "QSO"
elif args.source == 2:
    type_tag = "unknown"

if args.deltaf_weight == 0:
    ref_name = 'DELTA_F'
    ref_weight_name = None
elif args.deltaf_weight == 1:
    ref_name = 'DELTA_F'
    ref_weight_name = 'NPIX'
elif args.deltaf_weight == 2:
    ref_name = 'DELTA_F_WEIGHTED'
    ref_weight_name = 'TOTWEIGHTS'


# make a z_column dictionary for unk_tag:
rand_z_dict={
    "low": "Z_LOW",
    "mid": "Z_MID",
    "SRD_nz": "Z_SRD", # this one zlim has to be [0, 3]
}

# Update all tags
if args.unk_tag == "":
    unk_tag = args.unk_tag 
    rand_z_name = "Z"
else:
    unk_tag = "-" + args.unk_tag
    rand_z_name =rand_z_dict[args.unk_tag]

if args.ref_tag == "":
    ref_tag = args.ref_tag 
else:
    ref_tag = "-" + args.ref_tag 

if args.yaw_tag == "":
    yaw_tag = args.yaw_tag
else:
    yaw_tag = "-" + args.yaw_tag


saveroot = args.outroot + f"run-{args.sim_num}/"

path_unknown = saveroot + f"catalogue/{type_tag}{unk_tag}-zmin-{args.unk_zcut[0]}-zmax-{args.unk_zcut[1]}.fits"
path_reference = saveroot + f"catalogue{ref_tag}/delta-{sim_mode_tag}.fits"
path_unk_rand = "/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-w-z.fits"

print("Unknown: ", path_unknown)
print("Reference: ", path_reference)
print("Random: ", path_unk_rand)


if args.zbins_file != "":
    zsampf = np.loadtxt(args.zbins_file)
    edges = zsampf[:,0]
    zsamp = zsampf[:-1,1]
else:
    edges = np.linspace(float(args.zbins[0]), float(args.zbins[1]), int(args.zbins[2])+1)
    zsamp = (edges[1:] + edges[:-1])/2.

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
    CACHE_DIR = saveroot + f"yaw{yaw_tag}/cache_{sim_mode_tag}/"
    print("cache: ", CACHE_DIR)
    
    delete_and_recreate_cache_directory(CACHE_DIR)
    
    # set up the catalogues:
    cat_unknown = yaw.Catalog.from_file(
        cache_directory=os.path.join(CACHE_DIR, "unknown"),
        path=path_unknown,
        ra_name="RA",
        dec_name="DEC",
        redshift_name="Z",
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
        kappa_name=ref_name,
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
        redshift_name=rand_z_name,
        patch_centers=patch_centers,
        progress=PROGRESS,
        degrees=True,
    )

    print("Done loading catalogues")
    
    # measurements:

    print("Computing w_ss")
    w_ss= autocorrelate_scalar(
        config,
        cat_reference,
        progress=PROGRESS
    ) # returns a list, one for each scale, just pick the first here
    #   w_ss.to_file("...") -> store correlation pair counts as HDF5 file

    # save files
    for ii in range(len(theta_min)):
        cts_ss = w_ss[ii]
        cts_ss.to_file(saveroot + f"yaw{yaw_tag}/w_ss-{sim_mode_tag}-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.hdf5")
    
    print("Computing w_sp")
    w_sp = crosscorrelate_scalar(
        config,
        cat_reference,
        cat_unknown,
        unk_rand=cat_unk_rand,
        progress=PROGRESS
    ) # returns a list, one for each scale, just pick the first here
    #   w_sp.to_file("...") -> store correlation pair counts as HDF5 file

    # save them (if different scales, need to save each file!):
    for ii in range(len(theta_min)):
        cts_sp = w_sp[ii]
        cts_sp.to_file(saveroot + f"yaw{yaw_tag}/w_sp-{sim_mode_tag}-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.hdf5")
    """
    w_ss = []
    w_sp = []
    for ii in range(len(theta_min)):
        fname = saveroot + f"yaw{yaw_tag}/w_ss-{sim_mode_tag}-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.hdf5"
        w_ss.append(yaw.ScalarCorrFunc.from_file(fname))

        fname = saveroot + f"yaw{yaw_tag}/w_sp-{sim_mode_tag}-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.hdf5"
        w_sp.append(yaw.ScalarCorrFunc.from_file(fname))
    """
    
    wppfname = saveroot + f"yaw{yaw_tag}/w_pp-theta-min-{theta_min[0]}-max-{theta_max[0]}.hdf5"
    if os.path.isfile(wppfname)!=True:
        print("Computing w_pp")
        # also run unknown case:
        w_pp = autocorrelate(
            config,
            cat_unknown,
            random=cat_unk_rand,
            progress=PROGRESS
        )
        for ii in range(len(theta_min)):
            cts_pp = w_pp[ii]
            cts_pp.to_file(saveroot + f"yaw{yaw_tag}/w_pp-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.hdf5")
    else:
        w_pp=[]
        for ii in range(len(theta_min)):
            fname = saveroot + f"yaw{yaw_tag}/w_pp-theta-min-{theta_min[ii]}-max-{theta_max[ii]}.hdf5"
            w_pp.append(yaw.CorrFunc.from_file(fname))
    
    # COMPUTE REDSHIFTS
    print("Computing n(z)...")
    for ii in range(len(theta_min)):
        
        cts_sp = w_sp[ii]
        cts_ss = w_ss[ii]
        cts_pp = w_pp[ii]
    
        zz = cts_sp.binning.mids
        sp = cts_sp.sample().data
        ss = cts_ss.sample().data
        pp = cts_pp.sample().data

        deltaz = zz[1]-zz[0]

        fname = saveroot + f"yaw{yaw_tag}/nz_estimate-{sim_mode_tag}-wpp-theta-min-{theta_min[ii]}-max-{theta_max[ii]}" # .dat, .smp, .cov
        
        # get errorbar:
        samps = cts_sp.sample().samples/np.sqrt((cts_ss.sample().samples)*(cts_pp.sample().samples))/deltaz
        np.savetxt(fname+".smp", np.c_[zz, samps.T])
        
        cov = np.cov(samps.T)*njn
        np.savetxt(fname+".cov", cov)
        
        std = np.sqrt(np.diag(cov))
        out=np.c_[zz, -sp/np.sqrt(ss*pp)/deltaz,std]
        np.savetxt(fname+".dat", out)
        
        # does not include pp
        ncc = yaw.RedshiftData.from_corrfuncs(cross_corr=cts_sp, ref_corr=cts_ss, unk_corr=None)
        ncc.to_files(saveroot + f"yaw{yaw_tag}/nz_estimate-{sim_mode_tag}-theta-min-{theta_min[ii]}-max-{theta_max[ii]}")  # store as ASCII files with extensions .dat, .smp and .cov


if args.plot != 0: 

    # load results:

    # produce a plot for each scale 
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
