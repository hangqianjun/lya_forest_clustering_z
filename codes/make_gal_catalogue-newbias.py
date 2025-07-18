"""
This is the script for making the photometric galaxy catalogue (and QSO catalogue)
This will grab simulation results, select a sample of photometric galaxies
either in redshift, or by matching a given redshift distribution.
Output will be a fits file with RA, DEC, true redshift, and weights. 
A n(z) file will also be saved for the catalogue.

Run with ENV = pymaster.

New bias model.
"""
import os
import numpy as np
import healpy as hp
from astropy.io import fits
from orphics import mpi,stats
import argparse
import healpy
from pixell import utils
import lya_utils as lu


parser = argparse.ArgumentParser(description='Collect galaxy catalogues for yaw.')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
#parser.add_argument('-sim_root', type=str, default="", help='If provided overwrites the sim_num, load sim from this directory. File structure has to be consistent.')
parser.add_argument('-source', type=int, default=1, help='1=QSO; 2=galaxies')
parser.add_argument('-zcut', nargs='+', default=[1.8,3], help='Cuts in redshift. Provide bin edges')
parser.add_argument('-target_nz', type=str, default="", help='Directory to target the nz file. If provided, will try to match the n(z) distribution. Or, type SRD to automatically generate SRD redshifts for highest bin. Type LBG to generate selection for LSST Y10 LBG.')
#parser.add_argument('-match_srd_ngal', type=int, default=0, help='0=ignore, 1=match')
parser.add_argument('-mask', type=str, default="/pscratch/sd/q/qhang/desi-lya/desixlsst-mask-nside-128.fits", help='Directory to survey mask.')
parser.add_argument('-outroot', type=str, default="", help='Where to save the catalogues.')
parser.add_argument('-nchunks', type=int, default=1, help='How many chunks to split the data')
parser.add_argument('-run_mode', type=int, default=2, help='0=run chunks, 1=process chunks, 2=debug, runs 0 with 1 chunk.')
args = parser.parse_args()

print("Initializing...")
# load things, set up directories:
simroot = f"/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/notebook/new_biasing_model/new_bias_model/box-{args.sim_num}/results/"

zbins = args.zcut
saveroot = args.outroot + f"run-{args.sim_num}/catalogue/"

if args.target_nz != "":
    if args.target_nz == "SRD":
        n_bins=5
        sigma_z = 0.05  # photo-z variance (scatter = 0.05 * (1 + z))
        z_bias = 0
        # compute srd n(z) for last bin:
        redshift_range = np.linspace(0,3.5,100)
        redshift_distribution = lu.srd_tot_nz(redshift_range)
        srd_bins = lu.compute_equal_number_bounds(redshift_range, redshift_distribution, n_bins)
        
        # Loop over the bins: each bin is defined by the upper and lower edge of the bin
        x1 = srd_bins[n_bins-1]
        x2 = srd_bins[n_bins]
        source_nz = lu.true_redshift_distribution(x1, x2, sigma_z, z_bias, redshift_range, redshift_distribution)
        useind = redshift_range<3
        target_nz = np.c_[redshift_range[useind], source_nz[useind]]
        cat_tag = "-SRD_nz"
        scale=8.8
    elif args.target_nz == "LBG":
        target_nz = np.loadtxt("/global/homes/q/qhang/desi/lya/notebooks/nz_lbg_lssty10.txt")
        cat_tag = "-LBG_nz"
        scale=None
    else:
        target_nz = np.loadtxt(args.target_nz)
        cat_tag = "-custom_nz"
elif args.target_nz == "":
    cat_tag = ""

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
        Z_raw = np.array([])

        files = fname_chunks[task]
        
        """
        if args.run_mode == 2:
            # just run on one file to test
            files = [files[0]]
        """
        for mm in files:
        
            fname = simroot + f"out_srcs_s{args.source}_{mm}.fits"
            f=fits.open(fname)
            
            redshift_raw = f[1].data['Z_COSMO']
            redshift = f[1].data['Z_COSMO'] + f[1].data['DZ_RSD']
            ra = f[1].data['RA']
            dec = f[1].data['DEC']

            #for kk in range(nzbins):
            sel = (redshift > float(zbins[0]))&(redshift <= float(zbins[1]))
            # convert sel to index:
            sel = np.arange(len(redshift))[sel]
            
            # now also select using nz
            if args.target_nz != "":
                # scale roughly gives SRD density
                sel_nz = lu.downsamp_w_target_hist(redshift[sel], target_nz, seed=123, scale=scale)[0]
                # update sel:
                sel = sel[sel_nz]
            
            # let's downsample over the full footprint, otherwise introduces noise
            """
            Ngal = len(sel)
            # finally downsample to match the RSD number density if needed:
            if args.match_srd_ngal == 1:
                ngal, frac = lu.match_srd_ngal_one_file(f, Ngal) # need to be updated!!
                print(f"Number density: {ngal} / arcmin^2")
                if frac >= 1:
                    print("number density lower than RSD, no downsampling.")
                elif frac<1:
                    index = np.arange(Ngal)
                    choose = np.random.choice(index, size=int(Ngal*frac), replace=False)
                    # update sel
                    sel = sel[choose]
            """
            pix = hp.ang2pix(nside, np.radians(90 - dec[sel]), np.radians(ra[sel]))
            sel1 = np.in1d(pix,usepix)
            # update sel
            sel = sel[sel1]
            pix = pix[sel1]

            n = np.bincount(pix, minlength=npix)
            cc = np.histogram(redshift[sel], bins=80, range=[0,3])
            
            # updating Ngal, print Ngal, number density
            Ngal = len(sel)
            
            # estimate number density on the map:
            nmean = sum(n)/sum(n>0)
            area = hp.nside2pixarea(nside,degrees=True)*60**2
            ngal = nmean/area
            print(f"The file contains {Ngal} galaxies. The number density is {ngal} arcmin^-2.")
            
            galmap += n
            nz += cc[0]
        
            RA=np.append(RA, ra[sel])
            DEC=np.append(DEC, dec[sel])
            Z=np.append(Z, redshift[sel])
            Z_raw=np.append(Z_raw, redshift_raw[sel])
        
        data_holder = {
        'RA': RA,
        'DEC': DEC,
        'Z': Z,
        'Z_COSMO': Z_raw,
        }

        nzout = np.c_[(cc[1][1:] + cc[1][:-1])*0.5, nz]
        
        print("Number of objects in chunk: ", len(data_holder["RA"]))
        
        # now save:
        fname = saveroot + f"{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.fits"
        lu.save_catalog_to_fits(fname, data_holder, overwrite=True)
        np.savetxt(saveroot + f"nz-{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.txt",nzout)
        hp.write_map(saveroot + f"galmap-{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.fits", galmap, overwrite=True, dtype='int')

elif args.run_mode == 1:
    
    print("Combining chunks...")
    
    keys = ['RA', 'DEC', 'Z', 'Z_COSMO']
    
    data_holder = {}
    for key in keys:
        data_holder[key] = np.array([])
    nz = 0
    galmap = 0

    for task in range(args.nchunks):
        # data holder
        fname = saveroot +  f"{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.fits"
        fin = fits.open(fname)
        for key in keys:
            data_holder[key] = np.append(data_holder[key], fin[1].data[key])

        # nz:
        fname = saveroot + f"nz-{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.txt"
        fin = np.loadtxt(fname)
        nz += fin[:,1]
        if task == 0:
            zz = fin[:,0]

        # galmap:
        fname = saveroot + f"galmap-{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-{task}.fits"
        fin = hp.read_map(fname)
        galmap += fin
        
    # save    
    savename = saveroot +  f"{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.fits"
    lu.save_catalog_to_fits(savename, data_holder)
    print(f"saved: {savename}")

    savename = saveroot + f"nz-{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.txt"
    np.savetxt(savename, np.c_[zz, nz])
    print(f"saved: {savename}")

    savename = saveroot + f"galmap-{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.fits"
    hp.write_map(savename, galmap, overwrite=True, dtype='int')
    print(f"saved: {savename}")

    delete_names = [
        saveroot + f"{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-*.fits",
        saveroot + f"nz-{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-*.txt",
        saveroot + f"galmap-{type_tag}{cat_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}-chunk-*.fits",
    ]

    for f in delete_names:
        os.system(f"rm {f}")
        
    print(f"Files: {delete_names} have been safely deleted.")