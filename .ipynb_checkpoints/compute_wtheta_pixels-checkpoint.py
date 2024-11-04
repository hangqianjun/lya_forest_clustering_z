"""
Computes angular correlation function on a healpix pixel of nside=8
Loops over different pixels and combine the results
Also need to loop over the zbins in each slice
"""
from pixell import utils
import numpy as np
import healpy as hp
from orphics import mpi,stats
from astropy.io import fits
import treecorr
#import actxdesi_functions_nmt as adf_mnt
#import pymaster as nmt


def select_cat(f, pixel, zcut=[1.8,3.1], nside=8):
    redshift_cat= f[1].data['Z_COSMO']
    sel = (redshift_cat>zcut[0])&(redshift_cat<zcut[1])
    
    ra_cat = f[1].data['RA'][sel]
    dec_cat = f[1].data['DEC'][sel]

    pix_cat = hp.ang2pix(nside,
                     np.radians(90 - dec_cat),
                     np.radians(ra_cat))
    
    inmask = np.isin(pix_cat, np.array([int(pixel)]))
    #rint(len(ra_cat[inmask]))
    #print(np.intersect1d(np.unique(pix_cat), np.array([int(pixel)])))
    
    cat = treecorr.Catalog(ra=ra_cat[inmask], dec=dec_cat[inmask], ra_units='deg', dec_units='deg')
    return cat


def select_deltaF(pixel, zbin_use, nside=8):
    
    #num = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,27,28,30]
    # zbin between 0 and 17, last few bins very sparse...
    
    pixel_str = str(int(pixel))
        
    fname = f"/pscratch/sd/q/qhang/desi-lya/delta_F/comb_pixgroup.fits"
    fin = fits.open(fname)
    
    ra = fin[1].data['RA']
    dec =fin[1].data['DEC']
    zbins = fin[1].data['Z_BIN']
    
    pix = hp.ang2pix(nside,
                 np.radians(90 - dec),
                 np.radians(ra))
    
    # select objects inside the pixel and redshift bin
    sel = np.isin(pix, np.array([int(pixel)]))
    sel *= np.isin(zbins, zbin_use)
    
    deltaF = treecorr.Catalog(ra=ra[sel], dec=dec[sel],
                              ra_units='deg', dec_units='deg', 
                              k=fin[1].data['DELTA_F'][sel])
    totdF = np.array([np.sum(fin[1].data['DELTA_F'][sel]), len(fin[1].data['DELTA_F'][sel])])
    return deltaF, totdF


simroot = "/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/test_box/gal_box/results/"
saveroot = "/pscratch/sd/q/qhang/desi-lya/delta_F/"

zbin_use = np.array([1]) # need to think how to automate this!
nfin = 128
nside=8
comm,rank,my_tasks = mpi.distribute(nfin)
s = stats.Stats(comm)

pixel_list = np.loadtxt(saveroot + "pixel_list-nside-8.txt")

tot_nn = []
tot_kn = []
tot_dF = []
sample_xi = []

for task in my_tasks:
    
    fname = simroot + f"out_srcs_s2_{task}.fits"
    f=fits.open(fname)
    
    ra_cat = f[1].data['RA'][::50]
    dec_cat = f[1].data['DEC'][::50]    
    # figure out which pixels overlap:
    # show map:
    pix_cat = hp.ang2pix(nside,
                     np.radians(90 - dec_cat),
                     np.radians(ra_cat))
    uniq_pix, nobjcount = np.unique(pix_cat,return_counts=True)
    # check if enough objects in the pixel:
    uniq_pix = uniq_pix[nobjcount>100]
    
    use = np.intersect1d(uniq_pix, pixel_list)
    #print(use)
    
    if len(use)>0:
        for pixel in use:
            cat = select_cat(f, pixel, zcut=[1.8,3.1], nside=8)

            #for zbin_use in [1,3,8,12,16]:
            deltaF, totdF = select_deltaF(pixel, zbin_use, nside=8)

            # now compute nk, nn:
            nk = treecorr.NKCorrelation(min_sep=1, max_sep=20, nbins=10, sep_units='arcmin')
            nk.process(cat, deltaF)  

            nn = treecorr.NNCorrelation(min_sep=1, max_sep=20, nbins=10, sep_units='arcmin')
            nn.process(cat, deltaF)

            # register these things
            tot_nn.append(nn.npairs)

            tot_kn.append(nk.raw_xi*nn.npairs)

            tot_dF.append(totdF)

            meandF = totdF[0]/totdF[1]
            sample_xi.append(nk.raw_xi - meandF)
        
    r = np.exp(nk.meanlogr)
    
s.get_stacks()
tot_nn = utils.allgatherv(tot_nn,comm)
tot_kn = utils.allgatherv(tot_kn,comm)
tot_dF = utils.allgatherv(tot_dF,comm)
sample_xi = utils.allgatherv(sample_xi,comm)

# save these results:
np.savetxt(saveroot + f"tot_nn-zbin-{zbin_use[0]}.txt", tot_nn)
np.savetxt(saveroot + f"tot_kn-zbin-{zbin_use[0]}.txt", tot_dF)
np.savetxt(saveroot + f"sample_xi-zbin-{zbin_use[0]}.txt", sample_xi)
np.savetxt(saveroot + "rsamp.txt", r)