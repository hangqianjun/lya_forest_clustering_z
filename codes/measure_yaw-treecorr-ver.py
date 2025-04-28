import pylab as pl
import healpy as hp
from astropy.io import fits
import numpy as np

import os
import shutil

import treecorr

set_params=0
savedir = "/pscratch/sd/q/qhang/desi-lya/yaw/"

if set_params==0:
    njn=64
    theta_min=1
    theta_max=20
    unit='arcmin'
    folder = "test-njn-64-noscale-1-20-arcm/"
    
elif set_params==1:
    njn=128
    theta_min=1
    theta_max=20
    unit='arcmin'
    folder = "test-njn-128-noscale-1-20-arcm/"
    
elif set_params==3:
    njn=64
    theta_min=1
    theta_max=10
    unit='arcmin'
    folder = "test-njn-64-noscale-1-10-arcm/"


zsampf = np.loadtxt('/pscratch/sd/q/qhang/desi-lya/delta_F/zbins.txt')
edges = zsampf[:,0]
#zsamp = zsampf[:-1,1]

galaxy = treecorr.Catalog(file_name="/pscratch/sd/q/qhang/desi-lya/photometry-catalogue-overlap-zmin-1.8.fits",
                          ra_col='RA', dec_col='DEC',
                          ra_units='deg', dec_units='deg', 
                          npatch=njn,
                         )

fname = "/pscratch/sd/q/qhang/desi-lya/delta-laura-comb-overlap.fits"
cat = fits.open(fname)
z = cat[1].data['Z']

nk = treecorr.NKCorrelation(min_sep=theta_min, max_sep=theta_max, nbins=1, sep_units=unit,
                                   bin_slop=0, var_method='jackknife')

kk = treecorr.KKCorrelation(min_sep=theta_min, max_sep=theta_max, nbins=1, sep_units=unit,
                                   bin_slop=0, var_method='jackknife')

res_holder_nk = np.zeros((len(edges)-1,2))
res_holder_kk = np.zeros((len(edges)-1,2))
res_holder_ncc = np.zeros((len(edges)-1,2))

# columns: (xi, npairs, patch1, patch2)
for ii in range(len(edges)-1):
    
    sel = (z>=edges[ii])&(z<edges[ii+1])
    
    if len(z[sel])>0:
        
        print(f"Working on z slice {ii}")
        
        deltaF = treecorr.Catalog(
                          ra=cat[1].data['RA'][sel], dec=cat[1].data['DEC'][sel],
                          ra_units='deg', dec_units='deg', 
                          k=cat[1].data['DELTAF'][sel],
                          patch_centers=galaxy.patch_centers,
                         )
        
        nk.process(galaxy,deltaF)
        kk.process(deltaF)
        
        res_holder_nk[ii,0] = nk.raw_xi[0]
        res_holder_nk[ii,1] = np.sqrt(nk.raw_varxi[0])
        
        res_holder_kk[ii,0] = kk.xi[0]
        res_holder_kk[ii,1] = np.sqrt(kk.varxi[0])
        
        func = lambda corrs: corrs[0].raw_xi / np.sqrt(corrs[1].xi)
        corrs = [nk, kk]
        ratio = func(corrs)
        cov = treecorr.estimate_multi_cov(corrs, method='jackknife', func=func)
        res_holder_ncc[ii,0] = ratio[0]
        res_holder_ncc[ii,1] = np.sqrt(cov[0,0])
           
# save the results - njn samples and std etc.
np.savetxt(savedir + folder + "treecorr_nk.txt", res_holder_nk)
np.savetxt(savedir + folder + "treecorr_kk.txt", res_holder_kk)
np.savetxt(savedir + folder + "treecorr_ncc.txt", res_holder_ncc)