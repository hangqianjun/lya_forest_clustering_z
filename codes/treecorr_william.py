"""
Run in yaw_env
"""

#Computing
from astropy.io import fits
import numpy as np
import math
import treecorr
import time
import random
#import pyccl as ccl
import gc

#for masks:
#import healpy as hp
#import h5py as h5
#import healsparse as hs


#for saving:
import pickle
import lya_utils as lu

#for ploting
import matplotlib.pyplot as plt
from matplotlib import rcParams
from IPython.display import display, Math

#rsd_tag = "-norsd"
#zkey = "Z_COSMO"

rsd_tag = ""
zkey = "Z"

# load the data photometric sample and the randoms:
fin = fits.open('/pscratch/sd/q/qhang/desi-lya/results/run-0/catalogue/unknown-zmin-1.8-zmax-3.0.fits')
ra = fin[1].data['RA']
dec = fin[1].data['DEC']
z = fin[1].data[zkey]

fin = fits.open('/pscratch/sd/q/qhang/desi-lya/random-catalogue-overlap-w-z.fits')
ra_rand_use = fin[1].data['RA']
dec_rand_use = fin[1].data['DeC']


def w_xx(sample_x, theta_range, z_range, Ntheta, Nz, Npatch, z_for_rand ,Eta_rand):
    '''
    Code to compute the auto-correlation of sample_x,
    -------------------------------------
    Outputs:
    Wz: is a Nz x Nalpha array, where Nz is the number of z bins, and Nalpha the number of scale weighting
    Thus Wz[iz][ialpha] is the value for wxx for the z-bin iz, and the scale weighting ialpha
    
    Err is a Nz x Nalpha uncertainty array associated to WZ.

    Multi_cov is the Nz x Njkk x Nalpha array usefull to compute the full covariance!!

    For std tests, use directly WZ and Err. Usually redshift covariance is negligible.
    -------------------------------------
    Inputs:
    sample_x: name of the sample, the only option now is sample_x='photo' and 'eBOSS ELG'
    
    theta_range is an array = [theta_min,theta_max], with theta IN ARCMIN !!!!
    z_for_rand is an array = [zmin,zmax]

    Ntheta is the number of theta-bins (default use 10)
    Nz is the number of z-bins
    Npatch the number of Jkk patches to evaluate the cov
    
    z_for_rand==True means there are redshifts for randoms
    Eta_rand is how many more randoms you want, to evaluate the DD/RR counts
    -------------------------------------
    '''
    print('evaluate the auto-corr of ', sample_x)
    print('for',z_range[0], '<z<',z_range[1],'  with ', Nz, ' bins')

    zmin=z_range[0]
    zmax=z_range[1]
    dz=(zmax-zmin)/Nz
    theta_min=theta_range[0]/60 # arcmin to degree
    theta_max=theta_range[1]/60 # arcmin to degree
    
    # You can add other options, with another elif, eg with sample_x=='spec'
    if sample_x=='photo':
        ra_gal =ra
        dec_gal =dec
        z_gal = z
        w_gal = np.array([1 for i in ra_gal]) #if you do have weights, replace this array
        
        ra_rand = ra_rand_use
        dec_rand = dec_rand_use
        w_rand =  np.array([1 for i in ra_rand])
        if z_for_rand==True:
            z_rand=z_rand_use
    elif sample_x=='eBOSS ELG':
        ra_gal =ra_eboss_ELG_south
        dec_gal =dec_eboss_ELG_south
        z_gal =z_eboss_ELG_south
        w_gal =  weight_eboss_ELG_south#np.array([1 for i in ra_gal]) #if you do have weights, replace this array
        
        ra_rand =ra_rand_eboss_ELG_south
        dec_rand =dec_rand_eboss_ELG_south
        w_rand =  weight_rand_eboss_ELG_south
        if z_for_rand==True:
            z_rand=z_rand_eboss_ELG_south
    else:
        print('sample_x not included in the code')

    print(len(ra_gal) ,' galaxies ')
    print(len(ra_rand) ,'randoms ')

    
    Wz=[]
    Cov=[]
    Err=[]
    Multi_cov=[]
    
    #create a catalog to have the Jkk patches
    cat_patch = treecorr.Catalog(ra=ra_gal,dec=dec_gal,w=w_gal,ra_units='degrees',dec_units='degrees',npatch=Npatch)

    for iz in range(Nz):
        zmean_i=zmin+(iz+0.5)*dz
        zmin_i=zmin+(iz)*dz
        zmax_i=zmin+(iz+1)*dz

        print('zi=',round(zmean_i,3),' dz=',round(dz,3))
        
        sel_gal_subbin=((z_gal>=zmin_i)&(z_gal<zmax_i))
        cat_gal_subbin=treecorr.Catalog(ra=ra_gal[sel_gal_subbin],dec=dec_gal[sel_gal_subbin],w=w_gal[sel_gal_subbin],ra_units='degrees',dec_units='degrees',patch_centers=cat_patch.patch_centers)
        

        # Do we have z for randoms: 
        if z_for_rand==True:
            sel_rand_subbin=((z_rand>=zmin_i)&(z_rand<zmax_i))
        else:
            sel_rand_subbin=np.array([True for r in ra_rand])
        # Select Eta_rand more randoms than gal
        Ntot=len(ra_rand[sel_rand_subbin])
        Nrand=Eta_rand*len(ra_gal[sel_gal_subbin])
        Index=random.choices(range(Ntot), k=Nrand)
        
        
        ra_rand_select=ra_rand[sel_rand_subbin][Index]
        dec_rand_select=dec_rand[sel_rand_subbin][Index]
        w_rand_select=w_rand[sel_rand_subbin][Index]
    
        cat_rand_subbin=treecorr.Catalog(ra=ra_rand_select,dec=dec_rand_select,w=w_rand_select,ra_units='degrees',dec_units='degrees', patch_centers=cat_patch.patch_centers)
    
        wxx  = treecorr.NNCorrelation(min_sep=theta_min,max_sep=theta_max,nbins=Ntheta,var_method='jackknife',sep_units='degree',bin_slop=0.01)
        rrxx = treecorr.NNCorrelation(min_sep=theta_min,max_sep=theta_max,nbins=Ntheta,var_method='jackknife',sep_units='degree',bin_slop=0.01)
        drxx = treecorr.NNCorrelation(min_sep=theta_min,max_sep=theta_max,nbins=Ntheta,var_method='jackknife',sep_units='degree',bin_slop=0.01)

        #Now I use r as a name instead of theta
        rlist=wxx.rnom
        redges=wxx.right_edges-wxx.left_edges
        #print(rlist)
        #print(redges)
        def integ_wxx(w1):
            '''
            integrate w1 over theta, with different scales weighting, defined by List_alpha:
            W(theta)=theta**alpha/norm
            '''
            List_alpha=[-1,0,1]
            Results_alpha=[]
            Noweight=w1
            for alpha in  List_alpha:
                w1b=0
                norm=0
                for ir in range(len(w1)):
                    w1b+=w1[ir]*rlist[ir]**alpha*redges[ir]
                    norm+=rlist[ir]**alpha*redges[ir]
                Results_alpha.append(w1b/norm)

            return np.array(Results_alpha)

        def return_wxx(w1):
            return w1
            
        rrxx.process(cat_rand_subbin,cat_rand_subbin)
        drxx.process(cat_rand_subbin,cat_gal_subbin)
        
        wxx.process(cat_gal_subbin,cat_gal_subbin)
        wxx.calculateXi(rr=rrxx,dr=drxx)
         
        my_funct = lambda corrs: integ_wxx(corrs[0].xi)
        #my_funct =  lambda corrs: return_wxx(corrs[0].xi)
        corrs = [wxx]
            
        ratio = my_funct(corrs)  
        cov = treecorr.estimate_multi_cov(corrs, 'jackknife', func=my_funct)
        #multi_cov=treecorr.build_multi_cov_design_matrix(corrs,'jackknife', func=my_funct, comm=None)
        
        Wz.append(ratio)
        Cov.append(cov)
        Err.append([np.sqrt(cov[i][i]) for i in range(np.size(cov,0))])
        #Multi_cov.append(multi_cov[0])
        
        del(cat_gal_subbin)
        del(cat_rand_subbin)
        gc.collect()
    return(Wz,Err,Multi_cov)

#Nz = 40
Nz = 20
#Nz = 10
theta_min = 10
theta_max = 30
ntheta = 10

A0,B0,C0=w_xx(sample_x='photo', theta_range=[theta_min,theta_max], z_range=[2,3], Ntheta=ntheta, Nz=Nz, Npatch=64, z_for_rand=False ,Eta_rand=5)

# save all of them:
#filename = f"wpp{rsd_tag}-theta-{ntheta}bins-min-{theta_min}-max-{theta_max}-z-{Nz}bin.pkl"
filename = f"wpp{rsd_tag}-thetacomb-alpha-min-{theta_min}-max-{theta_max}-z-{Nz}bin.pkl"
lu.dump_save([A0,B0],filename)