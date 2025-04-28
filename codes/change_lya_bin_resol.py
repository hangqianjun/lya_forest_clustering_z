"""
Here we produce the lower resolution bins by combining
the 40-bin Lya files.

Given the redshift edges, we do:
sum(weighted_delta_F)/sum(tot_pix_weights)

Mainly for producing the 20-bin Lya files, labeled "-halfbin"
"""


#from pixell import utils
import numpy as np
import healpy as hp
#sfrom orphics import mpi,stats
from astropy.io import fits
#import treecorr
from glob import glob
#import argparse


root = "/pscratch/sd/q/qhang/desi-lya/results/"

sim_modes = ["raw", "true_cont","uncontaminated"]#,"LyCAN_noSNRcut", "LyCAN_SNRcut"]

Nsims = 10

bin_edges_original = np.linspace(2,3,41)

Nbins = 20
bin_edges_new = np.linspace(2,3,Nbins+1)


keys = ['RA', 'DEC', 'Z', 'ZQSO', 'DELTA_F', 'DELTA_F_WEIGHTED', 'NPIX', 'TOTWEIGHTS']


for ii in range(Nsims):

    for mode in sim_modes:

        fname = root + f"run-{ii}/catalogue/delta-{mode}.fits"
        fin = fits.open(fname)

        zz = fin[1].data['Z']

        data_holder = {}
        for key in keys:
            data_holder[key] = np.array([])

        for bb in range(Nbins):
            ind = (zz>=bin_edges_new[bb])&(zz<bin_edges_new[bb+1])
            
            # Getting the object ID for selected objects:
            ig, idra = np.unique(fin[1].data['RA'][ind], return_inverse=True)
            idra = idra.astype('str')

            ig, iddec = np.unique(fin[1].data['DEC'][ind], return_inverse=True)
            iddec = iddec.astype('str')
            
            ig, idz = np.unique(fin[1].data['ZQSO'][ind], return_inverse=True)
            idz = idz.astype('str')

            ID = idra + "d" + iddec + "z" + idz
            ig, final_ID = np.unique(ID,return_inverse=True)

            # now do bincounts 
            # bincounts are sorted in final_ID, which means it is the same order as sorted unique ID
            sum_delta = np.bincount(final_ID, weight=fin[1].data['DELTA_F'][ind] * fin[1].data['NPIX'][ind])
            new_npix = np.bincount(final_ID, weight=fin[1].data['NPIX'][ind])
            new_delta_f = sum_delta/new_npix

            sum_delta_w = np.bincount(final_ID, weight=fin[1].data['DELTA_F_WEIGHTED'][ind] * fin[1].data['TOT_WEIGHTS'][ind])
            new_totweights = np.bincount(final_ID, weight=fin[1].data['TOT_WEIGHTS'][ind])
            new_delta_f_weighted = sum_delta_w/new_totweights
            
            # we need to register ra, dec, zqso, and z for these things   
            #nobj = np.bincount(final_ID)
            #new_ra = np.bincount(final_ID, weight=fin[1].data['RA'][ind])/nobj
            #new_dec = np.bincount(final_ID, weight=fin[1].data['DEC'][ind])/nobj
            #new_zqso = np.bincount(final_ID, weight=fin[1].data['ZQSO'][ind])/nobj

            # final_ind gives sorted unique ID
            ig, final_ind = np.unique(ID, return_index=True)
            new_ra = fin[1].data['RA'][ind][final_ind]
            new_dec = fin[1].data['DEC'][ind][final_ind]
            new_zqso = fin[1].data['ZQSO'][ind][final_ind]

            # finally we put all z in one slice
            new_z = np.ones(len(sum_delta_w))*(bin_edges_new[bb] + bin_edges_new[bb+1])/2.

            data_holder['RA'] = np.append(data_holder['RA'],new_ra)
            data_holder['DEC'] = np.append(data_holder['DEC'],new_dec)
            data_holder['ZQSO'] = np.append(data_holder['ZQSO'],new_zqso)
            data_holder['Z'] = np.append(data_holder['Z'],new_z)
            data_holder['DELTA_F'] = np.append(data_holder['DELTA_F'],new_delta_f)
            data_holder['NPIX'] = np.append(data_holder['NPIX'],new_npix)
            data_holder['DELTA_F_WEIGHTED'] = np.append(data_holder['DELTA_F_WEIGHTED'],new_delta_f_weighted)
            data_holder['TOTWEIGHTS'] = np.append(data_holder['TOTWEIGHTS'],new_totweights)
            
        print(f"Total number of objects: {len(data_holder[key])}")
        # save    
        savename = saveroot +  f"delta-{sim_mode_tag}.fits"
        save_catalog_to_fits(savename, data_holder)
        print(f"saved: {savename}")