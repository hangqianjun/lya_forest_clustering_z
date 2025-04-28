"""
Split existing galaxy catalogue at z>1.8 into tomo bins defined by a Gaussian for testing purpose.

Note: number density is halved, be aware of the shot-noise difference.
"""

import numpy as np
from astropy.io import fits
import lya_utils as lu


type_tag='unknown'
zbins=[1.8,3.0]
sim_num = 0
root = "/pscratch/sd/q/qhang/desi-lya/results/"

Nsims = 10
for sim_num in range(1):

    print(f"Working on sim {sim_num}...")
    saveroot = root + f"run-{sim_num}/catalogue/"
    fname = saveroot + f"{type_tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.fits"
    fin = fits.open(fname)
    z = fin[1].data['Z']
    ind = np.arange(len(z))
    
    for ii in range(2):
        print(f"Working on config {ii}..")
        if ii == 0:
            # Gaussian distribution centred at z=2 and with sig=0.1
            p = lu.norm_dist(z,mu=2.0,sig=0.1)
            tag = "low"
        elif ii == 1:
            # Gaussian distribution centred at z=2.5 and with sig=0.1
            p = lu.norm_dist(z,sig=0.1)
            tag = "mid"
        
        idchosen = np.random.choice(ind, size=int(len(z)*0.5), replace=False, p=p/sum(p))

        print("Selection done, saving files...")
        
        # remember to save this n(z):
        cc = np.histogram(z[idchosen],bins=50,range=[1.8,3])
        nzout = np.c_[(cc[1][1:] + cc[1][:-1])*0.5, cc[0]]
        fname = saveroot + f"nz-{type_tag}-{tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.txt"
        np.savetxt(fname, nzout)
    
        # save the catalogue
        data_holder = {
            'RA': fin[1].data['RA'][idchosen],
            'DEC': fin[1].data['DEC'][idchosen],
            'Z': z[idchosen],
            }
        # now save:
        fname = saveroot + f"{type_tag}-{tag}-zmin-{zbins[0]}-zmax-{zbins[1]}.fits"
        lu.save_catalog_to_fits(fname, data_holder, overwrite=True)