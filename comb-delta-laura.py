from pixell import utils
import numpy as np
import healpy as hp
from orphics import mpi,stats
from astropy.io import fits
#import treecorr
from glob import glob

# def save fits file:
def save_catalog_to_fits(fname, data_matrix):
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
    t.writeto(fname)

print("Initializing...")

simroot = "/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/skewers_full_sky/output_files/"
hdu=fits.open(simroot + '13/1375/transmission-16-1375.fits.gz')

wave=hdu[2].data
emit = 1215.67
red = (wave-emit)/emit
nbin=20
ind = red<3.0
bin_edges = np.linspace(red[ind][0], red[ind][-1], nbin+1)
#print(bin_edges)
bin_cen = (bin_edges[1:]+bin_edges[:-1])/2.

print("Starting...")


simroot = "/global/cfs/cdirs/desicollab/users/lauracdp/photo-z_box/Delta/"
# grab all file names in this directory:
fname_list = glob(simroot + "*.fits.gz", recursive = True)


Nfiles=len(fname_list)
#Nfiles=2
print("Total files to go through: ", Nfiles)

saveroot = "/pscratch/sd/q/qhang/desi-lya/"


nodes=32
#nodes=2
if Nfiles%nodes ==0:
    ncomputes = int(Nfiles/nodes)
else:
    ncomputes = int(Nfiles/nodes)+1
    
print(ncomputes)

comm,rank,my_tasks = mpi.distribute(nodes)
s = stats.Stats(comm)

for task in my_tasks:
    
    ind1 = task*ncomputes
    ind2 = (task+1)*ncomputes
    
    use_fname_list = fname_list[ind1:ind2]
    
    data_holder={}
    
    for fname in use_fname_list:
    
        delta_F = fits.open(fname)
        # grab all objects in this file:
        nobj = len(delta_F)-1

        for jj in range(nobj):

            wavelength_log = delta_F[jj+1].data['LOGLAM']
            delta_l = delta_F[jj+1].data['DELTA']
            #weight_l = delta_F[1].data['WEIGHT']
            #cont_l = delta_F[1].data['CONT']

            # for each, bin in redshift: 
            objred = (10**wavelength_log-emit)/emit
            # compute the averaged spectra:
            bin_tag = np.digitize(objred, bin_edges)

            hduh = delta_F[jj+1].header
            ra = hduh['RA']
            dec = hduh['DEC']

            # now bin:
            for kk in range(nbin):
                useind = bin_tag == kk+1
                if len(objred[useind])>0:
                    num_pix = len(objred[useind])
                    deltaF = np.sum(delta_l[useind])/num_pix

                    if 'RA' not in list(data_holder.keys()):
                        data_holder['RA'] = ra
                        data_holder['DEC'] = dec
                        data_holder['Z_BIN'] = kk+1
                        data_holder['DELTA_F'] = deltaF
                        data_holder['NPIX'] = num_pix
                    else:
                        data_holder['RA'] = np.append(data_holder['RA'],ra)
                        data_holder['DEC'] = np.append(data_holder['DEC'],dec)
                        data_holder['Z_BIN'] = np.append(data_holder['Z_BIN'],kk+1)
                        data_holder['DELTA_F'] = np.append(data_holder['DELTA_F'],deltaF)
                        data_holder['NPIX'] = np.append(data_holder['NPIX'],num_pix)
                        
    print("Number of objects: ", len(data_holder['RA']))

    savename = saveroot + f"delta_F/delta-laura-part-{task}.fits"
    save_catalog_to_fits(savename, data_holder)
    print(f"saved: {savename}")