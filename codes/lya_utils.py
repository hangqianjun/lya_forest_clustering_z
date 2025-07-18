"""
Unitilty functions for lyacc project
"""
import numpy as np
from astropy.io import fits
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.special import erf
import healpy as hp
import os
import pickle


def save_catalog_to_fits(fname, data_matrix, overwrite=True):
    """
    Saving a table in numpy dictionary to fits.
    """
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


def norm_dist(x, mu=2.5, sig=0.2):
    """
    Returns a normalised Gaussian pdf with mu, sigma
    """
    return 1/sig/np.sqrt(2*np.pi)*np.exp(-0.5*(x-mu)**2/(2*sig)**2)


def downsamp_w_target_hist(z, target_dist, seed=123, select_highest=True, scale=None):
    """
    Selects a sub-sample of the array z with a target
    histogram distribution.
    
    z: input array, e.g. redshift of the catalogue
    
    target_hist: the target histogram, first column = zsamping, 
    later columns = target distribution, can contain multiple 
    distributions. If multiple, they cannot be normalized
    (so ratio between each is known) - otherwise they will be
    equi-populated bins.
    """
    zbin_edges, cumratio = get_target_ratio(z, target_dist,  scale=scale)
    tomo_assign = get_tomo_bin(z, zbin_edges, cumratio, seed)

    Nbins = target_dist.shape[1]-1

    useind = np.arange(len(z))

    selected_id = []
    if select_highest == False:
        # return all bin indices
        for ii in range(Nbins):
            selection = tomo_assign == ii + 1
            selected_id.append(useind[selection])
    elif select_highest == True:
        selection = tomo_assign == Nbins
        selected_id.append(useind[selection])
    return selected_id


def get_target_ratio(z, nz, scale=None):

    """
    If scale is provided, it will be used.
    If the scaled target distribution is higher than tot,
    it will be forced to tot
    """
    
    Nbins = nz.shape[1]-1
    
    # binning of the target hist
    zsamp = nz[:,0]
    dz = zsamp[1]-zsamp[0] # assumes linear sampling
    zbin_edges = np.append(zsamp-dz/2., zsamp[-1]+dz/2.)
    
    if Nbins > 1:
        # combination of all target hist
        nz_tomo_tot = np.sum(nz[:,1:], axis=1)
        # un-normalized target histogram in each bin
        nz_tomo_nonorm = nz[:,1:]/sum(nz_tomo_tot)/dz
        # normalize
        nz_tomo_tot = nz_tomo_tot/sum(nz_tomo_tot)/dz
    elif Nbins == 1:
        nz_tomo_tot = nz[:,1]/sum(nz[:,1])/dz
        nz_tomo_nonorm = nz_tomo_tot
        
    # now let's get the nz distribution of the sample:
    nz_this, ig = np.histogram(z, bins = zbin_edges) # this may take a while
    # normalize
    nz_this = nz_this/sum(nz_this)/dz

    if scale == None:
        # so now figure out the scaling first:
        ratio = nz_tomo_tot/nz_this
        ratio[np.isnan(ratio)]=0
        ratio[np.isinf(ratio)]=0
        scale=ratio.max()
    
    if Nbins >1:
        for ii in range(Nbins):
            use = nz_tomo_nonorm[:,ii]/scale/nz_this
            # the case where use > 1 needs some care
            if sum(use>1)>0:
                raise TypeError("ratio > 1 for Nbins >1. This case is not implemented. exiting...")
            if ii==0:
                ratio= np.c_[zsamp, use]
            else:
                ratio = np.c_[ratio, use]
    elif Nbins == 1:
        use = nz_tomo_nonorm/scale/nz_this
        # in the case where scale is provided:
        if sum(use>1)>0:
            use[use>1] = 1
        ratio= np.c_[zsamp, use]
    
    # unbinned objects goes here:
    ratio=np.c_[ratio, 1-np.sum(ratio[:,1:],axis=1)]
    cumratio = np.cumsum(ratio[:,1:],axis=1)
    
    return zbin_edges, cumratio


def get_tomo_bin(redshift, zbin_edges, cumratio, seed):
    # regulate the cumratio to avoid error in digitize:
    # null things close to zero, set to 1 for things close to one
    # this is because sometimes numpy can sum to 1.00000000002 or 
    # something crazy like this...
    digibins = np.append(np.zeros((cumratio.shape[0],1)), cumratio, axis=1)
    # fix any nans to zero
    digibins[np.isnan(digibins)]=0
    # set last column to one 
    digibins[:,-1]=1
    # regularize
    ind1 = np.around(digibins[:,1],5)==0
    digibins[ind1,1] = 0
    ind2 = np.around(digibins[:,-2],5)==1
    digibins[ind2,-2] = 1

    np.random.seed(seed)
    
    bins = np.digitize(redshift, zbin_edges)
    rand = np.random.uniform(size=len(redshift))
    tomo_assign = np.zeros(len(redshift))

    # now digitize the bins:
    for kk in range(len(zbin_edges)-1):
        ind = bins==kk+1
        tomo_assign[ind]=np.digitize(rand[ind], digibins[kk,:])
        
    return tomo_assign


def srd_tot_nz(z, z0=0.11, alpha=0.68):
    """
    params from SRD Eq.5
    CCLX: see params here: 
    https://github.com/LSSTDESC/CCLX/blob/master/parameters/lsst_desc_parameters.yaml
    alpha: 0.68  # power law index in the exponent (check eq. 5 in the SRD paper)
    z_0: 0.11  # pivot redshift (check eq. 5 in the SRD paper)
    """
    dz = z[1]-z[0]
    nz = z**2* np.exp(-(z/z0)**alpha)
    # normalize
    nz = nz/sum(nz)/dz
    return nz


def compute_equal_number_bounds(redshift_range, redshift_distribution, n_bins):
    """
    Determines the redshift values that divide the distribution into bins
    with an equal number of galaxies.

    Arguments:
        redshift_range (array): an array of redshift values
        redshift_distribution (array): the corresponding redshift distribution defined over redshift_range
        n_bins (int): the number of tomographic bins

    Returns:
        An array of redshift values that are the boundaries of the bins.

    Note: set range limits to [0,3.5] c.f.CCLX
    """

    # Calculate the cumulative distribution
    cumulative_distribution = cumulative_trapezoid(redshift_distribution, redshift_range, initial=0)
    total_galaxies = cumulative_distribution[-1]

    # Find the bin edges
    bin_edges = []
    for i in range(1, n_bins):
        fraction = i / n_bins * total_galaxies
        # Find the redshift value where the cumulative distribution crosses this fraction
        bin_edge = np.interp(fraction, cumulative_distribution, redshift_range)
        bin_edges.append(bin_edge)

    return [redshift_range[0]] + bin_edges + [redshift_range[-1]]


def true_redshift_distribution(upper_edge, lower_edge, variance, bias, redshift_range, redshift_distribution):
    """A function that returns the true redshift distribution of a galaxy sample.
     The true distribution of galaxies is defined as a convolution of an overall galaxy redshift distribution and
     a probability distribution p(z_{ph}|z)  at a given z (z_{ph} is a photometric distribution at a given z).
     Overall galaxy redshift distribution is a Smail type distribution (n(z) = (z/z_0)^alpha exp[-(z/z_0)^beta]).
     The true distribution defined here is following Ma, Hu & Huterer 2018
      (see https://arxiv.org/abs/astro-ph/0506614 eq. 6).

       Arguments:
           upper_edge (float): upper edge of the redshift bin
           lower_edge (float): lower edge of the redshift bin
           variance (float): variance of the photometric distribution
           bias (float): bias of the photometric distribution
        Returns:
            true_redshift_distribution (array): true redshift distribution of a galaxy sample"""
    # Calculate the scatter
    scatter = variance * (1 + redshift_range)
    # Calculate the upper and lower limits of the integral
    lower_limit = (upper_edge - redshift_range + bias) / np.sqrt(2) / scatter
    upper_limit = (lower_edge - redshift_range + bias) / np.sqrt(2) / scatter

    # Calculate the true redshift distribution
    true_redshift_distribution = 0.5 * np.array(redshift_distribution) * (erf(upper_limit) - erf(lower_limit))

    return true_redshift_distribution


def get_pix_area_one_file(fin):
    """
    Get the total sky area of a galaxy file (nside=8)
    """
    ra = (fin[1].data['RA'])[::100]
    dec = (fin[1].data['DEC'])[::100]
    nside=8
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    upix, counts = np.unique(pix, return_counts=True)
    Npix = len(upix)

    # Double check if counts are similar, 
    # so we don't have pix split in half et.c
    print(counts)

    area = hp.nside2pixarea(nside, degrees = True)*Npix # deg^2
    area = area * 60**2 # arcmin^2
    return area

def match_srd_ngal_one_file(fin, Ngal, srd_ngal=5.4):
    """
    Given a galaxy catalogue file (pixelized to nside=8),
    compute the 
    and  compute the number density
    and produce fractional difference with srd_ngal
    srd_ngal: unit: arcmin^-2, default is Y10 *single* source bin
    """
    area = get_pix_area_one_file(fin)
    ngal = Ngal/area
    frac = srd_ngal/ngal
    return ngal, frac


def dump_save(stuff,filename):
    '''This saves the dictionary and loads it from appropriate files'''
    with open(filename,'wb') as fout:
        pickle.dump(stuff,fout,pickle.HIGHEST_PROTOCOL)
        #json.dump(self.impute, fout, sort_keys=True, indent=3)
    print('written impute ditionary:',filename)
    return 0
def dump_load(filename):
    with open(filename,'rb') as fin:
        stuff=pickle.load(fin, encoding='latin1')
        #self.impute = json.load(fin)
    #print('loaded impute ditionary:',filename)
    return stuff


def get_w_comb(w_thetasplit, njn, Ntheta, Nbins, thetas, alpha, theta_mask):
        # do the selection:
        mask_min = theta_mask[0]
        mask_max = theta_mask[1]
        selind = np.where((thetas>=mask_min)&(thetas<=mask_max))[0][:-1]
            
        Theta_bincen = (thetas[1:] + thetas[:-1])/2.
        dTheta=np.array([thetas[i+1]-thetas[i] for i in range(Ntheta)])
        
        w_comb_jk = np.zeros((njn,Nbins))
        for jk in range(njn):
            data_to_get = w_thetasplit[jk, :].reshape((Ntheta,Nbins))
            for ii in range(Nbins):
                denom = sum(Theta_bincen**alpha*dTheta)
                w_comb_jk[jk, ii] = sum(data_to_get[selind,ii]*Theta_bincen[selind]**alpha*dTheta[selind])/denom
        mean = np.mean(w_comb_jk, axis=0)
        std = np.std(w_comb_jk, axis=0)*np.sqrt(njn)
        w_comb = np.c_[mean,std]
        w_comb = np.c_[w_comb, w_comb_jk.T]
        
        return w_comb
