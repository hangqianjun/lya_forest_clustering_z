"""
This file generates model to use for the inference.
It automatically takes care of the bookkeeping of
the model and data directories, as well as the inference
directories, based on the input parameters.

Need to run this with ccl, 
call it in the notebook with desc-python environment!!
"""

import numpy as np
import pylab as pl
from scipy import interpolate
import pickle
from scipy.integrate import quad, dblquad
import pyccl as ccl
import lya_utils as lu
import argparse
import math


parser = argparse.ArgumentParser(description='Generate pre-computed functions for models.')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
parser.add_argument('-outroot', type=str, default="", help='Path to the results; directory should stop before the level of simulation runs: /run-[n]/.')
parser.add_argument('-linear', type=int, default=1, help="0=nonlinear theory, 1=linear theory")
parser.add_argument('-unknown_bg', type=int, default=0, help="0=bias of unknown sample is known, 1=not unknwon")
parser.add_argument('-theta', nargs='+', default=[10,30,10], help="Lower and upper limit of the theta angles in arcmin, and how many bins")
parser.add_argument('-alpha', type=float, default=0, help="The scaling power to combine different thetas.")
parser.add_argument('-theta_mask', nargs='+', default=[-1,-1], help="additional lower and upper limit applied on theta in order to mask some scales.")
parser.add_argument('-zbins', nargs='+', default=[2,3,20], help='Zmin, Zmax, Nbin')
parser.add_argument('-yaw_tag', type=str, default="", help="tag for naming the yaw folders; used for different yaw settings such as number of redshift bins. Default is given by the default arguments above.")
parser.add_argument('-deltaf_weight', type=str, default="", help='not implemented')
parser.add_argument('-zgrid', nargs='+', default=[1.8,3.0,100], help="zgrid to use for the integral output.")
args = parser.parse_args()


cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.677, A_s=2.1e-9, n_s=0.9675,
                      transfer_function='boltzmann_camb')

def r_3d_rp(rp,z1,z2):
    return ((chi(z1)-chi(z2))**2+(rp)**2)**0.5

def mu_rsd(z1,z2,r):
    return (chi(z1)-chi(z2))/r

def ra_cosmo(z):
    return(ccl.comoving_angular_distance(cosmo, 1/(1+z)))

def chi(z):
    return(ccl.comoving_radial_distance(cosmo, 1/(1+z)))

def omega_m_z(z):
    return ccl.background.omega_x(cosmo, 1/(1+z), 'matter')

# logarithmic growth rate at a given redshift
def f_z(z):
    return ccl.growth_rate(cosmo,1/(1+z))

# ratio of the linear growth factor at z and at z_p
def D_z_p(z,z_p):
    return ccl.growth_factor(cosmo,1/(1+z))/ccl.growth_factor(cosmo,1/(1+z_p))


# now load the linear xi 0, 2, 4 here, construct interpolator as a func of r, z:
#root = "/export/donatello/qhang/nersc/qhang/desi/lya_forest_clustering_z/notebooks/"
root = "../notebooks/"
fin = np.loadtxt(root + "xi_L_0.txt")
r_Mpc, xi_L_0 = fin[:,0], fin[:,1]

fin = np.loadtxt(root + "xi_L_2.txt")
xi_L_2 = fin[:,1]

fin = np.loadtxt(root + "xi_L_4.txt")
xi_L_4 = fin[:,1]

# build interpolation functions
xi_L_0 = interpolate.interp1d(r_Mpc, xi_L_0)
xi_L_2 = interpolate.interp1d(r_Mpc, xi_L_2)
xi_L_4 = interpolate.interp1d(r_Mpc, xi_L_4)

# linear rsd xi:
def xi_rsd_lin(R,z,mu,beta,beta2=None,z_p=3):
    """Note beta is also a function of z, we will compute it later
    """
    if beta2 == None:
        beta2 = np.copy(beta)
        
    xi_0 = (1 + 1/3.*(beta + beta2) + 1/5.*beta*beta2)*xi_L_0(R)
    xi_2 = (2/3.*(beta + beta2) + 4/7.*beta*beta2)*xi_L_2(R)
    xi_4 = 8/35.*beta*beta2*xi_L_4(R)

    P_0 = 1
    P_2 = 1/2.*(3*mu**2-1)
    P_4 = 1/8.*(35*mu**4 - 30*mu**2 + 3)

    return (xi_0*P_0 + xi_2*P_2 + xi_4*P_4)*D_z_p(z,z_p)**2

def xi_rsd_bg_lin(R,z,mu,beta_ref,bias_ref,z_p=3):
    """
    This is the part that gets multiplied by bg(z).
    We will make it explicit in bias and beta
    """

    xi_0 = (1 + 1/3.*beta_ref)*xi_L_0(R)
    xi_2 = (2/3.)*beta_ref*xi_L_2(R)
    
    P_0 = 1
    P_2 = 1/2.*(3*mu**2-1)

    return (xi_0*P_0 + xi_2*P_2)*D_z_p(z,z_p)**2*bias_ref

def xi_rsd_fmu_lin(R,z,mu,f,beta_ref,bias_ref,z_p=3):
    """
    This is the part that's independent of bg(z).
    We will make it explicit in bias and beta
    """
    
    xi_0 = (1/3.*f + 1/5.*beta_ref*f)*xi_L_0(R)
    xi_2 = (2/3.*f + 4/7.*beta_ref*f)*xi_L_2(R)
    xi_4 = (8/35.*beta_ref*f)*xi_L_4(R)

    P_0 = 1
    P_2 = 1/2.*(3*mu**2-1)
    P_4 = 1/8.*(35*mu**4 - 30*mu**2 + 3)

    return (xi_0*P_0 + xi_2*P_2 + xi_4*P_4)*D_z_p(z,z_p)**2*bias_ref

# everything is essentially the same, except for now we use the non-linear power spectrum
Zlist=[2.+k/200 for k in range(200)]
R3d=np.logspace(-1.,2.5,num=40, endpoint=True, base=10.0, dtype=None, axis=0)

# load stuff here:
#root = "/export/donatello/qhang/nersc/qhang/desi/lya_forest_clustering_z/notebooks/"
Xi_3d=np.loadtxt(root + 'xi3d_estimator_z2_3_logrm1_p2.5.txt',skiprows=0,unpack=True)
Xi_3d=np.transpose(Xi_3d)

XI_interp=[]
for iz in range(len(Zlist)):
    XI_interp.append(interpolate.interp1d(R3d, Xi_3d[iz]))

Xi_3d_bar=np.loadtxt(root + 'xi3d_bar_estimator_z2_3_logrm1_p2.5.txt',skiprows=0,unpack=True)
Xi_3d_bar=np.transpose(Xi_3d_bar)

XI_bar_interp=[]
for iz in range(len(Zlist)):
    XI_bar_interp.append(interpolate.interp1d(R3d, Xi_3d_bar[iz]))
    
Xi_3d_barbar=np.loadtxt(root + 'xi3d_barbar_estimator_z2_3_logrm1_p2.5.txt',skiprows=0,unpack=True)
Xi_3d_barbar=np.transpose(Xi_3d_barbar)

XI_barbar_interp=[]
for iz in range(len(Zlist)):
    XI_barbar_interp.append(interpolate.interp1d(R3d, Xi_3d_barbar[iz]))


# now define the correlation function:
def find_z(x):
    arr=Zlist
    ind_max=len(arr)
    left, right = 0, ind_max - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    if left==ind_max:
        return ind_max-1
    return left

def xi_rsd_nlin(R,z,mu,beta,beta2=None):
    """Note beta is also a function of z, we will compute it later
    """
    if beta2 == None:
        beta2 = np.copy(beta)
        
    index_z=find_z(z) # find the best z-index for xi
    
    xi_r=XI_interp[index_z]
    xi_0 = (1 + 1/3.*(beta + beta2) + 1/5.*beta*beta2)*xi_r(R)

    xi_bar_r=XI_bar_interp[index_z]
    xi_2 = (2/3.*(beta + beta2) + 4/7.*beta*beta2)*(xi_r(R) - xi_bar_r(R))

    xi_barbar = XI_barbar_interp[index_z]
    xi_4 = 8/35.*beta*beta2*(xi_r(R) + 5/2.*xi_bar_r(R) - 7/2.*xi_barbar(R))
    
    P_0 = 1
    P_2 = 1/2.*(3*mu**2-1)
    P_4 = 1/8.*(35*mu**4 - 30*mu**2 + 3)

    return xi_0*P_0 + xi_2*P_2 + xi_4*P_4

def xi_rsd_bg_nlin(R,z,mu,beta_ref,bias_ref):
    """
    This is the part that gets multiplied by bg(z).
    We will make it explicit in bias and beta
    """
    
    index_z=find_z(z) # find the best z-index for xi
    xi_r=XI_interp[index_z]
    xi_bar_r=XI_bar_interp[index_z]

    xi_0 = (1 + 1/3.*beta_ref)*xi_r(R)
    xi_2 = (2/3.)*beta_ref*(xi_r(R) - xi_bar_r(R))
    
    P_0 = 1
    P_2 = 1/2.*(3*mu**2-1)

    return (xi_0*P_0 + xi_2*P_2)*bias_ref

def xi_rsd_fmu_nlin(R,z,mu,f,beta_ref,bias_ref):
    """
    This is the part that's independent of bg(z).
    We will make it explicit in bias and beta
    """
    
    index_z=find_z(z) # find the best z-index for xi
    xi_r=XI_interp[index_z]
    xi_bar_r=XI_bar_interp[index_z]
    xi_barbar = XI_barbar_interp[index_z]
    
    xi_0 = (1/3.*f + 1/5.*beta_ref*f)*xi_r(R)
    xi_2 = (2/3.*f + 4/7.*beta_ref*f)*(xi_r(R) - xi_bar_r(R))
    xi_4 = (8/35.*beta_ref*f)*(xi_r(R) + 5/2.*xi_bar_r(R) - 7/2.*xi_barbar(R))

    P_0 = 1
    P_2 = 1/2.*(3*mu**2-1)
    P_4 = 1/8.*(35*mu**4 - 30*mu**2 + 3)

    return (xi_0*P_0 + xi_2*P_2 + xi_4*P_4)*bias_ref


# this need to compute for each theta and each z centre
def wsp_full_rsd_intz1(zgrid, nz_law_ref, zlim, theta_arcmin, 
                           beta_law_ref, beta_law_unk, bias_law_ref, bias_law_unk, linear=True):
    
    """
    This function returns the xi_RSD integrated over the reference redshift distribution, 
    at a fixed anglar scale, and at a fixed redshift zgrid for the unknown sample.
    
    Parameters:
    - zgrid: fixed redshift for the unknown sample
    - nz_law_ref: reference redshift distribution function of the reference sample
    - zlim: integration limit for the reference sample, typically bin edges
    - theta_arcmin: angular scale of the pair separation in arcmin
    - beta_law_ref: beta(z) function of the refrence sample
    - beta_law_unk: beta(z, bias) function of the unknown sample, takes also the bias function
    - bias_law_ref: bias(z) function of the refrence sample
    - bias_law_unk: bias(z) function of the unknon sample
    - linear: Boolean, True = linear power spectrum, False = non linear power spectrum
    """
    
    zmin=zlim[0]
    zmax=zlim[1]
    
    theta_deg=theta_arcmin/60
    theta_rad=theta_deg/360*(2*math.pi)
    
    if linear == True:
        xi_func = xi_rsd_lin
        rlim=[0.01,1e4]
    elif linear == False:
        xi_func = xi_rsd_nlin
        rlim=[0.11,300]
    
    def integrate_z2(z2):
        zmean_pair = (zgrid + z2)/2.
        
        rp=theta_rad*ra_cosmo(zmean_pair) #from Mpc to rad, to deg
        r=r_3d_rp(rp,zgrid,z2) # what is the r3d given our redshifts
        mu = mu_rsd(zgrid,z2,r)

        if r<rlim[0]:# not too small...
            return 0
        if r>rlim[1]:# not too large...
            return 0
        
        beta1 = beta_law_unk(zgrid, bias_law_unk)
        beta2 = beta_law_ref(z2)
        nz_ref = nz_law_ref(z2)
        xilin = xi_func(r,zmean_pair,mu,beta1,beta2=beta2)*nz_ref
        return xilin*bias_law_unk(zgrid)*bias_law_ref(z2)
        
    return quad(integrate_z2, zmin, zmax, epsrel=10**(-3),epsabs=10**(-5),limit=100)[0]


# this combines the theta for each z centre
def w_theta_comb(w_theta_z, theta_edges, theta_weight_func, theta_weight_norm=True):
    """
    w_theta_z: 1D or 2D array of w_theta. 1D = fixed z; 2D allows different z values along axis=1.
    theta_egdes: bin edges to evaluate theta (typically lognormal bins)
    theta_weight_func: computes the weights of each theta
    theta_weight_norm: boolean to determine whether normalize the weight function, default is true.
    """
    
    Ntheta=len(theta_edges)-1
    theta_bincen = (theta_edges[1:] + theta_edges[:-1])/2.
    dtheta=np.array([theta_edges[i+1]-theta_edges[i] for i in range(Ntheta)])

    weights = theta_weight_func(theta_bincen)
    
    if theta_weight_norm == True:
        norm = sum(weights*dtheta)
        weights = weights/norm
        
    if len(w_theta_z.shape) == 1:
        w_comb = sum(w_theta_z*weights*dtheta)
        
    else:
        # make sure w_theta_z has theta variation along axis = 0!
        w_comb = np.sum(w_theta_z*weights[:,None]*dtheta[:,None],axis=0)
    
    return w_comb


# this does the integral for the unknown n(z), and is what we will call in MCMC
def wsp_full_rsd_intez2(w_z_func, nz_law_unk, zlim):
    """
    w_z_func: the function that computes the wsp at various z of the unknown sample
    nz_law_unk: unknown redshift distribution function
    zlim: integration limit of the function
    """
    
    zmin = zlim[0]
    zmax = zlim[1]
    
    def integrate_z2(z):
        return nz_law_unk(z)*w_z_func(z)
    
    return quad(integrate_z2, zmin, zmax, epsrel=10**(-3),epsabs=10**(-5),limit=100)[0]


# this need to compute for each theta and each z centre
def wsp_rsd_bg_intz1(zgrid, nz_law_ref, zlim, theta_arcmin, 
                           beta_law_ref, bias_law_ref, linear=True):
    
    """
    This function returns the xi_RSD integrated over the reference redshift distribution, 
    at a fixed anglar scale, and at a fixed redshift zgrid for the unknown sample,
    for the xi_rsd_bg_lin function.
    
    Parameters:
    - zgrid: fixed redshift for the unknown sample
    - nz_law_ref: reference redshift distribution function of the reference sample
    - zlim: integration limit for the reference sample, typically bin edges
    - theta_arcmin: angular scale of the pair separation in arcmin
    - beta_law_ref: beta(z) function of the refrence sample
    - bias_law_ref: bias(z) function of the refrence sample
    - linear: Boolean, True = linear power spectrum, False = non linear power spectrum
    """
    
    zmin=zlim[0]
    zmax=zlim[1]
    
    theta_deg=theta_arcmin/60
    theta_rad=theta_deg/360*(2*math.pi)
    
    if linear == True:
        xi_func = xi_rsd_bg_lin
        rlim=[0.01,1e4]
    elif linear == False:
        xi_func = xi_rsd_bg_nlin
        rlim=[0.11,300]
    
    def integrate_z2(z2):
        zmean_pair = (zgrid + z2)/2.
        
        rp=theta_rad*ra_cosmo(zmean_pair) #from Mpc to rad, to deg
        r=r_3d_rp(rp,zgrid,z2) # what is the r3d given our redshifts
        mu = mu_rsd(zgrid,z2,r)

        if r<rlim[0]:# not too small...
            return 0
        if r>rlim[1]:# not too large...
            return 0
        
        beta_ref = beta_law_ref(z2)
        bias_ref = bias_law_ref(z2)
        nz_ref = nz_law_ref(z2)
        xilin = xi_func(r,zmean_pair,mu,beta_ref,bias_ref)*nz_ref
        return xilin
        
    return quad(integrate_z2, zmin, zmax, epsrel=10**(-3),epsabs=10**(-5),limit=100)[0]


def wsp_rsd_fmu_intz1(zgrid, nz_law_ref, zlim, theta_arcmin, 
                           beta_law_ref, bias_law_ref, linear=True):
    
    """
    This function returns the xi_RSD integrated over the reference redshift distribution, 
    at a fixed anglar scale, and at a fixed redshift zgrid for the unknown sample,
    for the xi_rsd_bg_lin function.
    
    Parameters:
    - zgrid: fixed redshift for the unknown sample
    - nz_law_ref: reference redshift distribution function of the reference sample
    - zlim: integration limit for the reference sample, typically bin edges
    - theta_arcmin: angular scale of the pair separation in arcmin
    - beta_law_ref: beta(z) function of the refrence sample
    - bias_law_ref: bias(z) function of the refrence sample
    - linear: Boolean, True = linear power spectrum, False = non linear power spectrum
    """
    
    zmin=zlim[0]
    zmax=zlim[1]
    
    theta_deg=theta_arcmin/60
    theta_rad=theta_deg/360*(2*math.pi)
    
    if linear == True:
        xi_func = xi_rsd_fmu_lin
        rlim=[0.01,1e4]
    elif linear == False:
        xi_func = xi_rsd_fmu_nlin
        rlim=[0.11,300]
    
    def integrate_z2(z2):
        zmean_pair = (zgrid + z2)/2.
        
        rp=theta_rad*ra_cosmo(zmean_pair) #from Mpc to rad, to deg
        r=r_3d_rp(rp,zgrid,z2) # what is the r3d given our redshifts
        mu = mu_rsd(zgrid,z2,r)

        if r<rlim[0]:# not too small...
            return 0
        if r>rlim[1]:# not too large...
            return 0
        
        beta_ref = beta_law_ref(z2)
        bias_ref = bias_law_ref(z2)
        nz_ref = nz_law_ref(z2)
        f = ccl.growth_rate(cosmo,1/(1+z2)) # let's call ccl directly
        xilin = xi_func(r,zmean_pair,mu,f,beta_ref,bias_ref)*nz_ref
        return xilin
        
    return quad(integrate_z2, zmin, zmax, epsrel=10**(-3),epsabs=10**(-5),limit=100)[0]


# this does the integral for the unknown n(z) AND bg(z), and is what we will call in MCMC
def wsp_full_bg_rsd_intez2(w_z_func_bg, w_z_func_fmu, nz_law_unk, bias_law_unk, zlim):
    """
    w_z_func: the function that computes the wsp at various z of the unknown sample
    nz_law_unk: unknown redshift distribution function
    zlim: integration limit of the function
    """
    
    zmin = zlim[0]
    zmax = zlim[1]
    
    def integrate_z2(z):
        return nz_law_unk(z)*bias_law_unk(z)*w_z_func_bg(z) + nz_law_unk(z)*w_z_func_fmu(z)
    
    return quad(integrate_z2, zmin, zmax, epsrel=10**(-3),epsabs=10**(-5),limit=100)[0]


# load all the bias laws etc.
#root = "/export/donatello/qhang/nersc/qhang/desi/lya_forest_clustering_z/"
root = "./"
data=np.loadtxt(root + "../bias_dc2.txt")
Z_bias,bias_model=data[:,0],data[:,1]
bias_law_gal=interpolate.interp1d(Z_bias,bias_model,bounds_error=False,fill_value="extrapolate")

# Lya bias model:
def bias_law_lya(z):
    alpha = 2.9
    bias_zref = -0.1352
    zref = 2.4
    #factor = 0.906 # from lya auto-correlation fits
    return bias_zref * ((1 + z)/(1 + zref))**alpha #* factor

def beta_law_gal(z, bias_law):
    return omega_m_z(z)**0.55/bias_law(z)

# wss, fixed beta
def beta_law_lya(z):
    return 1.53

# theta weighting:
def theta_weight_func(theta):
    return theta**args.alpha

#####===========Directories and tags==================

if args.yaw_tag == "":
    yaw_tag = args.yaw_tag
else:
    yaw_tag="-" + args.yaw_tag
    
saveroot = args.outroot + f"run-{args.sim_num}/yaw{yaw_tag}/inference/model/"

if args.linear == 0: 
    linear=False
    lin_tag="nonlinear"
elif args.linear==1:
    linear=True
    lin_tag="linear"

if args.unknown_bg == 0:
    bg_tag = "known_bg"
elif args.unknown_bg == 1:
    bg_tag = "unknown_bg"
    
#####===========CODES START HERE====================

deltaz = (float(args.zbins[1]) - float(args.zbins[0]))/(float(args.zbins[2])) # reference redshift width

bias_law_ref = bias_law_lya
beta_law_ref = beta_law_lya
bias_law_unk = bias_law_gal
beta_law_unk = beta_law_gal

zbin_edges = np.linspace(float(args.zbins[0]),float(args.zbins[1]),int(args.zbins[2])+1)
meanz = (zbin_edges[1:] + zbin_edges[:-1])/2. # reference mean redshift

zgrid = np.linspace(float(args.zgrid[0]), float(args.zgrid[1]), int(args.zgrid[2])) # for the unknown sample

theta_edges= np.logspace(np.log10(float(args.theta[0])), np.log10(float(args.theta[1])), int(args.theta[2])+1)

if int(args.theta_mask[1]) != -1:
    # if set to default, do not apply mask.
    theta_min_mask, theta_max_mask = float(args.theta_mask[0]), float(args.theta_mask[1])
    useind = (theta_edges>=theta_min_mask)&(theta_edges<=theta_max_mask)
    theta_edges = theta_edges[useind]

theta_arcmin = (theta_edges[1:] + theta_edges[:-1])/2. # angular bins

fname_base = saveroot + f"wsp_int_{bg_tag}_{lin_tag}_theta_{round(theta_edges[0])}_{round(theta_edges[-1])}"
# save the results:
np.savetxt(fname_base + ".zgrid.txt", zgrid)
np.savetxt(fname_base + ".theta.txt", theta_arcmin)

if args.unknown_bg == 0:
    # compute integral with known bg
    wsp_int1 = np.zeros((len(zgrid), len(meanz), len(theta_arcmin)))

    
    for jj, z_ref in enumerate(meanz):
        
        print(z_ref)
        
        # for now define it as a top hat, in reality it should be give by n_F(z), close to a top hat.
        def nz_law_ref(z):
            if (z>(z_ref-deltaz/2.))&(z<(z_ref+deltaz/2.)):
                return 1/deltaz
            else:
                return 0
        zlim = [z_ref-deltaz/2., z_ref+deltaz/2.]
        
        for kk, theta in enumerate(theta_arcmin):
        
            for ii, z_unk in enumerate(zgrid):
                wsp_int1[ii, jj, kk] = wsp_full_rsd_intz1(z_unk, nz_law_ref, zlim, theta, 
                                   beta_law_ref, beta_law_unk, bias_law_ref, bias_law_unk, linear=linear)
    
    # combine angular bins:
    wsp_int1_thetacomb = np.zeros((len(meanz), len(zgrid)))

    for jj, z_ref in enumerate(meanz):
        w_theta_z = wsp_int1[:,jj,:].T
        wsp_int1_thetacomb[jj,:] = w_theta_comb(w_theta_z, theta_edges, theta_weight_func, theta_weight_norm=True)
    
    
    ## -> to unpack vstack, reshape results into the original shape
    np.savetxt(fname_base + ".thetasplit_int.txt", np.vstack(wsp_int1))
    np.savetxt(fname_base + f".thetacomb_alpha-{args.alpha}_int.txt", wsp_int1_thetacomb)
    print("saved files with prefix", fname_base)
        
            
elif args.unknown_bg == 1:
    # compute two separate integrals for unknown bg

    wsp_bg_int1 = np.zeros((len(zgrid), len(meanz), len(theta_arcmin)))
    wsp_fmu_int1 = np.zeros((len(zgrid), len(meanz), len(theta_arcmin)))
    
    for jj, z_ref in enumerate(meanz):
        
        print(z_ref)
        
        # for now define it as a top hat, in reality it should be give by n_F(z), close to a top hat.
        def nz_law_ref(z):
            if (z>(z_ref-deltaz/2.))&(z<(z_ref+deltaz/2.)):
                return 1/deltaz
            else:
                return 0
        zlim = [z_ref-deltaz/2., z_ref+deltaz/2.]
        
        for kk, theta in enumerate(theta_arcmin):
        
            for ii, z_unk in enumerate(zgrid):
                
                wsp_bg_int1[ii, jj, kk] = wsp_rsd_bg_intz1(z_unk, nz_law_ref, zlim, theta, 
                                       beta_law_ref, bias_law_ref, linear=linear)
                
                wsp_fmu_int1[ii, jj, kk] = wsp_rsd_fmu_intz1(z_unk, nz_law_ref, zlim, theta, 
                                       beta_law_ref, bias_law_ref, linear=linear)

    # combine angular bins:
    wsp_bg_int1_thetacomb = np.zeros((len(meanz), len(zgrid)))
    wsp_fmu_int1_thetacomb = np.zeros((len(meanz), len(zgrid)))
    
    for jj, z_ref in enumerate(meanz):
        w_theta_z = wsp_bg_int1[:,jj,:].T
        wsp_bg_int1_thetacomb[jj,:] = w_theta_comb(w_theta_z, theta_edges, theta_weight_func, theta_weight_norm=True)
        
        w_theta_z = wsp_fmu_int1[:,jj,:].T
        wsp_fmu_int1_thetacomb[jj,:] = w_theta_comb(w_theta_z, theta_edges, theta_weight_func, theta_weight_norm=True)
    
    # save the results:
    np.savetxt(fname_base + ".thetasplit_bg_int.txt", np.vstack(wsp_bg_int1))
    np.savetxt(fname_base + ".thetasplit_fmu_int.txt", np.vstack(wsp_fmu_int1))
    np.savetxt(fname_base + f".thetacomb_alpha-{args.alpha}_bg_int.txt", wsp_bg_int1_thetacomb)
    np.savetxt(fname_base + f".thetacomb_alpha-{args.alpha}_fmu_int.txt", wsp_fmu_int1_thetacomb)
    print("saved files with prefix", fname_base)








