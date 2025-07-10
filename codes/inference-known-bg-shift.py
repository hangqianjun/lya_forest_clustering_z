"""
Inference for:
- known b_g(z)
- shift model
"""

import sys
import time
import emcee
import numpy as np
from schwimmbad import MPIPool
import numpy as np
import lya_utils as lu
from scipy.integrate import quad
from scipy import interpolate
import argparse

parser = argparse.ArgumentParser(description='Generate pre-computed functions for models.')
parser.add_argument('-sim_num', type=int, default=0, help='Which sim to load in, 0-9')
parser.add_argument('-outroot', type=str, default="", help='Path to the results; directory should stop before the level of simulation runs: /run-[n]/.')
#parser.add_argument('-linear', type=int, default=1, help="0=nonlinear theory, 1=linear theory")
#parser.add_argument('-unknown_bg', type=int, default=0, help="0=bias of unknown sample is known, 1=not unknwon")
#parser.add_argument('-theta', nargs='+', default=[10,30,10], help="Lower and upper limit of the theta angles in arcmin, and how many bins")
parser.add_argument('-model_prefix', type=str, default="", help="Prefix of the model files.")
parser.add_argument('-alpha', type=float, default=0, help="The scaling power to combine different thetas.")
#parser.add_argument('-theta_range', nargs='+', default=[-1,-1], help="additional lower and upper limit applied on theta in order to mask some scales.")
#parser.add_argument('-zbins', nargs='+', default=[2,3,20], help='Zmin, Zmax, Nbin')
parser.add_argument('-data_file', type=str, default="", help="data file to load in; it should contain JK samples of the mock.")
parser.add_argument('-cov_file', type=str, default="", help="covariance file to load in; if not given, will use the date file to compute JK covariance.")
#parser.add_argument('-nz_model', type=str, default="shift", help="Choose n(z) model. Valid inputs are: shift, power_law, GP.")
parser.add_argument('-true_nz_path', type=str, default="", help="If the true nz is needed (e.g. in the shift model), load it from this path.")
parser.add_argument('-yaw_tag', type=str, default="", help="tag for naming the yaw folders; used for different yaw settings such as number of redshift bins. Default is given by the default arguments above.")
#parser.add_argument('-deltaf_weight', type=str, default="", help='not implemented')
#parser.add_argument('-zgrid', nargs='+', default=[1.8,3.0,100], help="zgrid to use for the integral output.")
args = parser.parse_args()

if args.yaw_tag == "":
    yaw_tag = args.yaw_tag
else:
    yaw_tag="-" + args.yaw_tag
    
saveroot = args.outroot + f"run-{args.sim_num}/yaw{yaw_tag}/inference/"
model_root = saveroot + "model/"

# load model files:
zgrid=np.loadtxt(model_root + f"{args.model_prefix}.zgrid.txt")
if args.yaw_tag == "":
    Nzbins=40
else:
    Nzbins=int(args.yaw_tag[:-3])

zbin_edges = np.linspace(2,3,Nzbins+1)
meanz = (zbin_edges[1:] + zbin_edges[:-1])/2.
if Nzbins>=20:
    meanz = meanz[:-1] # let's exclude the last bin in >= 20 bin case as it is not well measured

wsp_int1_thetacomb=np.loadtxt(model_root + f"{args.model_prefix}.thetacomb_alpha-{args.alpha}_int.txt")
    
# load data, covariance:
data_root = saveroot + "../" # assume data is in the same directory
wsp_measure = np.loadtxt(data_root + args.data_file)
data = wsp_measure
if Nzbins>=20:
    data = data[:-1,0]

if args.cov_file == "":
    njn = len(wsp_measure[0,2:])
    cov = np.cov(wsp_measure[:-1,2:])*njn
else:
    cov = np.loadtxt(data_root + args.cov_file)

nz_file = np.loadtxt(args.true_nz_path)
nz_gal_law = interpolate.interp1d(nz_file[:,0],nz_file[:,1],fill_value=0,bounds_error=False)

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

# define model and likelihood functions here:
def log_likelihood(theta, x, y, cov):
    shift_z = theta
    def nz_law_unk(z):
        return nz_gal_law(z+shift_z)
    zlim = [1.8,3.0]
    model = np.zeros(len(x))

    for jj, z_ref in enumerate(x):
        fw = wsp_int1_thetacomb[jj,:]
        w_z_func = interpolate.interp1d(zgrid, fw)
        model[jj] = wsp_full_rsd_intez2(w_z_func, nz_law_unk, zlim)
        
    invcov = np.linalg.inv(cov)
    diff = model-y
    return -0.5*np.dot(diff, np.matmul(invcov, diff))

def log_prior(theta):
    shift_z = theta
    if (shift_z>-0.5) & (shift_z<0.5):
        return 0
    return -np.inf
    
def log_probability(theta):
    x = meanz
    y = data
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, cov)

chain_root = saveroot + "chains/"
filename = chain_root + f"{args.model_prefix}.shift_model.chain.h5"

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    np.random.seed(42)
    initial = np.random.rand(32, 1)-0.5
    nwalkers, ndim = initial.shape
    nsteps = 3000

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)
    start = time.time()
    sampler.run_mcmc(initial, nsteps)
    end = time.time()
    print(end - start)
  
#samples = sampler.get_chain(flat=True)
#tau = sampler.get_autocorr_time()
#print(tau)
