# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:27:45 2015

@author: u242031
"""

#==============================================================================
# import modules
#==============================================================================


import numpy as np
import scipy.io as sio
import glob
import socket
import os
import sys
import adda_control_functions as acf
#==============================================================================
#  constants
#==============================================================================


#Speed of light
c0=299792458 #[m/s]



#==============================================================================
# Names/paths
#==============================================================================

#how to calculate


#where am I?
hereiam=socket.gethostname()

if hereiam[0:6]=='mlogin' or hereiam[0:7]=='mistral' or hereiam[0:2]=='m2': #we are on mistral

    prefix='/work/um0878/users/mbrath/'
    prefix_data='/work/um0878/user_data/mbrath/'

    calc_type='mistral'
    
    ncores=36
    Partition='compute2'

elif sys.platform.startswith('darwin'):

    # local (no-queue) mode (Phase 3): paths relative to this repo
    _here=os.path.dirname(os.path.abspath(__file__))+'/'
    _repo=os.path.dirname(_here[:-1])+'/'
    prefix=_here
    prefix_data=_repo
    calc_type='local'
    Partition='n/a'
    #number of cores
    ncores=4

else:
    prefix='/scratch/uni/u237/users/mbrath/'
    prefix_data='/scratch/uni/u237/user_data/mbrath/'
    
    calc_type='thunder'
    
    #number of cores
    ncores=32

    
    
#==============================================================================
# ADDA checkout, provenance and binary (Phase 3 - configurable binary)
#==============================================================================

#path of the adda checkout (cluster: <prefix>/adda; local: this repo root)
if calc_type=='local':
    adda_prefix=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
else:
    adda_prefix=prefix+'adda/'

#provenance strings for the SSDB CalculationData group
if calc_type=='mistral':
    cluster_name='DKRZ: mistral-compute2'
elif calc_type=='local':
    cluster_name='local: '+hereiam
else:
    cluster_name='TCUB: thunder'

#ADDA executable and launcher (consumed by the adda_sub_control_*.py scripts;
#they fall back to the historic 1.2 defaults if a config defines neither).
#Env overrides enable cross-version runs without editing configs:
#  ADDA_BINARY=<exe> ADDA_VERSION=<ver> python adda_main_control.py ...
adda_version='1.2'

if calc_type=='mistral':
    adda_binary=adda_prefix+'adda_1.2/src/mpi/adda_mpi'
    adda_launcher=['srun','-n',str(ncores)]
elif calc_type=='local':
    adda_binary=adda_prefix+'adda_1.2/src/seq/adda'
    adda_launcher=[]
else:
    adda_binary='adda_mpi'
    adda_launcher=['mpiexec','-n',str(ncores)]

adda_version=os.environ.get('ADDA_VERSION',adda_version)
adda_binary=os.environ.get('ADDA_BINARY',adda_binary)

#ramdrive
if calc_type=='mistral':
    ramdrive="/dev/shm/"
elif calc_type=='local':
    ramdrive=None      # system temp dir
else:
    ramdrive="/dev/shm/u242031/"#scat_params_path
scat_param_path=adda_prefix+'adda_python_scripts/scat_params.dat'

#alldir_params_path
alldir_param_path=adda_prefix+'adda_python_scripts/alldir_params.dat'

#avg_params_path
#avg_param_path=adda_prefix+'adda_aux_input/EvanSnowflake_aggregate/avg_params.dat'

#main path
main_path=prefix_data+'adda_results/'

#shape name
shape_name='HexPlaAggCrystal_Id20'

#shape folder_name
shape_foldername=shape_name

#shape mainpath
shape_mainpath=adda_prefix+'adda_shape_files/'

#shape_path
shape_path=shape_mainpath+shape_foldername+'/'

#subfolder
subpath=shape_foldername+'/'



#==============================================================================
# Particle definition
#==============================================================================


#get list with shapes
#shapelist = glob.glob(shape_path+"*[!.][!m][!a][!t]")
shapelist = glob.glob(shape_path+"*adda")
shapelist.sort()

#get shape size

shapelist_mat = glob.glob(shape_path+"*.mat")
shapelist_mat.sort()

Sizes=np.zeros([np.size(shapelist_mat)])
Dmax=np.zeros([np.size(shapelist_mat)])
fmin=np.zeros([np.size(shapelist_mat)])

for i in range(np.size(shapelist_mat)):

    dummy=acf.loadmat(shapelist_mat[i])
    meta_data=dummy['meta']
    Sizes[i]=meta_data['diameter_vol_eq']*1e6 #[µm]
    Dmax[i]=meta_data['diameter_max']*1e6 #[µm]
    fmin[i]=meta_data['frequency_min'] #[Hz]



#get rid of every second size
[dummy, un_indices] = np.unique(Dmax, return_inverse=True)

#get every second size
# logic=(un_indices & 1) == 1

logic = (un_indices % 16) == 0

Sizes=Sizes[logic]
Dmax=Dmax[logic]
fmin=fmin[logic]

shapelist_mat=[shapelist_mat[i] for i in range(len(shapelist_mat)) if logic[i]==True]
shapelist=[shapelist[i] for i in range(len(shapelist)) if logic[i] is logic[i]==True]




#Empty shape string list, important! with empty shape_str, script can recoqnize
#that shape read is to be set.
shape_str=[]
shape=shape_name


#n-time rotational symmetry of paticle around its own Z-axis
#sym_Zobj=0

#mirror symmetry of particle relative to the X-Y-plane of its own coordinate system
sym_XYobj=False

#mirror symmetry of particle relative to the X-Z-plane of its own coordinate system
sym_XZobj=False

#mirror symmetry of particle relative to the Y-Z-plane of its own coordinate system
sym_YZobj=False


#==============================================================================
#  Input parameters
#==============================================================================



frequencies=np.array([175.3,])*1e9
# frequencies=np.array([1, 1.4, 3, 5, 7, 9, 10, 10.65, 13.4, 15, 18.6, 24, 31.3,
# 31.5, 35.6, 50.1, 57.6, 88.8, 94.1, 115.3, 122.2, 164.1,
# 166.9, 175.3, 191.3, 228, 247.2, 314.2, 336.1, 439.3,
# 456.7, 657.3, 670.7, 862.4, 886.4])*1e9




#Temperature
temperatures=np.array([230.])#np.array([203.,233.,253.,273.])
#temperatures=np.array([190., 230., 270.])

#Number of particle tilt orientations (Euler angle beta)
number_of_betas=5

#Incidence angles
incidence_angles=np.linspace(0.,180.,num=10)

#estimated time for batch-job
#batch_time=100. #[minutes]

#Limits and break criteria
delta=0.01

#limit refinement for spherical approach
limit_sph=4

#precision of adda
eps=3

#precision of spherical harmonics expansion
eps_sphexp=0.005

#save all adda output
ZIPPING=False

#change the basic orientation
alpha_0=0
beta_0=0
gamma_0=0

#force new calculation
force_new=False

#size parameter limit
size_parameter_limit=10

#==============================================================================
# Adda inputs
#==============================================================================



#scat-output option
scat_output='-store_scat_grid'
scat_input='-scat_grid_inp'
scat_mat='-scat_matr'
scat_mat_type='both'
Csca_output='-Csca'
alldir_input='-alldir_inp'



#Scattering output...
# -Csca / -alldir_inp dropped (Phase 3): CrossSec*/Csca/mueller_alldir are
# never parsed by this pipeline (ARO extinction comes from the forward
# ampl_scatgrid column); skipping them saves the all-direction integration
# and avoids the 1.4/1.5 flag-combination validation traps (report §1.3)
scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type]


#Adda's static command line options
static_clo=shape_str+scatoutput_str


