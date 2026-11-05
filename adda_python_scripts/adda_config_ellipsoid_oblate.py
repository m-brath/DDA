# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:27:45 2015

@author: u242031
"""

#==============================================================================
# import modules
#==============================================================================


import numpy as np
import socket
import os
import sys


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
# Names/paths
#==============================================================================

force_new=True
##Basename of simulation run
#simbasename='sphere_test'


#scat_params_path
scat_param_path=adda_prefix+'adda_python_scripts/scat_params.dat'

#alldir_params_path
alldir_param_path=adda_prefix+'adda_python_scripts/alldir_params.dat'

#main path
main_path=prefix_data+'/adda_results/'

#subfolder
subpath='ellipsoid_spehere_ansatz/rawdata/'


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
    ramdrive="/dev/shm/u242031/"#==============================================================================
#  constants
#==============================================================================


#Speed of light
c0=299792458 #[m/s]

#number of cores
ncores=32

#==============================================================================
#  Input parameters
#==============================================================================

#frequencies
frequencies=np.array([183e9])#np.logspace(10.,12.,num=10)#[Hz]

#wavelenghts
lamb=c0/frequencies

#Temperature
temperatures=np.array([253.])#np.linspace(180.,273.15,num=5)

#scale factors
x=np.array([0.1,1,5])#np.linspace(0.1,10,)

#particle aspect ratio
part_ar=2

# particle size in volume equivalent diameter
#part_size=np.array([100.,200.])*1e-6 # [m]
part_size=lamb*x/np.pi*1e6

#particle(grid) size
#Sizes=np.array([100.])#np.logspace(1.,3.,num=8) #[1e-6m]
Sizes=part_size
Dmax=Sizes*(part_ar)**(1/3)

#Number of particle tilt orientations (Euler angle beta)
number_of_betas=5

#Incidence angles
incidence_angles=np.linspace(0.,180.,num=10)#np.linspace(0.,90.,num=10)

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


#==============================================================================
# Particle definition
#==============================================================================

#Empty shape string list, important! with empty shape_str, script can recoqnize
#that shape read is to be set.

#change the basic orientation
alpha_0=0
beta_0=0
gamma_0=0

#shape
shape='ellipsoid'
shape_str=['-shape',shape,'1.0',str(1/part_ar)]
shape_name=(shape+'_'+shape_str[2]+'_'+shape_str[3]+'_'+
            str(alpha_0)+'-'+str(beta_0)+'-'+str(gamma_0))




#n-time rotational symmetry of paticle around its own Z-axis
sym_Zobj=np.inf

#mirror symmetry of particle relative to the X-Y-plane of its own coordinate system
sym_XYobj=False

#mirror symmetry of particle relative to the X-Z-plane of its own coordinate system
sym_XZobj=False

#mirror symmetry of particle relative to the Y-Z-plane of its own coordinate system
sym_YZobj=False

#Half azimuthal averaging, use mirror symmetry
sym_Zlab=False

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
dpl_input='-dpl'
dpl_str='1000'


#Scattering output...
# -Csca / -alldir_inp dropped (Phase 3): CrossSec*/Csca/mueller_alldir are
# never parsed by this pipeline (ARO extinction comes from the forward
# ampl_scatgrid column); skipping them saves the all-direction integration
# and avoids the 1.4/1.5 flag-combination validation traps (report §1.3)
scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type]


#Adda's static command line options
static_clo=shape_str+scatoutput_str

