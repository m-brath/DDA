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
import adda_control_functions as acf
#==============================================================================
#  constants
#==============================================================================


#Speed of light
c0=299792458 #[m/s]

#number of cores
ncores=32


#==============================================================================
# Names/paths
#==============================================================================

#how to calculate
calc_type='mistral'

#where am I?
hereiam=socket.gethostname()

if hereiam[0:6]=='mlogin': #we are on mistral

    prefix='/mnt/lustre01/work/um0878/users/mbrath/'

    calc_type='mistral'

else:
    prefix='/scratch/uni/u237/users/mbrath/'


#scat_params_path
scat_param_path=prefix+'adda/adda_python_scripts/scat_params.dat'

#alldir_params_path
alldir_param_path=prefix+'adda/adda_python_scripts/alldir_params.dat'

#avg_params_path
#avg_param_path=prefix+'adda/adda_aux_input/EvanSnowflake_aggregate/avg_params.dat'

#main path
main_path=prefix+'adda_results/'

#shape name
shape_name='LiuShape671'

#shape folder_name
shape_foldername=shape_name

#shape mainpath
shape_mainpath=prefix+'adda/adda_shape_files/'

#shape_path
shape_path=shape_mainpath+shape_foldername+'/'

#subfolder
subpath=shape_foldername+'/'

#ramdrive
ramdrive='/dev/shm/u242031/'

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
for i in range(np.size(shapelist_mat)):

    dummy=acf.loadmat(shapelist_mat[i])
    meta_data=dummy['meta']
    Sizes[i]=meta_data['diameter_vol_eq']*1e6 #[µm]
    Dmax[i]=meta_data['diameter_max']*1e6 #[µm]



#Sizes=np.array([Sizes[0]])
#Dmax=np.array([Dmax[0]])

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



frequencies=np.array([183])*1e9#np.logspace(10.,12.,num=10)#[Hz]

#Temperature
temperatures=np.array([253.])#np.array([203.,233.,253.,273.])

#Number of particle tilt orientations (Euler angle beta)
number_of_betas=5

#Incidence angles
incidence_angles=np.linspace(0.,180.,num=10)

#Limits and break criteria for romberg integration
#limit_alpha=12
#limit_gamma=12


#estimated time for batch-job
#batch_time=100. #[minutes]


#limit refinement for spherical approach
limit_sph=4
#precision of adda
eps=3

delta=0.01

#save all adda output
ZIPPING=False

#change the basic orientation
alpha_0=0
beta_0=0
gamma_0=0



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
scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type,
                Csca_output,alldir_input,alldir_param_path]


#Adda's static command line options
static_clo=shape_str+scatoutput_str


