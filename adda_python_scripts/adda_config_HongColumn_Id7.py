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

else:
    prefix='/scratch/uni/u237/users/mbrath/'
    prefix_data='/scratch/uni/u237/user_data/mbrath/'
    
    calc_type='thunder'
    
    #number of cores
    ncores=16

    
    
#ramdrive
if calc_type=='mistral':
    ramdrive="/dev/shm/"
else:
    ramdrive="/dev/shm/u242031/"    


#scat_params_path
scat_param_path=prefix+'adda/adda_python_scripts/scat_params.dat'

#alldir_params_path
alldir_param_path=prefix+'adda/adda_python_scripts/alldir_params.dat'

#avg_params_path
#avg_param_path=prefix+'adda/adda_aux_input/EvanSnowflake_aggregate/avg_params.dat'

#main path
main_path=prefix_data+'adda_results/'

#shape name
shape_name='HongColumn_Id7'

#shape folder_name
shape_foldername=shape_name

#shape mainpath
shape_mainpath=prefix+'adda/adda_shape_files/'

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

logic = (un_indices % 2) == 0
logic=~logic

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



frequencies=np.array([5,])*1e9
# frequencies=np.array([1, 1.4, 3, 5, 7, 9, 10, 10.65, 13.4, 15, 18.6, 24, 31.3,
# 31.5, 35.6, 50.1, 57.6, 88.8, 94.1, 115.3, 122.2, 164.1,
# 166.9, 175.3, 191.3, 228, 247.2, 314.2, 336.1, 439.3,
# 456.7, 657.3, 670.7, 862.4, 886.4])*1e9




#Temperature
temperatures=np.array([230.,270.])#np.array([203.,233.,253.,273.])
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
limit_sph=3

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
scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type,
                Csca_output,alldir_input,alldir_param_path]


#Adda's static command line options
static_clo=shape_str+scatoutput_str


