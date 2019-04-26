# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:27:45 2015

@author: u242031
"""

#==============================================================================
# import modules
#==============================================================================


import numpy as np
import scipy as sp

#==============================================================================
#  load Gaussian quadrature points
#==============================================================================

#gauss_quad=sp.io.loadmat('/scratch/uni/u237/users/mbrath/adda/'+\
#            'adda_python_scripts/gauss_quadrature_points.mat')

#==============================================================================
#  constants
#==============================================================================


#Speed of light
c0=299792458 #[m/s]

#number of cores
ncores=16

#==============================================================================
#  Input parameters
#==============================================================================

#frequencies
frequencies=np.array([5*183e9])#np.logspace(10.,12.,num=10)#[Hz]

#Temperature
temperatures=np.array([250.])#np.linspace(180.,273.15,num=5)

#particle(grid) size
Sizes=np.array([200.])#np.logspace(1.,3.,num=8) #[1e-6m]

#Number of particle tilt orientations (Euler angle beta)
number_of_betas=5

#Incidence angles
incidence_angles=np.linspace(0.,180.,num=1)+47.                 #np.linspace(0.,180.,num=19)

#Limits and break criteria for romberg integration
limit_alpha=14
limit_gamma=12
delta=1e-9

#==============================================================================
# Particle definition
#==============================================================================


#shape
shape='ellipsoid'
shape_str=['-shape',shape,'1.0','0.666']


#n-time rotational symmetry of paticle around its own Z-axis
sym_Zobj=np.inf

#mirror symmetry of particle relative to the X-Y-plane of its own coordinate system
sym_XYobj=True

#Half azimuthal averaging, use mirror symmetry
sym_Zlab=True


#==============================================================================
# Names/paths
#==============================================================================

#Basename of simulation run
simbasename=shape+'_test'

#scat_params_path
scat_param_path='/scratch/uni/u237/users/mbrath/adda/adda_python_scripts/scat_params.dat'

#alldir_params_path
alldir_param_path='/scratch/uni/u237/users/mbrath/adda/adda_python_scripts/alldir_params.dat'

#main path
main_path='/scratch/uni/u237/users/mbrath/adda/adda_results/'

#subfolder
#subpath='ellipsoid_simulations/'
subpath='prototyping2/'

#ramdrive
ramdrive="/dev/shm/u242031/"+subpath



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
dpl_str='10'


#Scattering output...
scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type,
                Csca_output,alldir_input,alldir_param_path,dpl_input,dpl_str]


#Adda's static command line options
static_clo=shape_str+scatoutput_str

