# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:27:45 2015

@author: u242031
"""

#==============================================================================
# import modules
#==============================================================================


import numpy as np
#import scipy as sp



#==============================================================================
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
x=np.array([0.1,1,10])#np.linspace(0.1,10,)

#particle aspect ratio
part_ar=0.5

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
eps=4

#==============================================================================
# Names/paths
#==============================================================================


##Basename of simulation run
#simbasename='sphere_test'


#scat_params_path
scat_param_path='/scratch/uni/u237/users/mbrath/adda/adda_python_scripts/scat_params.dat'

#alldir_params_path
alldir_param_path='/scratch/uni/u237/users/mbrath/adda/adda_python_scripts/alldir_params.dat'

#main path
main_path='/scratch/uni/u237/users/mbrath/adda_results/'

#subfolder
subpath='ellipsoid_spehere_ansatz/rawdata/'


#ramdrive
ramdrive="/dev/shm/u242031/"

#==============================================================================
# Particle definition
#==============================================================================

#Empty shape string list, important! with empty shape_str, script can recoqnize
#that shape read is to be set.

#change the basic orientation
alpha_0=0
beta_0=90
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
scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type,
                Csca_output,alldir_input,alldir_param_path]#,dpl_input,dpl_str]


#Adda's static command line options
static_clo=shape_str+scatoutput_str

