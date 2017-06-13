#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:58:25 2015

@author: u242031
"""
#==============================================================================
# import modules
#==============================================================================

import subprocess as sp
#import glob
import os
import scipy
import numpy as np
import shutil
#import adda_config  #=> adda_config.shape

from adda_config import *


#==============================================================================
#  constants/paths
#==============================================================================


#Speed of light
c0=299792458 #[m/s]

#frequencies
freq=np.logspace(9.,11.,num=10)#[Hz]

#particle(grid) size
Dx=100. #[1e-6m]

#Refractive index
m=1.25+0.1j

#dipoles per wavelength
dpl=10.

#number of core
ncores=4

#adda command
main_command=['mpiexec','-n',str(ncores),'adda_mpi']

#shape
shape='ellipsoid'

#shape arguments
shape_args=[1.,0.25]

#particle orientation [alpha,beta,gamma]
p_orient=[0.,45.,0.]

#Incidence angle
theta_inc=[22.5]

#Number of polar scattering angles
ntheta=18

#scat-output option
scat_output='-store_scat_grid'
scat_input='-scat_grid_inp'
phi_int='-phi_integr'
Csca_output='-Csca'
alldir_input='-alldir_inp'

#scat_params_path
scat_param_path='/scratch/uni/u237/users/mbrath/adda/adda_python_scripts/scat_params.dat'

#avg_params_path
#avg_param_path='/scratch/uni/u237/users/mbrath/adda/adda_python_scripts/avg_params.dat'

#alldir_params_path
alldir_param_path='/scratch/uni/u237/users/mbrath/adda/adda_python_scripts/alldir_params.dat'

#main path
main_path='/scratch/uni/u237/users/mbrath/adda/adda_results/'

#subfolder
subpath='test_simulation_'+str(1)

#==============================================================================
# start computation
#==============================================================================

#calculate wavelength
lamb=c0/freq #[m]

#converting to micrometer
lamb=lamb*1e6 #[1e-6 m]

#storage
storage_path=main_path+subpath


#if not os.path.exists(storage_path):
#    os.makedirs(storage_path)

if os.path.exists(storage_path):
    shutil.rmtree(storage_path)
#    os.makedirs(storage_path)

#change directory for not putting all the trash in my python script folder
#os.chdir(storage_path)




index=0
for i in lamb[0:2]:

    index=index+1

    #Make arguments for adda
    #-----------------------

    #Shape
    shape_str=['-shape',shape,str(shape_args[0]),str(shape_args[1])]
#    shape_str=['-shape',shape]

    #Size
    size_str=['-size',str(Dx)]

    #refractive index
    m_str=['-m',str(m.real),str(m.imag)]

    #Particle orientation
    pori_str=['-orient',str(p_orient[0]),str(p_orient[1]),str(p_orient[2])]

    #Wavelength
    wavelen_str=['-lambda',str(i)]

    #Scattering output...
    scatoutput_str=[scat_output,scat_input,scat_param_path,
                    Csca_output,alldir_input,alldir_param_path]

    #Incidence direction
    px=[np.sin(theta_inc[0]*np.pi/180),0,np.cos(theta_inc[0]*np.pi/180)]
    prop_str=['-prop',str(px[0]),str(px[1]),str(px[2])]

    #Output directory
    dir_str=['-dir',storage_path]

    #command_str
    commando=main_command+shape_str+m_str+pori_str+size_str\
            +wavelen_str\
            +scatoutput_str+prop_str+dir_str


    print('  ')
    print(commando)

    sp.call(commando)
