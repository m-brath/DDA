#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:02:00 2017

@author: u242031
"""

#import subprocess as sub
#import glob
import os
#import scipy as sp
import numpy as np
#import scipy.io as sio
#import sys

#pdb.set_trace()
#from adda_config import *
import adda_control_functions as acf
#import icosphere_grid as ico
import matplotlib.pyplot as plt



#==============================================================================
#  constants
#==============================================================================


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
frequencies=np.array([183e9])#np.logspace(10.,12.,num=10)#[Hz]

#wavelenghts
lamb=c0/frequencies

#Temperature
temperatures=np.array([253.])#np.linspace(180.,273.15,num=5)

#scale factors
x=np.array([0.1])#np.linspace(0.1,10,)

#particle aspect ratio
part_ar=0.5

# particle size in volume equivalent diameter
part_size=lamb*x/np.pi*1e6

#particle(grid) size
Sizes=part_size
Dmax=Sizes*(part_ar)**(1/3)

#Number of particle tilt orientations (Euler angle beta)
number_of_betas=5

#Incidence angles
#incidence_angles=np.linspace(0.,180.,num=10)#np.linspace(0.,90.,num=10)

#Limits and break criteria
delta=0.000001

#limit refinement for spherical approach
limit_sph=2

#precision of adda
eps=3

#save all adda output
ZIPPING=True


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
# input params
#==============================================================================


#Temperature
T=temperatures[0]

#Wavelength in micrometer
lamb=c0/frequencies[0]*1e6 #[1e-6 m]

#Size
D=Sizes[0]

dmax=Dmax[0]


incidence_angles=np.array([[-1/np.sqrt(2),0,1/np.sqrt(2)],
                           [0,-1,0]])

alpha=np.array([0,0])        
beta=np.array([45,45])        
gamma=np.array([0,90])        
        
        

#permittivity of Ice
eps_ice=acf.eps_ice_matzler06(frequencies[0],T)

#refractive Index
m=np.sqrt(eps_ice)

        
#==============================================================================
#         
#==============================================================================

#prepare
zipname_vid=('euler_a{alpha:07.0f}_b{beta:07.0f}_g{gamma:07.0f}').format
zip_fid='_test_sph.zip'


for i in range(0,np.size(incidence_angles,axis=0)):
    
    
    
    
    
    #==============================================================================
    # Particle definition
    #==============================================================================
    
    #Empty shape string list, important! with empty shape_str, script can recoqnize
    #that shape read is to be set.
    
    #change the basic orientation
    alpha_0=alpha[i]
    beta_0=beta[i]
    gamma_0=gamma[i]
    
    #shape
    shape='ellipsoid'
    shape_str=['-shape',shape,'1.0',str(1/part_ar)]
    shape_name=(shape+'_'+shape_str[2]+'_'+shape_str[3]+'_'+
                str(alpha_0)+'-'+str(beta_0)+'-'+str(gamma_0))
    
    
    #==============================================================================
    # Adda inputs
    #==============================================================================
    
    main_command=['mpiexec','-n',str(ncores),'adda_mpi']
    
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
    
    
    
    #==============================================================================
    # storage preperations
    #==============================================================================
    
    #generate storage folders
    filename=(shape_name+'_deq{DEQ:05.0f}um_freq{FRQ:05.1f}GHz_T{TEMPERATURE:04.0f}K').format
    strings={'DEQ':(D), 'FRQ':(c0/(lamb)*1e-3),'TEMPERATURE':T }
    
    storage_path=os.path.join(main_path+subpath,filename(**strings))
    
    if not os.path.exists(storage_path):
            os.makedirs(storage_path)
    
    if not os.path.exists(ramdrive):
            os.makedirs(ramdrive)
    
        
    
    ##Name of zipfile
    zipstrings={'alpha':(alpha[i]), 'beta':(beta[i]), 'gamma':(gamma[i]) }

    zipname=os.path.join(storage_path,shape+'_'+zipname_vid(**zipstrings)+zip_fid)
    

    #define input_data dictionary
    input_data={}

    #wavelength
    input_data['lambda']=lamb

    #Particle Size
    input_data['Size']=D/2.# Sizes is defined as diameter but adda needs radius

    #refractive Index
    input_data['refrative_index']=m

    #incidence angle
    input_data['theta']=incidence_angles[i,:]

    #alpha
    input_data['alpha']=alpha_0

    #beta angle
    input_data['beta']=beta_0

    #gamma angle
    input_data['gamma']=gamma_0

    #precision
    input_data['eps']=eps


    #ramdrive
    input_data['ramdrive']=ramdrive

    #main command for adda
    input_data['main_command']=main_command

    #Adda's static command line options
    input_data['static_clo']=static_clo

    #Name of the zipname
    input_data['zipname']=zipname

    input_data['zipping']=ZIPPING

    #remove existing zip            
    if os.path.exists(zipname):
        os.remove(zipname)                
    
    #Call Adda
    output_data=acf.adda_call(input_data)
    
    mueller_mat=output_data['mueller_array']
    phi=output_data['phis_grid']
    theta=output_data['thetas_grid']

    [PHI,THETA]=np.meshgrid(phi,theta)
    
    fig = plt.figure()
    
    for j in range(1,17):
        
        
        
        plt.subplot(4, 4, j)
        h1=plt.pcolormesh(PHI,THETA,mueller_mat[:,:,j-1])
        
        if j==1:
            plt.ylabel('\theta [deg]')
            
        if j==16: 
            plt.xlabel('\theta [deg]')
        
        cbar = fig.colorbar(h1)
    
#        print('bla')
        
        
   
    qm=np.shape(mueller_mat)    
    rpb=np.pi*2/qm[1]
    binshift=int(np.round( 45*np.pi/180/rpb ))
    mueller_mat_rot=np.roll(mueller_mat,binshift,axis=1)
        
    fig = plt.figure()
    
    for j in range(1,17):
        
        
        
        plt.subplot(4, 4, j)
        h1=plt.pcolormesh(PHI,THETA,mueller_mat_rot[:,:,j-1])
        
        if j==1:
            plt.ylabel('\theta [deg]')
            
        if j==16: 
            plt.xlabel('\theta [deg]')
        
        cbar = fig.colorbar(h1)
    
#        print('bla')        