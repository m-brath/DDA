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
import os
import adda_control_functions as acf
#==============================================================================
#  constants
#==============================================================================


#Speed of light
c0=299792458 #[m/s]

#number of cores
ncores=32

#in how many chunks should the calculation be splitted
num_chunks=10

#maximum job length
duration=1.1/ncores; #[h]

#==============================================================================
# Names/paths
#==============================================================================

#where am I?
hereiam=os.getcwd()

if hereiam.startswith('/mnt/lustre01/'): #we are on mistral

    prefix='/mnt/lustre01/work/um0878/users/mbrath/'

    calc_type='mistral'

else:
    prefix='/scratch/uni/u237/users/mbrath/'

    calc_type='thunder'
    
#calc_type='noqueue'


#scat_params_path
scat_param_path=prefix+'adda/adda_python_scripts/scat_params.dat'

#alldir_params_path
alldir_param_path=prefix+'adda/adda_python_scripts/alldir_params.dat'

#avg_params_path
avg_param_path=prefix+'adda/adda_shape_files/shape_data/avg_params_eps2.dat'

#main path
main_path=prefix+'adda_results/'

#shape name
shape_name='ExampleAggregatesRnd'

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
batch_time=[3600]*len(Sizes)#[seconds]


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


## create list
liste=np.ndarray((8,len(shapelist)*len(frequencies)*len(temperatures)),dtype=object)

idx=-1

for i in range(0,len(shapelist)):
    for j in range(0,len(frequencies)):
        for k in range(0,len(temperatures)):
            
            idx=idx+1
            
            ref_idx=np.sqrt(acf.eps_ice_matzler06(frequencies[j],temperatures[k])) #permittivity of Ice
            temp=np.ndarray((1,1))
                        
            
            liste[0,idx]=[shape_name]
            
            temp[0,0]=Sizes[i]/2/1e6
            liste[1,idx]=temp.copy() #radius in list not diameter
            
            temp[0,0]=eps
            liste[2,idx]=temp.copy()
            
            temp[0,0]=np.real(ref_idx)
            liste[3,idx]=temp.copy()
            
            temp[0,0]=np.imag(ref_idx)
            liste[4,idx]=temp.copy()
            
            temp[0,0]=c0/frequencies[j]
            liste[5,idx]=temp.copy()
            
            temp_str=str.split(shapelist[i],sep='/')            
            liste[6,idx]=[temp_str[-1]]
            
            temp[0,0]=batch_time[i]
            liste[7,idx]=temp.copy()


#==============================================================================
#  chunking
#==============================================================================

#batch time

time = batch_time#[i[0,0] for i in batch_time]

chunks=acf.chunking(time,ncores,duration)

#==============================================================================
# Adda inputs
#==============================================================================


##scat-output option
#scat_output='-store_scat_grid'
#scat_input='-scat_grid_inp'
scat_mat='-scat_matr'
scat_mat_type='both'
Csca_output='-Csca'
alldir_input='-alldir_inp'
#dpl_input='-dpl'
#dpl_str='10'
orient_input=['-orient','avg']



#Scattering output...
#scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type,
#                Csca_output,alldir_input,alldir_param_path]+orient_input+[avg_param_path]
scatoutput_str=[scat_mat,scat_mat_type]+orient_input+[avg_param_path]


#Adda's static command line options
static_clo=shape_str+scatoutput_str

