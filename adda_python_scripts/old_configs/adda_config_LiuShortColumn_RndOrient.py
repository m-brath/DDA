# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:27:45 2015

@author: u242031
"""

#==============================================================================
# import modules
#==============================================================================


import scipy.io as sio
#import glob
import os
import adda_control_functions as acf



#where am I?
hereiam=os.getcwd()

if hereiam.startswith('/mnt/lustre01/'): #we are on mistral

    prefix='/mnt/lustre01/work/um0878/users/mbrath/'

    calc_type='mistral'

else:
    prefix='/scratch/uni/u237/users/mbrath/'

    calc_type='thunder'


#==============================================================================
#  constants
#==============================================================================


#Speed of light
c0=299792458 #[m/s]

#Number of used cores
if calc_type=='mistral':

    ncores=36
    Partition='compute2'

else:
    ncores=16


#in how many chunks should the calculation be splitted
num_chunks=10

#maximum job length
duration=8; #[h]

#==============================================================================
# Names/paths
#==============================================================================


#scat_params_path
scat_param_path=prefix+'adda/adda_python_scripts/scat_params.dat'

#alldir_params_path
alldir_param_path=prefix+'adda/adda_python_scripts/alldir_params.dat'

#avg_params_path
avg_param_path=prefix+'adda/adda_shape_files/shape_data/avg_params_eps2.dat'

#main path
main_path=prefix+'adda_results/'

#shape name
shape_name='LiuShortColumn'

#shape folder_name
shape_foldername=shape_name+'/'

#shape mainpath
shape_mainpath=prefix+'adda/adda_shape_files/shape_data/'

#shape_path
shape_path=shape_mainpath+shape_foldername+'/'

#subfolder
subpath=shape_foldername

#ramdrive
ramdrive='/dev/shm/u242031/'

#==============================================================================
# Particle definition
#==============================================================================

#get from matlab list with the shapes to calculate
dummy=sio.loadmat(shape_mainpath+shape_name+'.cell.mat')
liste=dummy['jobSet_'+shape_name]


#Empty shape string list, important! with empty shape_str, script can recoqnize
#that shape read is to be set.
shape_str=[]
shape='shapefile'


#n-time rotational symmetry of paticle around its own Z-axis
sym_Zobj=1

#mirror symmetry of particle relative to the X-Y-plane of its own coordinate system
sym_XYobj=True

#Half azimuthal averaging, use mirror symmetry
sym_Zlab=True

#==============================================================================
#  chunking
#==============================================================================



#batch time
batch_time=liste[7,:]
time = [i[0,0] for i in batch_time]

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

#scatoutput_str=[orient_input,avg_param_path]

#Adda's static command line options
static_clo=shape_str+scatoutput_str

