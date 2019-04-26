# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:27:45 2015

@author: u242031
"""

#==============================================================================
# import modules
#==============================================================================


import numpy as np
#import scipy.io as sio
import glob
import socket

#==============================================================================
#  constants
#==============================================================================


#Speed of light
c0=299792458 #[m/s]

#number of cores
ncores=16


#==============================================================================
# Names/paths
#==============================================================================

#how to calculate
calc_type='queue'

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
avg_param_path=prefix+'adda/adda_aux_input/adda_files_hexPlateAggregates2/avg_params.dat'

#main path
main_path=prefix+'adda_results/'

#shape folder_name
shape_foldername='adda_files_hexPlateAggregates2'

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
shapelist = glob.glob(shape_path+"*.adda")
shapelist.sort()

#get shape size
Dmax=np.zeros([np.size(shapelist)])
for i in range(np.size(shapelist)):

    idxs=shapelist[i].find('dmax')+4
    idxe=shapelist[i].find('um.adda')

    Dmax[i]=float(shapelist[i][idxs:idxe])

shapelist_xml = glob.glob(shape_path+"*.xml")
shapelist_xml.sort()

Sizes=np.zeros([np.size(shapelist_xml)])
for i in range(np.size(shapelist_xml)):

    fh=open(shapelist_xml[i])
    lines=fh.readlines()
    Sizes[i]=float(lines[19][0:-1])*1e6

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
#  Input parameters
#==============================================================================

#frequencies
sub2=np.array([0.98,1.02])
sub3=np.array([0.96,1.,1.04])
sub4=np.array([0.94,0.98,1.02,1.06])
sub5=np.array([0.92,0.96,1.,1.04,1.08])

frequencies=np.hstack((880*sub2,664*sub2,448*sub2,325.15*sub3,243*sub2,183.15*sub4,
                    154.8*sub3,118.75*sub4,90.5*sub3,54*sub5,31.4*sub2,
                    23.8*sub2))*1e9
frequencies.sort()


#frequencies=np.array([24e9,450e9,900e9])#np.logspace(10.,12.,num=10)#[Hz]

#Temperature
temperatures=np.array([253.])#np.array([203.,233.,253.,273.])

#Number of particle tilt orientations (Euler angle beta)
number_of_betas=19

#Incidence angles
incidence_angles=np.linspace(0.,90.,num=10)

#Limits and break criteria for romberg integration
limit_alpha=12
limit_gamma=12
delta=1e-2

#estimated time for batch-job
batch_time=1. #[minutes]


###############################################################################
######## FOR TESTING #######################FOR TESTING #######################
test_index=5
shapelist=shapelist[0:test_index]
shapelist_xml=shapelist_xml[0:test_index]
Sizes=Sizes[0:test_index]

frequencies=frequencies[0:3]
###############################################################################
###############################################################################



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

