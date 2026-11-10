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
import glob

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

#shape path
shape_path='/scratch/uni/u237/users/mbrath/adda/adda_shape_files/adda_files_664GHz/'

#subfolder
#subpath='ellipsoid_simulations/'
subpath='SectorLikeSnowFlake/'

#ramdrive
ramdrive="/dev/shm/u242031/"

#==============================================================================
# Particle definition
#==============================================================================

#get list with shapes
shapelist = glob.glob(shape_path+"*[!.][!m][!a][!t]")
shapelist.sort()

#get shape size
#Sizes=np.zeros([np.size(shapelist)])
#for i in range(np.size(shapelist)):
#
#    idxs=shapelist[i].find('dmax')+4
#    idxe=shapelist[i].find('um')
#
#    Sizes[i]=float(shapelist[i][idxs:idxe])
#get shape size
Dmax=np.zeros([np.size(shapelist)])
for i in range(np.size(shapelist)):

    idxs=shapelist[i].find('dmax')+4
    idxe=shapelist[i].find('um')

#    if idxe==-1:
#        idxe=shapelist[i].find('um_')

    Dmax[i]=float(shapelist[i][idxs:idxe])

shapelist_xml = glob.glob(shape_path+"*.xml")
shapelist_xml.sort()

Sizes=np.zeros([np.size(shapelist_xml)])
for i in range(np.size(shapelist_xml)):

    fh=open(shapelist_xml[i])
    lines=fh.readlines()
    Sizes[i]=float(lines[3][0:-1])*1e6







#indices=np.argsort(Sizes)
#
#Sizes=Sizes[indices]

#Empty shape string list, important! with empty shape_str, script can recoqnize
#that shape read is to be set.
shape_str=[]
shape='shapefile'


#n-time rotational symmetry of paticle around its own Z-axis
sym_Zobj=6

#mirror symmetry of particle relative to the X-Y-plane of its own coordinate system
sym_XYobj=True

#Half azimuthal averaging, use mirror symmetry
sym_Zlab=True

#==============================================================================
#  Input parameters
#==============================================================================

#frequencies
frequencies=np.array([340e9])#np.logspace(10.,12.,num=10)#[Hz]

#Temperature
temperatures=np.array([250.])#np.linspace(180.,273.15,num=5)

#particle(grid) size
#Sizes=np.array([200.])#np.logspace(1.,3.,num=8) #[1e-6m]

#Number of particle tilt orientations (Euler angle beta)
number_of_betas=19

#Incidence angles
incidence_angles=np.linspace(0.,53.,num=2)                #np.linspace(0.,180.,num=19)

#Limits and break criteria for romberg integration
limit_alpha=12
limit_gamma=12
delta=1e-3

###############################################################################
######## FOR TESTING #######################FOR TESTING #######################
#test_index=range(0,41,2)
shapelist=shapelist[0:15:2]+[shapelist[24]]
shapelist_xml=shapelist_xml[0:15:2]+[shapelist_xml[24]]
Sizes=np.hstack((Sizes[0:15:2],Sizes[24]))#Sizes[0:15:2]
Dmax=np.hstack((Dmax[0:15:2],Dmax[24]))
#shapelist=[shapelist[0]]
#shapelist_xml=[shapelist_xml[0]]
#Sizes=[Sizes[0]]

#frequencies=frequencies[0:3]
###############################################################################
###############################################################################

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
#dpl_input='-dpl'
#dpl_str='10'


#Scattering output...
scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type,
                Csca_output,alldir_input,alldir_param_path]#,dpl_input,dpl_str]


#Adda's static command line options
static_clo=shape_str+scatoutput_str

