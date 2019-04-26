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
import typhon as ty

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
shape_path='/scratch/uni/u237/users/mbrath/adda/adda_shape_files/shape_data/LiuThickPlate/'

#path to the frequency vector
freq_path='/scratch/uni/u237/users/mbrath/adda/adda_shape_files/DDA_fgrid.xml'

#shape name
shape_name='LiuThickPlate'

#subfolder
subpath=shape_name+'TEST/'

#ramdrive
ramdrive="/dev/shm/u242031/"

#==============================================================================
# Particle definition
#==============================================================================

#get list with shapes
#shapelist = glob.glob(shape_path+"*[!.][!m][!a][!t]")
shapelist = glob.glob(shape_path+"*adda")
shapelist.sort()

#get shape size

shapelist_xml = glob.glob(shape_path+"*.xml")
shapelist_xml.sort()

Sizes=np.zeros([np.size(shapelist_xml)])
Dmax=np.zeros([np.size(shapelist_xml)])
for i in range(np.size(shapelist_xml)):

#    test=ty.arts.xml.load(shapelist_xml[i])

    fh=open(shapelist_xml[i])
    lines=fh.readlines()
    Sizes[i]=float(lines[19][0:-1])*1e6
    Dmax[i]=float(lines[16][0:-1])*1e6






#indices=np.argsort(Sizes)
#
#Sizes=Sizes[indices]

#Empty shape string list, important! with empty shape_str, script can recoqnize
#that shape read is to be set.
shape_str=[]
shape=shape_name


#n-time rotational symmetry of paticle around its own Z-axis
sym_Zobj=2

#mirror symmetry of particle relative to the X-Y-plane of its own coordinate system
sym_XYobj=True

#mirror symmetry of particle relative to the X-Z-plane of its own coordinate system
sym_XZobj=True

#mirror symmetry of particle relative to the Y-Z-plane of its own coordinate system
sym_YZobj=True

#Half azimuthal averaging, use mirror symmetry
sym_Zlab=True



#==============================================================================
#  Input parameters
#==============================================================================

#frequencies
frequencies=ty.arts.xml.load(freq_path)
frequencies.sort()


#Temperature
temperatures=np.linspace(190.,270.,num=3)


#Number of particle tilt orientations (Euler angle beta)
number_of_betas=19

#Incidence angles
incidence_angles=np.linspace(0.,180.,num=5)                #np.linspace(0.,180.,num=19)

#Limits and break criteria for romberg integration
limit_alpha=6
limit_gamma=6
delta=1e-3

#precision of adda
eps=3

###############################################################################
######## FOR TESTING #######################FOR TESTING #######################
#test_index=range(0,41,2)
shapelist=[shapelist[0],shapelist[8],shapelist[54]]
shapelist_xml=[shapelist_xml[0],shapelist_xml[8],shapelist_xml[54]]
Sizes=[Sizes[0],Sizes[8],Sizes[54]]
Dmax=[Dmax[0],Dmax[0],Dmax[54]]

frequencies=[frequencies[4],frequencies[15]]
temperatures=[temperatures[1]]
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

