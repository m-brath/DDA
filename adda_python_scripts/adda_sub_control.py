#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:58:25 2015

@author: u242031
"""
#==============================================================================
# import modules
#==============================================================================

import subprocess as sub
import glob
import os
import scipy
import numpy as np
import shutil
import sys

import zipfile
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

modes = { zipfile.ZIP_DEFLATED: 'deflated',
          zipfile.ZIP_STORED:   'stored'}

from adda_config import *


#==============================================================================
#  constants/paths
#==============================================================================




#adda command
main_command=['mpiexec','-n',str(ncores),'adda_mpi']

#Basename of simulation run
simbasename=shape+'_test'

#==============================================================================
# Get strings from sys.argv
#==============================================================================

#Temperature index
Tindex=int(sys.argv[1])

#Frequency index
findex=int(sys.argv[2])

#Size index
Dindex=int(sys.argv[3])

#==============================================================================
# storage preperations
#==============================================================================


##storage
storage_path=main_path+subpath

if not os.path.exists(storage_path):
    os.mkdir(storage_path)


#check ramdrive
if not os.path.exists(ramdrive):
    os.makedirs(ramdrive)



#==============================================================================
# start computation
#==============================================================================


#Temperature
T=temperatures[Tindex]

#Wavelength in micrometer
lamb=c0/frequencies[findex]*1e6 #[1e-6 m]

#Size
D=Sizes[Dindex]

#permittivity of Ice
eps=eps_ice_matzler06(frequencies[findex],T)

#refractive Index
m=np.sqrt(eps)


#Name of zipfile
zipname=storage_path+shape+'-'+str(Tindex)+'-'+str(findex)+'-'+str(Dindex)+'_test.zip'

#Clean up, if zip-file is already existing
if os.path.exists(zipname):
    os.remove(zipname)


with zipfile.ZipFile(zipname, mode='w') as zf:

    theta_index=-1
    for theta in incidence_angles:
        theta_index=theta_index+1

        gamma_index=-1
        for gamma in p_orientation_gamma:
            gamma_index=gamma_index+1

            beta_index=-1
            for beta in p_orientation_beta:
                beta_index=beta_index+1

                alpha_index=-1
                for alpha in p_orientation_alpha:
                    alpha_index=alpha_index+1

                    #Make arguments for adda
                    #-----------------------


                    #refractive index
                    m_str=['-m',str(m.real),str(m.imag)]

                    #Wavelength
                    wavelen_str=['-lambda',str(lamb)]

                    #Size
                    size_str=['-size',str(D)]

                    #Particle orientation
                    pori_str=['-orient',str(alpha),str(beta),str(gamma)]

                    #Incidence direction
                    px=[np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)]
                    prop_str=['-prop',str(px[0]),str(px[1]),str(px[2])]

                    #Output directory
                    temp_dir=simbasename
                    path_name=ramdrive+'/'+temp_dir
                    dir_str=['-dir',path_name]

                    #command_str
                    commando=main_command+static_clo+\
                              m_str+wavelen_str+size_str+pori_str+\
                              prop_str+dir_str


                    print('  ')
                    print(commando)

                    sub.call(commando)

                    filelist = glob.glob(path_name+'/*')



                    for filename in filelist:


                        folder_structure=os.path.join('theta_'+str(theta_index),
                        'gamma_'+str(gamma_index),'beta_'+str(beta_index),
                        'alpha_'+str(alpha_index),os.path.basename(filename))

                        zf.write(filename, folder_structure, compress_type=compression)


                    #clean up
                    shutil.rmtree(path_name)







