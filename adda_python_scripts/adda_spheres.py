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



#==============================================================================
#  constants/paths
#==============================================================================


#number of core
ncores=8

#adda command
main_command=['mpiexec','-n',str(ncores),'adda_mpi']



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

findex=0

#Name of zipfile
zipname=storage_path+'/spheretest.zip'

#Clean up, if zip-file is already existing
if os.path.exists(zipname):
    os.remove(zipname)




with zipfile.ZipFile(zipname, mode='w') as zf:

    for gamma in p_orientation_gamma:

        for beta in p_orientation_beta:

            for alpha in p_orientation_alpha:

                for theta in incidence_angles:








                    #Make arguments for adda
                    #-----------------------



                    #refractive index
                    m_str=['-m',str(m.real),str(m.imag)]

                    #Wavelength
                    wavelen_str=['-lambda',str(lamb)]



                    #Output directory
                    o_dir=simbasename+'.'+str(findex)+'-'+str(Tindex)
                    path_name=ramdrive+'/'+o_dir
                    dir_str=['-dir',path_name]

                    #command_str
                    commando=main_command+shape_str+m_str+size_str\
                            +wavelen_str\
                            +scatoutput_str+dir_str


                    print('  ')
                    print(commando)

                    sp.call(commando)

                    filelist = glob.glob(path_name+'/*')



                    for filename in filelist:

                        zf.write(filename, os.path.join(o_dir,os.path.basename(filename)), compress_type=compression)


                    #clean up
                    shutil.rmtree(path_name)







