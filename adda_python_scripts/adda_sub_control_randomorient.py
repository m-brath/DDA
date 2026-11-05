#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==============================================================================
# import modules
#==============================================================================


import os
import sys
import numpy as np
import glob
import shutil

import zipfile
try:
#    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

modes = { zipfile.ZIP_DEFLATED: 'deflated',
          zipfile.ZIP_STORED:   'stored'}

#import pdb


#pdb.set_trace()

import adda_control_functions as acf


#==============================================================================
# Get strings from sys.argv (debug)
#==============================================================================

##Configuration file
#adda_config='adda_config_ExampleAggregates_RndOrient'
#
#
##start index
#idx_s=0
#
##end index
#idx_e=-999
#
#
##from adda_config import *
#print(adda_config)
#
##index = raw_input('input something!: ')
#
#exec('from '+adda_config+' import *')



#==============================================================================
# Get strings from sys.argv
#==============================================================================

#Configuration file
adda_config=sys.argv[1]

#start index
idx_s=int(sys.argv[2])

#end index
idx_e=int(sys.argv[3])


#from adda_config import *
print(adda_config)

#index = raw_input('input something!: ')

exec('from '+adda_config+' import *')



#==============================================================================
#  constants/paths
#==============================================================================

#adda command — config-driven (Phase 3), same contract as
#adda_sub_control_spharmonics.py: adda_binary/adda_launcher/adda_version
if 'adda_binary' not in locals():

    adda_binary='adda_mpi'

if 'adda_launcher' not in locals():

    adda_launcher=['mpiexec','-n',str(ncores)]

main_command=list(adda_launcher)+[adda_binary]


#temp directory for not putting all the trash in my controlfile folder
temporary_folder=os.path.join(prefix,'adda_results','temp')



#==============================================================================
# start computation
#==============================================================================


#change directory for not putting all the trash in my controlfile folder
os.makedirs(temporary_folder, exist_ok=True)
os.chdir(temporary_folder)

if idx_e != -999:

    indices=range(idx_s,idx_e+1)

else:
    indices=chunks[idx_s]

if np.size(indices) ==1 and np.isscalar(indices):
    indices=[indices]

#filename = 'test-{:05.0f}-{:05.0f}K'.format
filename=(shape_name+'_deq{DEQ:05.0f}um_freq{FRQ:05.1f}GHz_T{TEMPERATURE:04.0f}K').format


for idx in indices:

    if np.size(shape_str)==0:
        
        read_shape_str=['-shape','read',shape_path+liste[6,idx][0]]

        static_clo_temp=read_shape_str+static_clo

    else:
        # analytic shapes: the -shape tokens are expected to be part of
        # static_clo already (as in the habit configs where they are added
        # above). Fixes the former NameError for static_clo_temp (:204 uses
        # it unconditionally) — report §1.6-10.
        static_clo_temp=list(static_clo)






    #Wavelength in micrometer
    lamb_temp=liste[5,idx]
    lamb=lamb_temp[0,0]*1e6 #[1e-6 m]



    #refractive Index
    m_imag=liste[4,idx]
    m_real=liste[3,idx]
    m=m_real[0,0]+m_imag[0,0]*1j

    #radius eq. vol.
    req=liste[1,idx]
    req=req[0,0]*1e6 #[1e-6 m]

    #batch time
    batch_time=liste[7,idx]
    batch_time=batch_time[0,0]


    #precision
    eps=liste[2,idx]
    eps=eps[0,0]

    #Temperature (Phase 5, report §1.6-10): the 9-row `liste` format carries an
    #explicit temperature in row 8 (rebuilt TRO configs use it). The legacy
    #8-row format encoded the FLAT INDEX as "T####K" (wrong label in the
    #storage folder name); accepted only with a loud warning.
    if np.shape(liste)[0] >= 9:
        temperature=liste[8,idx]
        temperature=temperature[0,0]
    else:
        temperature=idx
        print('WARNING: liste has no temperature row (8-row legacy format) — '
              'storage folder will encode the flat index as T####K; rebuild '
              'the config with a 9-row liste (row 8 = temperature).')


    #create storage folder
    #For the output, I would prefer if the folders containing the adda output has the following name format (using matlab format):
    #
    #
    #    '%s_deq%05.0fum_freq%05.1fGHz_T%03.0fK'
    #
#    shape, d_max, frequency, temperature

    strings={'DEQ':req, 'FRQ':(c0/(lamb)*1e-3),'TEMPERATURE':temperature }



    storage_path=os.path.join(main_path+subpath,filename(**strings))

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)




    #define input_data dictionary
    input_data={}

    #wavelength
    input_data['lambda']=lamb

    #Particle Size
    input_data['radius']=req

    #refractive Index
    input_data['refractive_index']=m

    #precision
    input_data['eps']=eps

    #ramdrive
    input_data['main_path']=storage_path

    #main command for adda
    input_data['main_command']=main_command

    #Adda's static command line options
    input_data['static_clo']=static_clo_temp

    #how to calculate
    input_data['calc_type']=calc_type

    #maximum time for batch jobs
    input_data['batch_time']=batch_time



    print('   ')
    print('===============================================================')
    print('Index = '+str(idx))
    print('Frequency = '+str(c0/(lamb)*1e-3)+' GHz')
    print('Radius = '+str(req)+' µm')
    print('x = '+str(2*np.pi/(lamb)*req))
    print('===============================================================')
    print('   ')

    #Call Adda inclusive integration over alpha and gamma angles
    acf.adda_call_randomorient(input_data)


    #==============================================================================
    #   store results as one zip-file
    #==============================================================================
    filelist = glob.glob(storage_path+'/*')



    #Name of zipfile
    zipname=storage_path+".zip"


    with  zipfile.ZipFile(zipname,'w') as zf:
        for file in filelist:
            zf.write(file, os.path.basename(file), compress_type=compression)
        print('closing')

    shutil.rmtree(storage_path)
    print(storage_path+": deleted")


















