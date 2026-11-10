#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:58:25 2015

@author: u242031
"""
#==============================================================================
# import modules
#==============================================================================

#import subprocess as sub
import glob
import os
import socket
import sys
import numpy as np
#import pdb
import subprocess as sub

import adda_control_functions as acf
#import shutil
#import adda_config  #=> adda_config.shape


#where am I?
hereiam=socket.gethostname()

if hereiam[0:6]=='mlogin' or hereiam[0:7]=='mistral' or hereiam[0:2]=='m2': #we are on mistral

    prefix='/mnt/lustre01/work/um0878/users/mbrath/'

    calc_type='mistral'

elif sys.platform.startswith('darwin'):

    # local (no-queue) mode (Phase 3): paths relative to this repo
    calc_type='local'
    prefix=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'

else:
    prefix='/scratch/uni/u237/users/mbrath/'
    calc_type='thunder'

#path of the configfile folder
if calc_type=='local':
    script_path=prefix+'adda_python_scripts/'
else:
    script_path=prefix+'adda/adda_python_scripts/'

#change directory for not putting all the trash in my controlfile folder
temporary_folder=prefix+'adda_results/temp'

#==============================================================================
# choose configuration
#==============================================================================

#look after the adda output
filelist = glob.glob(script_path+'/adda_config*.py')
filelist.sort()

counter=-1
for files in filelist:
    counter=counter+1
    dummy=os.path.basename(files)+' ==>> '+str(counter)
    print(dummy)

index = input('input something!: ')

adda_config=filelist[int(index)]
adda_config=os.path.basename(adda_config)
adda_config=adda_config[0:(len(adda_config)-3)]

print(adda_config)


exec('from '+adda_config+' import *')

##==============================================================================
##  constants/paths
##==============================================================================
#
##adda command
#main_command=['mpiexec','-n',str(ncores),'adda_mpi']
#


#==============================================================================
# calculate chunk length
#==============================================================================

#but only if this was not done in the config script


#
if 'chunks' not in locals():

    dummy=np.shape(liste)

    index=range(0,dummy[1])

    ##calculate chunk length
    chnk_len=int(np.ceil(len(index)/num_chunks))
    flag=1

else:

    num_chunks=len(chunks)
    flag=0




#==============================================================================
# let the games begin
#==============================================================================



#change directory for not putting all the trash in my controlfile folder
os.makedirs(temporary_folder, exist_ok=True)
os.chdir(temporary_folder)


#start
for j in range(0,num_chunks):


    if flag==1:

        indices=index[(chnk_len*(j)):(chnk_len*(j+1)):1]
        idx_s=indices[0]
        idx_e=indices[-1]

        print(indices)

    elif flag==0:
        idx_s=j
        idx_e=-999




    python_str=[script_path+'adda_sub_control_randomorient.py',
                adda_config,
                str(idx_s),str(idx_e)]

    if calc_type=='local':
        # local (no-queue) mode: run the chunk directly (Phase 3)
        command_str=python_str
    else:
        sbatch_str=['/scratch/uni/u237/sw/bin/sbatch_simple','-x', 'ctc[132-133]',
                 'ADDA_'+str(j)+'_'+shape_name, str(ncores)]

        command_str=sbatch_str+python_str

    print(command_str)
#    print(python_str)


    if calc_type=='local':
        sub.check_call([sys.executable]+command_str)
    else:
        sub.call(command_str)
#    sub.call(python_str)


#    pdb.set_trace()

# go back to controlfile folder
os.chdir(script_path)




