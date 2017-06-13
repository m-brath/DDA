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

import numpy as np
import pdb
#import shutil
#import adda_config  #=> adda_config.shape


#path of the configfile folder
script_path='/scratch/uni/u237/users/mbrath/adda/adda_python_scripts/'

#change directory for not putting all the trash in my controlfile folder
temporary_folder='/scratch/uni/u237/user_data/mbrath/adda_results/temp'

#==============================================================================
# choose configuration
#==============================================================================

#look after the adda output
filelist = glob.glob(script_path+'/adda_config*.py')

counter=-1
for files in filelist:
    counter=counter+1
    dummy=os.path.basename(files)+' ==>> '+str(counter)
    print(dummy)

index = input('input something!: ')

adda_config=filelist[int(index)]
adda_config=os.path.basename(adda_config)
adda_config=adda_config[0:(len(adda_config)-3)]

#print(adda_config)


exec('from '+adda_config+' import *')
#from adda_config import *

#==============================================================================
# let the games begin
#==============================================================================

#change directory for not putting all the trash in my controlfile folder
os.chdir(temporary_folder)

fmin_exist='fmin' in locals()

if fmin_exist==True:
    fmin_unique=np.unique(fmin)
    size_unique=np.unique(Sizes)
    memory=np.zeros(len(size_unique),dtype=bool)
    
if 'force_new' in locals()==False:
    force_new=False



Tindex=-1
for T in temperatures:
    Tindex=Tindex+1

    findex=-1
    for f in frequencies:
        findex=findex+1


        Dindex=-1
        for D in Sizes:
            Dindex=Dindex+1
            
            print('==========================================================')
            print(adda_config)
            print('Particle size = '+ str(D)+'*10^-6 m')
            
            if fmin_exist==True:
            
                df=fmin_unique-f
                df[df<0]=np.inf
                
                df_index=np.argmin(df)
                
                print('Frequency = '+str(f)+' Hz')
                print('fmin = '+str(fmin[Dindex])+' Hz')
                print('f_corr = '+str(fmin_unique[df_index])+' Hz')
                
                if  fmin_unique[df_index]!=fmin[Dindex]:                    
                    
                    print('wrong particle realisation')
                    print('...continuing with next size')
                    continue
                
                #check if particle was already simulated
                filename=(shape_name+'_deq{DEQ:05.0f}um_freq{FRQ:05.1f}GHz_T{TEMPERATURE:04.0f}K').format
                strings={'DEQ':(D), 'FRQ':(f*1e-9),'TEMPERATURE':T }
                
                storage_path=os.path.join(main_path+subpath,filename(**strings))

                datalist=glob.glob(storage_path+'/*mueller_data.xml.bin')
                
                if len(datalist)==1 and force_new==False:
                    
                    print('The Particle was already calculated.')
                    print('So, I skip it!')                    
                    print('...continuing with next size')
                    continue


            #python string
#            python_str=[script_path+'adda_sub_control_gridref.py',str(Tindex),
#                     str(findex),str(Dindex),adda_config]

            
#            python_str=[script_path+'adda_sub_control_spherical_approach.py',str(Tindex),
#                     str(findex),str(Dindex),adda_config]

            python_str=[script_path+'adda_sub_control_spharmonics.py',str(Tindex),
                     str(findex),str(Dindex),adda_config]



            #Command string (queue)
            sbatch_str=['/scratch/uni/u237/sw/bin/sbatch_simple','-x', 'ctc[132-134]',
                        'ADDA_'+str(Tindex)+'-'+str(findex)+'-'+str(Dindex)+'-'
                        +shape,str(ncores)]

            #queue
            command_str=sbatch_str+python_str

            #non queue
#            command_str=python_str
            
            print('    ')
            print(command_str)
            print('==========================================================')
            print('    ')
            print('    ')
            
#            pdb.set_trace()
            sub.call(command_str)