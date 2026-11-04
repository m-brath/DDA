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
#import pdb
import socket


#where am I?
hereiam=socket.gethostname()

if hereiam[0:6]=='mlogin' or hereiam[0:7]=='mistral' or hereiam[0:2]=='m2': #we are on mistral

    prefix='/work/um0878/users/mbrath/'
    prefix_data='/work/um0878/user_data/mbrath'

    calc_type='mistral'


else:
    prefix='/scratch/uni/u237/users/mbrath/'
    prefix_data='/scratch/uni/u237/user_data/mbrath/'



#path of the configfile folder
script_path=prefix+'/adda/adda_python_scripts/'

#change directory for not putting all the trash in my controlfile folder
temporary_folder=prefix_data+'/adda_results/temp'

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
size_unique=np.unique(Sizes)

if fmin_exist==True:
    fmin_unique=np.unique(fmin)
    memory=np.zeros(len(size_unique),dtype=bool)

if ('force_new' in locals())==False:
    force_new=False

if ('size_parameter_limit' in locals())==False:
    size_parameter_limit=np.inf

count=0
joblist=[]

Tindex=-1
for T in temperatures:
    Tindex=Tindex+1

    findex=-1
    for f in frequencies:
        findex=findex+1

        #get the maximum size parameter
        size_parameter_unique=f*np.pi*size_unique*1e-6/c0


        idx=np.where(size_parameter_unique<size_parameter_limit)
        if len(idx[0])==len(size_parameter_unique):
            max_size_parameter=np.inf
        else:
            max_size_parameter=size_parameter_unique[idx[0][-1]+1]

        Dindex=-1
        for D in Sizes:
            Dindex=Dindex+1


            size_parameter=f*np.pi*D*1e-6/c0

            print('==========================================================')
            print(adda_config)
            print('Particle size = '+ str(D)+'*10^-6 m')
            print('Size parameter = '+ str(size_parameter))
            print('max. Size parameter = '+str(max_size_parameter))



            #check for size parameters, if size parameter is greater than 10 skip it.
            if size_parameter>max_size_parameter:

                print('    Size_parameter = '+str(size_parameter)+
                      ' > max_size_parameter = '+str(max_size_parameter)  )
                print('    ...continuing with next size')
                continue


            #get the right particle discretization
            if fmin_exist==True:

                #get the number of realisation for that size
                r_idx=np.where(D==Sizes)[0]

                if len(r_idx)>1:

                    fmintemp=fmin[r_idx]

                    df=fmintemp-f
                    df[df<0]=np.inf

                    df_index=np.argmin(df)

                    print('Frequency = '+str(f)+' Hz')
                    print('fmin = '+str(fmin[Dindex])+' Hz')
                    print('f_corr = '+str(fmintemp[df_index])+' Hz')


                    if  fmintemp[df_index]!=fmin[Dindex]:

                        print('wrong particle realisation')
                        print('...continuing with next size')
                        continue


            #check if particle was already simulated
            filename=(shape_name+'_deq{DEQ:05.0f}um_freq{FRQ:05.1f}GHz_T{TEMPERATURE:04.0f}K').format
            strings={'DEQ':(D), 'FRQ':(f*1e-9),'TEMPERATURE':T }

            storage_path=os.path.join(main_path+subpath,filename(**strings))

            if os.path.exists(storage_path):
#                os.makedirs(storage_path)

                # note: '*_mueller_data.xml' was the old (ARTS-XML) artifact name;
                # current runs write '*_mueller_data.pkl' — accept both
                datalist=glob.glob(storage_path+'/*mueller_data.xml')+glob.glob(storage_path+'/*mueller_data.pkl')

                if len(datalist)==1 and force_new==False:

                    print('The Particle was already calculated.')
                    print('So, I skip it!')
                    print('...continuing with next size')
                    continue


            #python string classical method
#            python_str=[script_path+'adda_sub_control_gridref.py',str(Tindex),
#                     str(findex),str(Dindex),adda_config]

            #python string scanning method, regular grids
#            python_str=[script_path+'adda_sub_control_spherical_approach.py',str(Tindex),
#                     str(findex),str(Dindex),adda_config]

            #python string scanning method, spherical harmonics
            python_str=[script_path+'adda_sub_control_spharmonics.py',str(Tindex),
                     str(findex),str(Dindex),adda_config]



            #Command string (queue on mistral)
            if calc_type=='mistral':

                ##hier mistral sbatch_simple
                sbatch_str=['/work/um0878/sw/bin/sbatch_simple_improved',Partition,
                            'ADDA_'+str(Tindex)+'-'+str(findex)+'-'+str(Dindex)+'-'
                            +shape,str(ncores),'08:00:00']

            #Command string (queue on thunder)
            else:
                sbatch_str=['/scratch/uni/u237/sw/bin/sbatch_simple',#'-x', 'ctc13[6-9]',
                    'ADDA_'+str(Tindex)+'-'+str(findex)+'-'+str(Dindex)+'-'
                    +shape,str(ncores)]


            #queue
            command_str=sbatch_str+python_str

            #non queue
            # command_str=python_str

            print('    ')
            print(' '.join(command_str))
            print('==========================================================')
            print('    ')
            print('    ')
            count=count+1
            joblist=joblist+[filename(**strings)]

            sub.call(command_str)

print('  ')
print('Jobs send to queue:')
[print(job) for job in joblist]
print('=======================================================================')
print(str(count)+' jobs send to the queue')

