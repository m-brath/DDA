#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:58:25 2015

@author: u242031
"""
#==============================================================================
# import modules
#==============================================================================


import glob
import os
import numpy as np
import typhon as ty
import socket
import subprocess as sub
#import pdb

#==============================================================================
#  some basics
#==============================================================================


#where am I?
hereiam=socket.gethostname()

if hereiam[0:6]=='mlogin' or hereiam[0:7]=='mistral' or hereiam[0:2]=='m2': #we are on mistral

    prefix='/work/um0878/users/mbrath/'
    prefix_data='/work/um0878/user_data/mbrath'

    calc_type='mistral'
    ncores=18
    Partition='compute2'


else:
    prefix='/scratch/uni/u237/users/mbrath/'
    prefix_data='/scratch/uni/u237/user_data/mbrath/'

    calc_type='thunder'
    ncores=16


#path of the configfile folder
script_path=prefix+'/adda/adda_python_scripts/'

#change directory for not putting all the trash in my controlfile folder
temporary_folder=prefix_data+'/adda_results/temp'


#==============================================================================
# configurations
#==============================================================================

#beta angles
p_orientation_beta=np.linspace(0,180,num=19)

#incidence angles
incidence_angles=np.linspace(0,180.,num=19)

#create regular grid
regular_phasematrix=True

#export to netcdf
export_netdf=False

#grid size
grid_size=181

#basic orientation of particle
alpha_0=0
beta_0=0
gamma_0=0

#use parallel module
parallel=True

#accuracy of spherical harmonics representation of phase matrix
eps_phase=0.001


#crystal_type='SingleCrystal'
crystal_type='Aggregate'

pristine_type='Pristine'

orientation_type='Azimuthally_random'

medium='ice'

queueing=True
#==============================================================================
# paths
#==============================================================================

#raw data path (ADDA results)
#rawdatapath=prefix_data+'/adda_results/LiuSectorSnowflake_Id3/'
#rawdatapath=prefix_data+'/adda_results/IconSnow_Id28/'
rawdatapath=prefix_data+'/adda_results/HexPlaAggCrystal_Id20/'
#rawdatapath=prefix_data+'/adda_results/HongPlate_Id9/'
#rawdatapath=prefix_data+'/adda_results/HongColumn_Id7/'

#path for temp results, important if phase matrix on regular grid is wanted  for
#   comparisons etc...

if regular_phasematrix==True:

    avgdatapath=rawdatapath[0:-1]+'_horizavg_test/'
else:
    avgdatapath=None

database_path=prefix_data+'/adda_results/SSD/'


#==============================================================================
#  make config variable
#==============================================================================

config={}
config['p_orientation_beta']=p_orientation_beta
config['incidence_angles']=incidence_angles
config['regular_phasematrix']=regular_phasematrix
config['grid_size']=grid_size
config['alpha_0']=alpha_0
config['beta_0']=beta_0
config['gamma_0']=gamma_0
config['rawdatapath']=rawdatapath
config['avgdatapath']=avgdatapath
config['export_netdf']=export_netdf
config['database_path']=database_path
config['ncores']=ncores
config['parallel']=parallel
config['eps_phase']=eps_phase
config['crystal_type']=crystal_type
config['pristine_type']=pristine_type
config['orientation_type']=orientation_type

#==============================================================================
# the processing
#==============================================================================



if __name__ == "__main__":

    #change directory for not putting all the trash in my controlfile folder
    os.chdir(temporary_folder)

    #look after the adda output
    folderlist = glob.glob(rawdatapath+'*_deq*')
    folderlist.sort()

    counter=-1

    #Sizes and frequencies
    D=np.zeros(len(folderlist))*np.NaN
    F=np.zeros(len(folderlist))*np.NaN

    SizeAndFreqs={}
    SizeAndFreqs['Frequency']=F
    SizeAndFreqs['Size']=D

    Units={}
    Units['Frequency']='GHz'
    Units['Size']='um'


    i=-1
    for folder in folderlist:

        counter=counter+1
        folder=os.path.basename(folder)

        i=i+1

        #get size D  and frequency F from filename
        dummy=folder.split('_')
        SizeAndFreqs['Size'][i]=float(dummy[2][3:-2])
        SizeAndFreqs['Frequency'][i]=float(dummy[3][4:-3])

        print(folder+' ('+str(counter)+')')

    print('config:')
    [print(key+': '+str(config[key])) for key in config.keys()]
    print('  ')
    print('======================================================================')
    print('choose data')
    print('type a number within in brackets')
    print('or type a list with numbers that should be processed')
    print('or type an array with numbers that should be processed')
    print('or type "Frequency" if you want the data frequency sorted')
    print('or type "Size" if you want the data frequency sorted')
    print('or type "all" for all data.')
    index = input(' ==>>  ')
    #index='all'



    if index=='all':
        print('whole data set')
        print('======================================================================')
        print('reprocess all or update missing ones')
        print('type "Process" if you want reprocess all')
        print('or type "Update" if you just want to process missing ones')

        if export_netdf==True:
            idx2 = input(' ==>>  ')

            if idx2=='Update':

                missing=np.zeros(len(folderlist),dtype=bool)
                m_cnt=-1

                for drawer in folderlist:

                    m_cnt=m_cnt+1

                    if m_cnt%100==0:
                        print('m_cnt = '+str(m_cnt))

                    particle=os.path.basename(drawer)
                    strings=particle.split('_')

                    shape_name=strings[0]+'_'+strings[1]
                    size_str='Dv'+strings[2][1:]
                    Freq=float(strings[3][4:9])
                    T=float(strings[4][1:-1])

                    FreqTemp_format=('Freq{FRQ:08.3f}GHz_T{TEMPERATURE:03.0f}K.nc').format
                    FreqTemp_strings={'FRQ':Freq,'TEMPERATURE':T }
                    FreqTemp=FreqTemp_format(**FreqTemp_strings)

                    #folder structure of SSDB
                    folder_structure=[orientation_type,medium,crystal_type,
                          pristine_type,shape_name]

                    folder_path=os.path.join(*folder_structure)

                    #datapath
                    storage_path=os.path.join(database_path,folder_path)

                    beta_folders=glob.glob(storage_path+'/*')

                    logic_beta=np.zeros(len(beta_folders),dtype=bool)
                    ind=-1
                    for beta_folder in beta_folders:

                        ind=ind+1

                        helga=glob.glob(beta_folder+'/*'+size_str+'*')

                        if len(helga)==0:

                            logic_beta[ind]=bool(len(helga))

                        else:
#                            horst=glob.glob(helga[0]+'/'+FreqTemp)
#
#                            logic_beta[ind]=bool(len(horst))
#                        
#                            if len(horst)==0:
                            
                            listdir=os.listdir(helga[0])
                            F_helga=np.array([np.ceil(float(i[4:12])*10)/10 for i in listdir])
                            T_helga=np.array([np.round(float(i[17:20])) for i in listdir])
                            
                            logic_F=F_helga==np.ceil(Freq*10)/10
                            logic_T=T_helga==np.round(T)
                            
                            logic_FT=logic_F * logic_T
                            
                            logic_beta[ind]=bool(sum(logic_FT))    

                    #mark if
                    if sum(logic_beta)!=len(p_orientation_beta):
                        print('ploed')

                    missing[m_cnt]=(sum(logic_beta)!=len(p_orientation_beta))

                #adjust folderlist
                folderlist=[folderlist[j] for j in range(len(folderlist)) if missing[j]==True]

                i=-1
                print('particles to be updated...')
                for folder in folderlist:

                    counter=counter+1
                    folder=os.path.basename(folder)

                    i=i+1

                    #get size D  and frequency F from filename
                    dummy=folder.split('_')
                    SizeAndFreqs['Size'][i]=float(dummy[2][3:-2])
                    SizeAndFreqs['Frequency'][i]=float(dummy[3][4:-3])

                    
                    print(folder+' ('+str(counter)+')')
                print('   ')
                print(str(sum(missing))+' particles need to be processed')

        else:
            print('The whole dataset will be (re-)processed')


    elif index!='Frequency' and index!='Size':

        index=eval(index)

        if np.size(index)==1:
            index=[index]

        folderlist=[folderlist[i] for i in index]

    else:

        #==============================================================================
        #   first filter
        #==============================================================================

        #property to sort
        p2s=SizeAndFreqs[index]

        #get the other selection possibility
        not_index=[i for i in SizeAndFreqs.keys() if i!=index][0]

        #make unique
        p2s_unique,indices=np.unique(p2s, return_inverse=True)


        print('  ')
        print(index)
        for i in range(len(p2s_unique)):

            print(str(p2s_unique[i])+Units[index]+' ('+str(i)+')')

        print('======================================================================')
        print('choose data')
        print('type a number within in brackets')
        print('or type a list with numbers that should be processed')
        print('or type an array with numbers that should be processed')
        idx2 = input(' ==>>  ')

        idx2=eval(idx2)

        if np.size(idx2)==1:
            idx2=[idx2]

        #find the desired selections
        sorted_folderlist=[]
        other_property_list=[]
        for i in idx2:



            helga=[folderlist[j] for j in range(len(folderlist)) if indices[j]==i]
            horst=[SizeAndFreqs[not_index][j] for j in range(len(folderlist)) if indices[j]==i]


            sorted_folderlist=sorted_folderlist+helga
            other_property_list=other_property_list+horst

        #==============================================================================
        #   second filter
        #==============================================================================




        #property to sort
        p2s=np.array(other_property_list)

        #make unique
        p2s_unique,indices=np.unique(p2s, return_inverse=True)


        print('  ')
        print(not_index)
        for i in range(len(p2s_unique)):

            print(str(p2s_unique[i])+Units[not_index]+' ('+str(i)+')')

        print('======================================================================')
        print('choose data')
        print('type a number within in brackets')
        print('or type a list with numbers that should be processed')
        print('or type an array with numbers that should be processed')
        print('or type "all" for all data.')
        idx2 = input(' ==>>  ')

        if idx2=='all':
            print('whole data set')
            sorted2_folderlist=sorted_folderlist

        else:
            idx2=eval(idx2)

            if np.size(idx2)==1:
                idx2=[idx2]

            #find the desired selections
            sorted2_folderlist=[]
            for i in idx2:

                helga=[sorted_folderlist[j] for j in range(len(sorted_folderlist)) if indices[j]==i]

                sorted2_folderlist=sorted2_folderlist+helga




        #the selection=========================================================
        folderlist=sorted2_folderlist



    t=ty.utils.Timer().start()

    counter=-1
    for folder in folderlist:
        counter=counter+1

        folder=os.path.basename(folder)

        beta_0t=beta_0

        print(folder)
        print('beta = '+str(beta_0t))

#
        #do the processing using queue

        #python string scanning method, spherical harmonics
        python_str=[script_path+'adda_process_aro_particles_single.py',folder]

        #Command string (queue on mistral)
        if calc_type=='mistral':

            ##hier mistral sbatch_simple
            sbatch_str=['/work/um0878/sw/bin/sbatch_simple_improved',Partition,
                        'ADDA_'+folder,str(ncores),'08:00:00']

        #Command string (queue on thunder)
        else:
            sbatch_str=['/scratch/uni/u237/sw/bin/sbatch_simple','-p', 'uni-compute2',#'-x', 'ctc[132,133,134]',
                'ADDA_'+folder,str(ncores)]


        if queueing==True:
            #queue
            command_str=sbatch_str+python_str
        elif queueing==False:
            #non queue
            command_str=python_str

        print('    ')
        print(' '.join(command_str))
        print('==========================================================')
        print('    ')
        print('    ')

        sub.call(command_str)
