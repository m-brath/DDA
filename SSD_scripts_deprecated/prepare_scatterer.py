#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:45:31 2018

@author: Manfred Brath
"""

import utils
import assp
# import typhon.arts.xml as tax
import numpy as np
import glob
#==============================================================================
#  additional functions
#==============================================================================

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_particle_sizes(orientations_folder_list,sizeparam,sizes):
     #check sizes
    particlelist = glob.glob(orientations_folder_list[0][j]+'/*'+sizeparam+'*')
    particlelist.sort()

    if len(particlelist)==0:
        particlelist = glob.glob(orientations_folder_list[0][j]+'/*D'+sizeparam[1:]+'*')
        particlelist.sort()

    p_sizes=np.ones(len(particlelist))
    for i in range(len(particlelist)):

        particle=utils.os.path.basename(particlelist[i])
        strings=particle.split('_')

        index=np.array([strng.find(sizeparam[1:]) for strng in strings])
        logic_idx=index>=0

        if sizeparam=='dmax' or sizeparam=='dveq':
            unit='um'
        else:
            unit='kg'

        index_unit=np.array([strng.find(unit) for strng in strings])

        p_sizes[i]=np.array([float((size[ixs+3:ixe]))
                        for size,ixs,ixe,lidx in zip(strings,index,index_unit,logic_idx)
                        if lidx==True])*1e-6




    selected_sizes=sizes*np.NaN
    for i in range(len(sizes)):
        selected_sizes[i]=find_nearest(p_sizes, sizes[i])

    selected_sizes=np.unique(selected_sizes)

    return selected_sizes
#==============================================================================
#  configuration
#==============================================================================


# location of SSDB in your system
SSDBpath = '/scratch/u237/data/scattering_databases/arts_ssdb/SSD'



# main_outpath = '/ceph_pool/ec/scratch/u237/user_data/mbrath/adda_results/SSD_XML_MultiFreq/'
# main_outpath = '/ceph_pool/ec/scratch/u237/users/mbrath/arts_sandkiste/doit-hacking/scattering_data/'
main_outpath = '/scratch2/u237/users/mbrath/XX-nadir_radar/scattering_data'
#sdBase = 'LargePlateAggregate_Id20'
#habID = 20 # habit ID
sdBase = 'PlateType1_Id9'
habID = 9 # habit ID
otype= 'TotallyRandom'

sdExt = 'xml'
sdSize = 'dveq'

# properties of particles to extract SSP for
minD = 1e-6 # minimum size to extract
maxD = 1e-3 # maximum size to extract
sizeparam = 'dmax' # size parameter to apply extraction limits on.
                    # available (param [unit]): 'dmax' [m], 'dveq' [m], 'mass' [kg]
sizes=np.logspace(np.log10(minD),np.log10(maxD),10)

#F_range_array=[[150e9,170e9],[225e9,248e9],[650e9,672e9]]
F_range_array=[[5e9,100e9]]



tmin = 0. # minimum temperature to extract [K]
tmax = float("inf") # maximum temperature to extract

#==============================================================================
#  some initialization
#==============================================================================

# Init database
utils.ssdb_init( SSDBpath )


#get orientations
habitfolder=utils.ssdb_habits(habID)
habitfolder = [folder for folder in habitfolder if otype in folder][0]
orientations_folder_list=list(utils.ssdb_particle_folders(habitfolder))

idx=np.argsort(orientations_folder_list[2])
orientations_folder_list=[[folder[j] for j in idx] for folder in orientations_folder_list ]


#==============================================================================
#  let's do the extraction
#==============================================================================


for j in range(len(idx)):
#

    orient=orientations_folder_list[1][j]

    outpath=utils.os.path.join(main_outpath,sdBase,orient)

    #Check if path exists
    # if  not utils.os.path.exists(outpath):
    utils.os.makedirs(outpath,exist_ok=True)


    print('...importing '+orient)

    S=[]
    M=[]

    #get particle sizes
    selected_sizes=get_particle_sizes(orientations_folder_list,sizeparam,sizes)

    for i in range(len(selected_sizes)):


        minD=selected_sizes[i]*0.99
        maxD=selected_sizes[i]*1.01

        print('   ')
        print('min(D)='+str(minD))
        print('max(D)='+str(maxD))

        freq_flag=False

        for nu in range(len(F_range_array)):

            f_range=F_range_array[nu]
            print(f_range)

            if not freq_flag:
                # Import data, with some cropping in size and freq
                Stemp,Mtemp = assp.assp_import_ssdb( habID, orient, allow_nodata=False,
                                             size_range=[minD, maxD], size_type=sizeparam,
                                             freq_range=f_range,
                                             temp_range=[tmin, tmax] )
                freq_flag=True


            else:
                # Import data, with some cropping in size and freq
                Stemp2,dumpIt = assp.assp_import_ssdb( habID, orient, allow_nodata=False,
                                             size_range=[minD, maxD], size_type=sizeparam,
                                             freq_range=f_range,
                                             temp_range=[tmin, tmax] )
                if len(Stemp2)>0:

                    Stemp[0].pha_mat_data=np.concatenate((Stemp[0].pha_mat_data,Stemp2[0].pha_mat_data))
                    Stemp[0].ext_mat_data=np.concatenate((Stemp[0].ext_mat_data,Stemp2[0].ext_mat_data))
                    Stemp[0].abs_vec_data=np.concatenate((Stemp[0].abs_vec_data,Stemp2[0].abs_vec_data))
                    Stemp[0].f_grid=np.concatenate((Stemp[0].f_grid,Stemp2[0].f_grid))
                else:

                    ##hier noch einfuegen, dass wenn bei gebener groesse und frequenz die daten nicht vorhanden sind
                    ## die daten von der letzen verfuegbaren groesse bei gegebener frequenz genommen wird.

                    print('   ')
                    print('------------------------------------------------------------------------------')
                    print('no scattering data found for size range of ')
                    print('min(D)='+str(minD))
                    print('   to')
                    print('max(D)='+str(maxD))
                    print('and frequency range of')
                    print(f_range)
                    print('So, I will use the data from the previous size for this size')
                    print('------------------------------------------------------------------------------')


                    #take previous size
                    Sprev=S[i-1]
                    logic_freq=(f_range[0]<=Sprev.f_grid) * (Sprev.f_grid <=f_range[1])

                    #put it to the latest size
                    Stemp[0].pha_mat_data=np.concatenate((Stemp[0].pha_mat_data,Sprev.pha_mat_data[logic_freq,:]))
                    Stemp[0].ext_mat_data=np.concatenate((Stemp[0].ext_mat_data,Sprev.ext_mat_data[logic_freq,:]))
                    Stemp[0].abs_vec_data=np.concatenate((Stemp[0].abs_vec_data,Sprev.abs_vec_data[logic_freq,:]))
                    Stemp[0].f_grid=np.concatenate((Stemp[0].f_grid,Sprev.f_grid[logic_freq]))


            filename = '%s/%s' %(outpath,sdBase)
            fmt = 'binary'
            # tax.save(Stemp[0],'%s.scat_data.%s' %(filename,sdExt),format=fmt)
            # tax.save(Mtemp[0],'%s.scat_meta.%s' %(filename,sdExt), format=fmt)



        S.append(Stemp[0])
        M.append(Mtemp[0])

    S_array=[S]
    M_array=[M]

    filename = '%s/%s' %(outpath,sdBase)
    fmt = 'binary'
    # tax.save(S_array,'%s.scat_data.%s' %(filename,sdExt),format=fmt)
    # tax.save(M_array,'%s.scat_meta.%s' %(filename,sdExt), format=fmt)

