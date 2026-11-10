#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:28:58 2017

@author: u242031
"""
import numpy as np
import glob
import os
from netCDF4 import Dataset
from datetime import datetime

from adda_control_functions import loadmat


'''
All functions in this module serve only one goal. The goal is to export the 
azimuthally averaged adda results to the single scattering database with its 
own specific format. The function here made explicitly for this only task

The main function, which should be called is export2SSDBnetCDF.

Furthermore these functions work only in combination with
the python script "definitions_for_python.py

'''

#==============================================================================
# save variable in group
#==============================================================================


def put_variables_in_group0(group0,output_data,SSD_definition,fixed_values,b_i):

    for i in range(len(SSD_definition['variables'])): 
        
#        print(i)
#        print(SSD_definition['variables'][i]+' <--- '+SSD_definition['RAW_variables'][i])            
        
        if len(SSD_definition['RAW_variables'][i])>0:                            

            # Phase 6: exact-backscatter matrix — its own layout: the raw
            # array is (n_theta_inc,4,4) (m^2/sr, converted in
            # rawdata2average), the netCDF variable (phaMatElem,aa_inc,za_inc)
            # with the same row-major 4x4 Stokes enumeration as the phase
            # matrix. Absent in output_data from pre-Phase-6 runs -> the
            # variable keeps its fill value (NaN).
            if (SSD_definition['RAW_variables'][i].find('back_mat_horiz_avg')>-1):

                if 'back_mat_horiz_avg' in output_data:

                    back_raw=output_data['back_mat_horiz_avg'][b_i]

                    #reordering (element index becomes the fastest axis)
                    back_temp=np.transpose(back_raw,(1,2,0))

                    back_temp=back_temp.reshape((16,1,np.shape(back_raw)[0]))

                    group0.variables[SSD_definition['variables'][i]][:]=back_temp

            elif (SSD_definition['RAW_variables'][i].find(SSD_definition['RAW_variables'][8])>-1 or 
                SSD_definition['RAW_variables'][i].find(SSD_definition['RAW_variables'][10])>-1 or 
                SSD_definition['RAW_variables'][i].find(SSD_definition['RAW_variables'][11])>-1):
            
                raw_temp=output_data[SSD_definition['RAW_variables'][i]][b_i]
                
                #reshaping                    
                raw_temp=raw_temp.reshape(np.shape(raw_temp)+(1,))                      
                
                #case: phase matrix
                if (SSD_definition['RAW_variables'][i].find('phase_matrix_sph')>-1):

                    #reordering                                                                     
                    pha_temp=np.transpose(raw_temp,(1,3,0,2))
                    
                    if (SSD_definition['variables'][i].find('real')>-1):
                        
                        group0.variables[SSD_definition['variables'][i]][:]=np.real(pha_temp)
                        
                    elif (SSD_definition['variables'][i].find('imag')>-1):
            
                        group0.variables[SSD_definition['variables'][i]][:]=np.imag(pha_temp)
                
                #case: extinction matrix
                elif (SSD_definition['RAW_variables'][i].find('ext_mat_horiz_avg')>-1):
                    
                    #reordering                                                                     
                    ext_temp=np.transpose(raw_temp,(1,2,3,0))
                    
                    q=np.shape(ext_temp)
                    
                    newshape=(q[0]*q[1],)+q[2:4]
                    
                    ext_temp=ext_temp.reshape(newshape)
                    
                    group0.variables[SSD_definition['variables'][i]][:]=ext_temp[[0,1,11],:,:]
                    
                    
                #case: absorption vector    
                elif (SSD_definition['RAW_variables'][i].find('abs_vector')>-1):     
                                   
                    #reordering                                                                     
                    abs_temp=np.transpose(raw_temp,(1,2,0))
                    
                    group0.variables[SSD_definition['variables'][i]][:]=abs_temp[[0,1],:,:]
                
                
            else:
            
                group0.variables[SSD_definition['variables'][i]][:]=output_data[SSD_definition['RAW_variables'][i]]
        
        else:
        
            group0.variables[SSD_definition['variables'][i]][:]=fixed_values[SSD_definition['variables'][i]]

 
    return group0

#==============================================================================
# put variable in group1
#==============================================================================


def put_variables_in_group1(group1,ShD_definition,meta_data,output_data):
    
    for i in range(len(ShD_definition['variables'])): 
            
            idx=[j for j in meta_data.keys() if ShD_definition['variables'][i]==j]
            
#            print(i)
#            print(ShD_definition['variables'][i])
            
            if len(idx)>0:
                
                if ShD_definition['variables'][i]=='N_dipoles':
                
                    group1.variables[ShD_definition['variables'][i]][:]=(
                            int(meta_data[ShD_definition['variables'][i]]))
                        
                else:
                        
                    group1.variables[ShD_definition['variables'][i]][:]=(
                            meta_data[ShD_definition['variables'][i]])
            
            else:
                
                if ShD_definition['variables'][i]=='refrIndex_real':
                    
                    group1.variables[ShD_definition['variables'][i]][:]=(
                                np.real(output_data['refr_index']))
                    
                elif ShD_definition['variables'][i]=='refrIndex_imag':
                    
                    group1.variables[ShD_definition['variables'][i]][:]=(
                                np.imag(output_data['refr_index']))
                    
                elif ShD_definition['variables'][i]=='dpl':
                    
                    group1.variables[ShD_definition['variables'][i]][:]=(
                                int(1./meta_data['lambda_dpl']))
                    
                else:
                    
                    idx=[j for j in output_data.keys() if (ShD_definition['variables'][i]+'_0')==j]
                    
                    group1.variables[ShD_definition['variables'][i]][:]=(
                            output_data[(ShD_definition['variables'][i]+'_0')])


    return group1


#==============================================================================
#  add attributes to variable
#==============================================================================

def add_attribute_to_variable(group0,SSD_definition,SSD_attributes):

    for variable in group0.variables.keys():
    
        i=[j for j in range(len(SSD_definition['variables'])) if SSD_definition['variables'][j]==variable][0]
    
    
        for attribute in SSD_attributes:
    
            group0.variables[variable].setncattr(attribute,SSD_definition[attribute][i])
            
    
    return group0


#==============================================================================
#  add attribute to group1 
#==============================================================================

def add_attribute_to_group1(group1,ShD_definition,meta_data,refrIndex_model,
                            meta_data_path):

    for i in range(len(ShD_definition['descriptions_general'])):
            
#        print(i)
#        print(ShD_definition['descriptions_general'][i])
        
        idx=[j for j in meta_data.keys() if (
                ShD_definition['meta_description_general'][i]==j)]
#        
        if len(idx)>0:
        
             group1.setncattr(ShD_definition['descriptions_general'][i],
                         meta_data[ShD_definition['meta_description_general'][i]])
             
        else:
                       
            if ShD_definition['descriptions_general'][i]=='refrIndex_model':
            
                group1.setncattr(ShD_definition['descriptions_general'][i],
                                  refrIndex_model)
        
            elif ShD_definition['descriptions_general'][i]=='shape_file':
                
                helga=meta_data_path.split('/')
                
                group1.setncattr(ShD_definition['descriptions_general'][i],
                                  helga[-2]+'/'+helga[-1])

    return group1


#==============================================================================
#  add attribute to group2 
#==============================================================================


def add_attribute_to_group2(group2,CaD_definition,output_data, ac):

    for i in range(len(CaD_definition['descriptions_general'])):
        
        if type(CaD_definition['values'][i])==str:
            group2.setncattr(CaD_definition['descriptions_general'][i],
                             CaD_definition['values'][i])
        
        elif type(CaD_definition['values'][i])==list:                
            
            if CaD_definition['values'][i][0]=='Phase_sph_eps':
                
                horst=output_data['eps_phase']
                
            else:
                # Phase 3: values resolve from the config module; optional
                # entries (older configs) fall back to 'n/a'
                horst=getattr(ac,CaD_definition['values'][i][0],'n/a')
                
            if type(horst)==bool:
                horst=int(horst)
            elif CaD_definition['values'][i][0]=='scat_param_path':
                horst=horst.split('/')[-1]
        
            group2.setncattr(CaD_definition['descriptions_general'][i],
                         horst)

    return group2

#==============================================================================
#  write SSDB-netcdf output
#==============================================================================

def write_SSDBnetCDF(netcdf_filename,b_i,output_data,meta_data,ac,meta_data_path):
    
    from definitions_for_netcdf import (groupnames, SSD_dimnames, SSD_definition,
                                    SSD_attributes, fixed_values,dimensions,
                                    ShD_definition,ShD_attributes,version,
                                    SSD_general_attribute,refrIndex_model,
                                    CaD_definition)
    
    
    #add phase matrix dimension to dimesion array
    dimensions[0]=np.size(output_data['phase_matrix_sph'],-1)
    
    #add  incidence angle dimensions
    dimensions[2]=np.size(output_data['theta_inc'],-1)
    

    with Dataset(netcdf_filename, "w") as rootgrp:

        for group in groupnames:
            rootgrp.createGroup(group)
            
        
        
        #general---------------------------------------------------------------
        
#        print('  ')
#        print('--------------------------------------------------------------')
#        print('    general')
#        print('--------------------------------------------------------------')
#        print('  ')
        
        rootgrp.setncattr('date',datetime.now().date().isoformat())
        rootgrp.setncattr('version',version)
        
        
        #group0----------------------------------------------------------------
            
#        print('  ')
#        print('--------------------------------------------------------------')
#        print('    group0')
#        print('--------------------------------------------------------------')
#        print('  ')

        #get group0
        group0 = rootgrp[groupnames[0]]
        
        #set group0 general attributes
        group0.setncattr('orient_type',SSD_general_attribute)
        
        #make dimensions in group0
        for i in range(len(SSD_dimnames)):

            group0.createDimension(SSD_dimnames[i], dimensions[i])


        #make variables in group0
        for i in range(len(SSD_definition['variables'])):

            if i<2:

                group0.createVariable(SSD_definition['variables'][i],
                                  SSD_definition['dimensions'][i],
                                  fill_value=SSD_definition['fill_values'][i])
            else:
                                
                if i in (8,9,13):
                    do_zip=True                    
                else:
                    do_zip=False
                
                group0.createVariable(SSD_definition['variables'][i],
                                  SSD_definition['datatypes'][i],
                                  dimensions=SSD_definition['dimensions'][i],
                                  fill_value=SSD_definition['fill_values'][i],
                                  zlib=do_zip)



        #put variable into group0
        group0=put_variables_in_group0(group0,output_data,SSD_definition,fixed_values,b_i)


        #add attributes
        group0=add_attribute_to_variable(group0,SSD_definition,SSD_attributes)
        
        #group1----------------------------------------------------------------

#        print('  ')
#        print('--------------------------------------------------------------')
#        print('    group1')
#        print('--------------------------------------------------------------')
#        print('  ')

        #get group1
        group1 = rootgrp[groupnames[1]]
        
        #set attributes of group1        
        group1=add_attribute_to_group1(group1,ShD_definition,meta_data,
                                           refrIndex_model,meta_data_path)
        
        #create variables
        for i in range(len(ShD_definition['variables'])):


            group1.createVariable(ShD_definition['variables'][i],
              ShD_definition['datatypes'][i],dimensions=(),
              fill_value=ShD_definition['fill_values'][i])
            
        #put variable into group1
        group1=put_variables_in_group1(group1,ShD_definition,meta_data,
                                           output_data)    
          
        #add attributes
        group1=add_attribute_to_variable(group1,ShD_definition,
                                             ShD_attributes)

        
        #group2----------------------------------------------------------------

#        print('  ')
#        print('--------------------------------------------------------------')
#        print('    group2')
#        print('--------------------------------------------------------------')
#        print('  ')

        #get group1
        group2 = rootgrp[groupnames[2]]      
        
        
        #set attributes of group2 
        group2=add_attribute_to_group2(group2,CaD_definition,output_data, ac)
    
    print('   ')
    print('written to '+netcdf_filename)


    
#==============================================================================
#   load meta data 
#==============================================================================

def load_shape_meta_data(output_data,ac):

    search_name=('*Dveq{DEQ:05.0f}um*.mat').format
    strings={'DEQ':(output_data['diameter_volume_equ']*1e6)}
    
    shapelist = glob.glob(ac.shape_path+search_name(**strings))
    
    fmin=np.zeros([np.size(shapelist)])
    
    for i in range(np.size(shapelist)):
    
        dummy=loadmat(shapelist[i])
        meta_data=dummy['meta']
        fmin[i]=meta_data['frequency_min'] #[Hz]
    
    df=fmin-output_data['Frequency']
    df[df<0]=np.inf
    
    df_index=np.argmin(df)
    
    #load meta data
    meta_data_path=shapelist[df_index]
    dummy=loadmat(meta_data_path)
    meta_data=dummy['meta']

    return meta_data, meta_data_path



#==============================================================================
#  Export to netcdf
#==============================================================================

def export2SSDBnetCDF(output_data,config):
    ''' export function, which should be called from the function:
        adda_process_single_AroParticle within the python script
        adda_process_aro_particles_single.py
    '''
    
    
    #get some types from config
    crystal_type=config['crystal_type']
    pristine_type=config['pristine_type']
    orientation_type=config['orientation_type']
    database_path=config['database_path']
    
    
    #get shape path from config file
    ac=__import__('adda_config_'+output_data['shape_name'])
    
    
    
    # get meta data of particle   
    [meta_data, meta_data_path]=load_shape_meta_data(output_data,ac)
    
    
    #Make particle size name    
    particle_size=('Dmax{DMAX:05.0f}um_Dveq{DVEQ:05.0f}um_Mass{MASS:1.5e}kg').format
    
    particle_size_strings={'DMAX':(meta_data['diameter_max']*1e6),
             'DVEQ':(meta_data['diameter_vol_eq']*1e6),
             'MASS':(meta_data['mass'])}
    
    
    filelist=[]

    # loop over the titl angles
    for b_i in range(len(output_data['beta'])):
        
        #beta angle
        beta_angle=('AzimuthallyRandom_beta{BETA:03.0f}deg').format
        beta_angle_string={'BETA':output_data['beta'][b_i]}
        
        #folder structure of SSDB
        folder_structure=[orientation_type,meta_data['medium'],crystal_type,
                          pristine_type,output_data['shape_name'],
                          beta_angle(**beta_angle_string),
                          particle_size(**particle_size_strings)]
        
        folder_path=os.path.join(*folder_structure)
        
        #make storage path
        storage_path=os.path.join(database_path,folder_path)
        
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
            
            
        # make filename
        filename=('Freq{FRQ:08.3f}GHz_T{TEMPERATURE:03.0f}K.nc').format
        filename_strings={'FRQ':(output_data['Frequency']*1e-9),
                          'TEMPERATURE':output_data['Temperature'] }        
        
        
        #Fullpath 
        netcdf_filename=os.path.join(storage_path,filename(**filename_strings))
        
        #write netcdf file       
        write_SSDBnetCDF(netcdf_filename,b_i,output_data,meta_data,ac,meta_data_path)
        
        
        filelist=filelist+[netcdf_filename]
        
    return filelist