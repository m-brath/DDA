#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:58:25 2015

@author: u242031
"""
#==============================================================================
# import modules
#==============================================================================


import os
import numpy as np
import adda_orientational_avg_spharmonics_functions as aoa
import scipy.io as sio
import spherical_harmonics_for_adda as spha
import sys
import typhon as ty
import functools
import adda2SSDBnetcdf_functions as a2s

from adda_process_aro_particles import config


# Force flush, otherwise output can disappear from slurm log file
print = functools.partial(print, flush=True)

#==============================================================================
# function definitions
#==============================================================================

def adda_process_single_AroParticle(config,folder):


    #beta angles
    p_orientation_beta=config['p_orientation_beta']

    #incidence angles
    incidence_angles=config['incidence_angles']

    #create regular grid
    regular_phasematrix=config['regular_phasematrix']

    #export netcdf
    export_netdf=config['export_netdf']

    #grid size
    grid_size=config['grid_size']

    alpha_0=config['alpha_0']
    beta_0=config['beta_0']
    gamma_0=config['gamma_0']

    rawdatapath=config['rawdatapath']
    avgdatapath=config['avgdatapath']

    ncores=config['ncores']
    parallel=config['parallel']
    eps=config['eps_phase']


    inputdatapath=rawdatapath+folder+'/'
    if avgdatapath==None:
        outputdatapath=None
    else:
        outputdatapath=avgdatapath+folder+'/'

    #do the averaging
    output_data=aoa.rawdata2average(inputdatapath, incidence_angles, p_orientation_beta,
                                outputdatapath,
                                alpha_0,beta_0,gamma_0,parallel=parallel,num_cores=int(ncores))


    #==============================================================================
    #   slower transformation, but more accurate in forward direction
    #==============================================================================

    #reduce the number of cores for phasematrix expansion, due to memory issues
    ncores_reduced=int(ncores/2)
    
    
    #transform to phase matrix
    [phase_mat, max_ntrunc_pha, max_len_coeffs_pha]=aoa.mueller2phase_reexpand(
            output_data,grid_size_theta=grid_size,eps=eps,parallel=parallel,
            core_limit=ncores_reduced)


    phase_mat=spha.reshape_sphexp_2_common_exp(np.max(max_ntrunc_pha[:]),
                                               np.max(max_len_coeffs_pha[:]),phase_mat)

    output_data['eps_phase']=eps
    output_data['phase_matrix_sph']=phase_mat
#    output_data['phase_matrix_sph_old']=phase_mat.copy()
    output_data['max_ntrunc']=int(np.max(max_ntrunc_pha[:]))
    output_data['max_len_coeffs']=int(np.max(max_len_coeffs_pha[:]))

    #==========================================================================


    #calculate absorption vector
    ext_xsec=np.real(output_data['ext_mat_horiz_avg'][:,:,:,0])
    sca_xsec=np.real(output_data['phase_matrix_sph'][:,:,0:4,0]/(1/2*np.sqrt(1/np.pi)))
                #factor is due to the normalization of the spherical harmonics

#    abs_vector=np.real(output_data['ext_mat_horiz_avg'][:,:,:,0]-
#                output_data['phase_matrix_sph'][:,:,0:4,0]/(1/2*np.sqrt(1/np.pi)))
                #factor is due to the normalization of the spherical harmonics
    abs_vector=ext_xsec-sca_xsec
#    output_data['abs_vector_old']=abs_vector.copy()

    #check if first element of abs-vector is negative ans set it zero
    a1=abs_vector[:,:,0]

    logic_a1=a1<0
    if sum(logic_a1.ravel())>0:
        diagnostic=[min(abs(a1[logic_a1]/ext_xsec[:,:,0][logic_a1])),
                np.median(np.ravel((abs(a1[logic_a1]/ext_xsec[:,:,0][logic_a1])))),
                max(abs(a1[logic_a1]/ext_xsec[:,:,0][logic_a1]))]
    else:
        diagnostic=[np.nan,np.nan,np.nan]

    frac_of_neg_a1=sum(logic_a1.ravel())/np.size(logic_a1)

    #calculate rescaling
    rescale=1.+a1/sca_xsec[:,:,0]

#    a1[logic_a1]=0
#    abs_vector[:,:,0]=a1


    #if negative a1 found rescale phase matrix
    if frac_of_neg_a1>0:
        print('    ')
        print('oh no...we have to rescale the phase matrix')

        for i in range(np.size(a1,axis=0)):
            for j in range(np.size(a1,axis=1)):

                if logic_a1[i,j]==True:

                    print('   ')
                    print('rescaling...')
                    print('beta      = '+str(output_data['beta'][i]))
                    print('theta_inc = '+str(output_data['theta_inc'][j]))
                    print('recaling  = '+str(rescale[i,j]))

                    phase_mat[i,j,:,:]= phase_mat[i,j,:,:]*rescale[i,j]


        output_data['phase_matrix_sph']=phase_mat.copy()
        ext_xsec=np.real(output_data['ext_mat_horiz_avg'][:,:,:,0])
        sca_xsec=np.real(output_data['phase_matrix_sph'][:,:,0:4,0]/(1/2*np.sqrt(1/np.pi)))
        abs_vector=ext_xsec-sca_xsec

    a1=abs_vector[:,:,0]
    logic_a1=a1<0
    a1[logic_a1]=0.
    abs_vector[:,:,0]=a1

    output_data['rescale']=rescale
    output_data['abs_vector']=abs_vector
    #Export to SSDBnetcdf======================================================

    print('   ')
    print('===============================================================')
    print('shape = '+output_data['shape_name'])
    print('Temperature = '+str(output_data['Temperature'])+' K')
    print('Frequency = '+str(output_data['Frequency'])+' GHz')
    print('Size = '+str(output_data['diameter_volume_equ'])+' m')
    print('Fraction of corrected a(1) element = '+str(frac_of_neg_a1))
    print('Minimum deviation of neg. Abs_vector = '+str(diagnostic[0]))
    print('Median deviation of Abs_vector       = '+str(diagnostic[1]))
    print('Maximum deviation of neg. Abs_vector = '+str(diagnostic[2]))
    print('===============================================================')
    print('   ')

    if export_netdf==True:
        nc_file_list=a2s.export2SSDBnetCDF(output_data,config)
    else:
        nc_file_list=[]


    if outputdatapath is not None:

        #save regular
#        if regular_phasematrix==True:
#            [phase_matrix_reg,theta_s,phi_s]=aoa.create_regular_representation(
#                    output_data,grid_size_theta=grid_size,gridtype='regular')
#
#
#            output_data['theta_s']=theta_s
#            output_data['phi_s']=phi_s
#            output_data['phase_matrix_reg']=np.single(phase_matrix_reg)
        if regular_phasematrix==True:
            
            lmax = spha.get_lmax(output_data['phase_matrix_sph'][0,0,0,:])
            
            [phase_matrix_reg,theta_s,phi_s]=aoa.create_regular_representation(
                    output_data,lmax,grid_size_theta=grid_size,gridtype='regular',full_phi=True)
            # [phase_matrix_reg, theta_s, phi_s] = aoa.create_regular_representation(
            #     output_data, grid_size_theta=grid_size, gridtype='regular')
            
            output_data['theta_s']=theta_s
            output_data['phi_s']=phi_s
            output_data['phase_matrix_reg']=np.single(phase_matrix_reg)        
            del output_data['phase_matrix_sph']
            output_data['mue_mat_horiz_avg_sph']=output_data['mue_mat_horiz_avg_sph'][:,:,0,0:2]


        #Check if path exists
        if  not os.path.exists(outputdatapath):

            os.makedirs(outputdatapath)

        ##Save output in mat format
        matfile_name=os.path.join(outputdatapath,output_data['shape_name']+
                                  '_horiz_avg_phasematrix.mat')

        sio.savemat(matfile_name, {'output_data':output_data}, appendmat=True,
                format='5', long_field_names=True)

    return nc_file_list#,frac_of_neg_a1,output_data


#==============================================================================
# preperations
#==============================================================================


#folder
folder=sys.argv[1]


#folder debug
#folder='HexPlaAggCrystal_Id20_deq00197um_freq166.9GHz_T0270K'

# folder='IconSnow_Id28_deq01159um_freq005.0GHz_T0270K'
#folder='HongPlate_Id9_deq00999um_freq164.0GHz_T0190K'
#folder='HongPlate_Id9_deq00735um_freq671.0GHz_T0230K'
#folder='HexPlaAggCrystal_Id20_deq00986um_freq164.1GHz_T0230K'

print(folder)

#==============================================================================
# the actual processing
#==============================================================================
with ty.utils.Timer():
    nc_file_list=adda_process_single_AroParticle(config,folder)



