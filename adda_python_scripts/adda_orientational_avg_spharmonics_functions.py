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
import scipy.io as sio
import adda_control_functions as acf
import icosphere_grid as ico
from joblib import Parallel, delayed
import multiprocessing
#import typhon.plots 
#from netCDF4 import Dataset
#import netCDF4
import matplotlib.pyplot as plt
#import itertools
import typhon as ty
import spherical_harmonics_for_adda as spha
import shtns
import copy
import pickle
import scipy as sp
from datetime import datetime
from posix_spawn import spawn_command
from memory import memstring

import functools
# Force flush, otherwise output can disappear from slurm log file
print = functools.partial(print, flush=True)

#==============================================================================
# stoke rotation matrix
#==============================================================================
def rotater_mat(eta):
    #Important eta has to be in rad!!


    P=np.zeros((4,4))
    P[0,0]=1
    P[1,1]=np.cos(2*eta)
    P[2,1]=np.sin(2*eta)
    P[1,2]=-P[2,1]
    P[2,2]=P[1,1]
    P[3,3]=1

    return P

#==============================================================================
#  calculate_rotation_angles for mueller to phase matrix transformation
#==============================================================================

def calculate_rotation_angles(theta_inc, theta_s, phi_s):
    """calculates the rotation angles for the  mueller to phase matrix 
        transformation"""
        
    
    
    ql=np.shape(phi_s)

    #allocate
    PHI1mat=np.zeros((len(theta_inc),ql[0],ql[1]))*np.nan
    PHI2mat=np.zeros((len(theta_inc),ql[0],ql[1]))*np.nan
    
    print('calculating stokes rotation angles...')    
    print('theta size = '+str(ql[0]))
    print('phi size = '+str(ql[1]))
    print('   ')
    
    #calculate transformation matrices
    for m in range(0, np.size(theta_inc)):
        
        #Incidence direction (Laboratory system)
        t_i=theta_inc[m]*np.pi/180
        p_i=0*np.pi/180
         
        #Incidence vector and the corresponding vector of the vertical (parallel)
        #polarization
        k_i=np.array([np.sin(t_i)*np.cos(p_i),np.sin(t_i)*np.sin(p_i),np.cos(t_i)])
        v_i=np.array([np.cos(t_i)*np.cos(p_i),np.cos(t_i)*np.sin(p_i),-np.sin(t_i)])
        
        print(str(m))
        
        #let's go through all scattering angles
        for j in range(0,ql[0]): #theta / polar angle
            
            for i in range(0,ql[1]): #phi / azimuth angle
        
            
                #scattering angle
                t_s=theta_s[j,i]
                p_s=phi_s[j,i]
        
        
                #scattering vector and the corresponding vector of the vertical
                #(parallel) polarization
                k_s=np.array([np.sin(t_s)*np.cos(p_s),np.sin(t_s)*np.sin(p_s),np.cos(t_s)])
                v_s=np.array([np.cos(t_s)*np.cos(p_s),np.cos(t_s)*np.sin(p_s),-np.sin(t_s)])
        
        
                #scattering angle in the scattering plane
                kk=np.cross(k_s,k_i)
                sinTheta=np.sqrt(np.dot(kk,kk))
    
                #normal vector of the scattering plane
                if sinTheta==0.:
                    n=kk
                else:
                    n=kk/sinTheta
                
                #vector, that is parallel to scattering plane and normal to k_i
                P_i=np.cross(n,k_i)
                
                #decide if rotate anti clockwise or clockwise
                tester_i=np.dot(v_i,n)
                
                #catch numerical errors
                acos_phi1=np.dot(v_i,P_i)
                if acos_phi1>1:
                    acos_phi1=1
                elif acos_phi1<-1:
                    acos_phi1=-1
                
                #get the rotation angle
                if tester_i<=0:
                    phi1=np.arccos(acos_phi1) 
                elif tester_i>0:
                    phi1=-np.arccos(acos_phi1)
                else:
                    phi1=0    
                    
                    
                #vector, that is parallel to scattering plane and normal to k_s
                P_s=np.cross(n,k_s);
                
                #decide if rotate anti clockwise or clockwise
                tester_s=np.dot(v_s,n);
    
                #catch numerical errors                        
                acos_phi2=np.dot(v_s,P_s)
                if acos_phi2>1:
                    acos_phi2=1
                elif acos_phi2<-1:
                    acos_phi2=-1
                
                #get the rotation angle
                if tester_s<=0:
                    phi2=np.arccos(acos_phi2) 
                elif tester_s>0:
                    phi2=-np.arccos(acos_phi2)
                else:
                    phi2=0
                
                
                PHI1mat[m,j,i]=phi1;
                PHI2mat[m,j,i]=phi2;
                
    return PHI1mat, PHI2mat  

#==============================================================================
#  calculate_rotation_angles for mueller to phase matrix transformation (parallel)
#==============================================================================

def core(t_i,p_i,theta_s, phi_s):
        
    #Incidence vector and the corresponding vector of the vertical (parallel)
    #polarization
    k_i=np.array([np.sin(t_i)*np.cos(p_i),np.sin(t_i)*np.sin(p_i),np.cos(t_i)])
    v_i=np.array([np.cos(t_i)*np.cos(p_i),np.cos(t_i)*np.sin(p_i),-np.sin(t_i)])
    
    
    ql=np.shape(phi_s)

    #allocate
    PHI1mat=np.zeros((ql[0],ql[1]))*np.nan
    PHI2mat=np.zeros((ql[0],ql[1]))*np.nan
    
    
    print('theta_inc = '+str(t_i*180/np.pi))
    
    #let's go through all scattering angles
    for j in range(0,ql[0]): #theta / polar angle
        
        for i in range(0,ql[1]): #phi / azimuth angle
    
        
            #scattering angle
            t_s=theta_s[j,i]
            p_s=phi_s[j,i]
    
    
            #scattering vector and the corresponding vector of the vertical
            #(parallel) polarization
            k_s=np.array([np.sin(t_s)*np.cos(p_s),np.sin(t_s)*np.sin(p_s),np.cos(t_s)])
            v_s=np.array([np.cos(t_s)*np.cos(p_s),np.cos(t_s)*np.sin(p_s),-np.sin(t_s)])
    
    
            #scattering angle in the scattering plane
            kk=np.cross(k_s,k_i)
            sinTheta=np.sqrt(np.dot(kk,kk))

            #normal vector of the scattering plane
            if sinTheta==0.:
                n=kk
            else:
                n=kk/sinTheta
            
            #vector, that is parallel to scattering plane and normal to k_i
            P_i=np.cross(n,k_i)
            
            #decide if rotate anti clockwise or clockwise
            tester_i=np.dot(v_i,n)
            
            #catch numerical errors
            acos_phi1=np.dot(v_i,P_i)
            if acos_phi1>1:
                acos_phi1=1
            elif acos_phi1<-1:
                acos_phi1=-1
            
            #get the rotation angle
            if tester_i<=0:
                phi1=np.arccos(acos_phi1) 
            elif tester_i>0:
                phi1=-np.arccos(acos_phi1)
            else:
                phi1=0    
                
                
            #vector, that is parallel to scattering plane and normal to k_s
            P_s=np.cross(n,k_s);
            
            #decide if rotate anti clockwise or clockwise
            tester_s=np.dot(v_s,n);

            #catch numerical errors                        
            acos_phi2=np.dot(v_s,P_s)
            if acos_phi2>1:
                acos_phi2=1
            elif acos_phi2<-1:
                acos_phi2=-1
            
            #get the rotation angle
            if tester_s<=0:
                phi2=np.arccos(acos_phi2) 
            elif tester_s>0:
                phi2=-np.arccos(acos_phi2)
            else:
                phi2=0
            
            
            PHI1mat[j,i]=phi1;
            PHI2mat[j,i]=phi2;
            
    return PHI1mat,PHI2mat


def calculate_rotation_angles_parallel(theta_inc, theta_s, phi_s,core_limit=16):
    """calculates the rotation angles for the  mueller to phase matrix 
        transformation"""        
          
        
    num_cores = multiprocessing.cpu_count()
    
    #uncomment if number of core should be limited
    if num_cores>core_limit:
        num_cores=core_limit    
        
    
                
    
    ql=np.shape(phi_s)

    #allocate
    PHI1mat=np.zeros((len(theta_inc),ql[0],ql[1]))*np.nan
    PHI2mat=np.zeros((len(theta_inc),ql[0],ql[1]))*np.nan
    
    print('calculating stokes rotation angles...')    
    print('theta size = '+str(ql[0]))
    print('phi size = '+str(ql[1]))
    print('   ')
    
    #calculate transformation matrices
#    for m in range(0, np.size(theta_inc)):
#        
#        #Incidence direction (Laboratory system)
#        t_i=theta_inc[m]*np.pi/180
#        p_i=0*np.pi/180
#        
#        
#        [PHI1mat[m,:,:],PHI2mat[m,:,:]]=core(t_i,p_i,theta_s, phi_s)
        
    theta_i=theta_inc*np.pi/180
    p_i=0*np.pi/180

    results = Parallel(n_jobs=num_cores,max_nbytes=None)(delayed(core)
                         (t_i,p_i,theta_s, phi_s) 
                         for t_i in theta_i)
             
    for i in range(0,len(results)):
        PHI1mat[i,:,:]=results[i][0]
        PHI2mat[i,:,:]=results[i][1]

                
    return PHI1mat, PHI2mat  


#==============================================================================
#  mueller to phase matrix transformation
#==============================================================================

def mueller2phase(output_data):
    """transforms the mueller matrices in spherical harmonics representation
       to stokes phase matrix in spherical harmonics representation"""


    #some basic definition
    c0=299792458
    
    #wavelength
    lamb=c0/output_data['Frequency']
    
    #incidence angle
    theta_inc=output_data['theta_inc']
    
    #truncation number of spherical harmonics expansion
    max_ntrunc=output_data['max_ntrunc']
    
    
    
    #create spherical object
    if max_ntrunc % 2 ==0:
        min_grid_size=max_ntrunc+2
    else:
        min_grid_size=max_ntrunc+1
    
    x = spha.Spharmt(min_grid_size*2+1,min_grid_size,max_ntrunc,1,gridtype='gaussian')
    
    #spatial grid
    xlons,xlats = np.meshgrid(x.lons, x.lats)
    
    #get sizes
    qm=np.shape(output_data['mue_mat_horiz_avg_sph'])
    ql=np.shape(xlons)
    
    #calculate rotation angles      
    [PHI1mat, PHI2mat]=calculate_rotation_angles(theta_inc, xlats, xlons)
    
    
    #allocate
    mueller_rot=np.zeros(qm,dtype=complex)*np.nan
    mueller_rot_spatial=np.zeros((qm[0],qm[1],qm[2],ql[0],ql[1]))*np.nan
    
    phase_mat_spatial=np.zeros((qm[0],qm[1],qm[2],ql[0],ql[1]))*np.nan
    phase_mat=np.ones(qm,dtype=complex)*np.nan
    
    
    # do the transformation
    for i in range(0,qm[0]):#loop over the beta angles
            
        for j in range(0,qm[1]):#loop over the incidence angles
            
            for k in range(0,qm[2]):# loop over the matrix elements
                
                #As the angular scattering grid of the mueller matrix from an adda
                #simulation is always relative to the incidence vector, the scattering grid
                #has to be transformed into the laboratory angualr scattering grid. For our
                #adda calculations always holds that the incidence vector is always within
                #the X-Z-plane. Therefore we only have to rotate along the Y-axis.

            
                #Rotate along Y-axis (phi_i=0), and then we got our scattering angle
                #relative to the laboratory system
                m_ijk=np.complex128(output_data['mue_mat_horiz_avg_sph'][i,j,k,:])
                mueller_rot[i,j,k,:]=x.yrotate(m_ijk.copy(),np.pi*theta_inc[j]/180)
                
    
                #transform mueller from spectral to spatial  
                mueller_rot_spatial[i,j,k,:,:]=x.spectogrd(mueller_rot[i,j,k,:])
                
    
            #do the polarization transformation
            for u in range(0,ql[0]): #theta / polar angle
            
                for v in range(0,ql[1]): #phi / azimuth angle
            
                    m_uv=mueller_rot_spatial[i,j,:,u,v]
                    
                    m_uv=np.reshape(m_uv,(4,4))
                    
                    #rotate from the incidence plane to scattering plane
                    rot1=rotater_mat(PHI1mat[j,u,v])
                    rot2=rotater_mat(-PHI2mat[j,u,v])               
                       
                    #the actual transformation from mueller matrix to phase matrix
                    stokes_mat=np.dot(np.dot(rot2,m_uv), rot1)
                      
                    phase_mat_spatial[i,j,:,u,v]=stokes_mat.flatten()*lamb**2/(4*np.pi**2)
    
    
            #expand phase matrix into spherical harmonics        
            for k in range(0,qm[2]):
                phase_mat[i,j,k,:]=x.grdtospec(phase_mat_spatial[i,j,k,:,:])

    return phase_mat



#==============================================================================
#    mueller2phase  core transformation, outsourced due to parallelization
#==============================================================================

def mueller2phase_core(j,m_i,theta_inc,lamb,PHI1mat,PHI2mat,eps,
                       lat_new,lon_new,qm,min_grid_size,max_ntrunc,grid_size_theta):
    
    print('  ')
    spawn_command('free -m')
    print(f'Before transforming: {memstring()}')
    print('now do transformation...')        
    print('theta: '+str(theta_inc[j])+'deg')
    print('   ')

    
    j=int(j)
    
    
    x = spha.Spharmt(min_grid_size*2+1,min_grid_size,max_ntrunc,1,gridtype='gaussian')    
    x_fine = spha.Spharmt(grid_size_theta*2+1,grid_size_theta,max_ntrunc,1,gridtype='gaussian')    
    
       
    #spatial grids
    xlons,xlats = np.meshgrid(x.lons, x.lats)
    [xlons_fine,xlats_fine] = np.meshgrid(x_fine.lons, x_fine.lats)
    
    ql=np.shape(xlons_fine)
    
    
    mueller_rot_spatial=np.zeros((qm[2],ql[0],ql[1]))*np.nan
    phase_mat_spatial=np.zeros((qm[2],ql[0],ql[1]))*np.nan
        
    phase_mat_ijk=[None] * qm[2]
    for k in range(0,qm[2]):# loop over the matrix elements
        
        #As the angular scattering grid of the mueller matrix from an adda
        #simulation is always relative to the incidence vector, the scattering grid
        #has to be transformed into the laboratory angualr scattering grid. For our
        #adda calculations always holds that the incidence vector is always within
        #the X-Z-plane. Therefore we only have to rotate along the Y-axis.

    
        #Rotate along Y-axis (phi_i=0), and then we got our scattering angle
        #relative to the laboratory system
#        m_ijk=np.complex128(output_data['mue_mat_horiz_avg_sph'][i,j,k,:])
        m_ijk=np.complex128(m_i[j,k,:])
        
        
        if k%5!=0:
            
            #transform to spatial
            temp=x.spectogrd(m_ijk)
            
            #interpolate in theta direction                    
            dlidx=2
            qtemp=np.shape(temp)
            
            if qtemp[1]%2==0:
                binshift=int(qtemp[1]/2)
            elif qtemp[1]%2==1:
                binshift=int((qtemp[1]-1)/2)
            
            m_mid=temp[dlidx:-dlidx:1,:]
            m_up=np.roll(np.flipud(m_mid),binshift,axis=1)
            m_dn=np.roll(np.flipud(m_mid),binshift,axis=1)                    
            m_temp=np.vstack((m_up,m_mid))
            m_temp=np.vstack((m_temp,m_dn))
                               
            
            lat_mid=xlats[dlidx:-dlidx:1,:]
            lat_dn=xlats[dlidx:-dlidx:1,:]+np.pi
            lat_up=xlats[dlidx:-dlidx:1,:]-np.pi
            lat_temp=np.vstack((lat_up,lat_mid))
            lat_temp=np.vstack((lat_temp,lat_dn))
            
            fun=sp.interpolate.interp1d(lat_temp[:,0], m_temp,kind='linear',axis=0)
            temp_int_lat=fun(lat_new[:,0])
            
            qtemp=np.shape(temp_int_lat)
            temp_new=np.array([np.interp(lon_new[0,:],xlons[0,:], temp_int_lat[i,:],period=2*np.pi) for i in range(qtemp[0])])
                                                 
            
            #transform in spherical harmonics
            [sph_coeffs, x2, _,  __]=spha.sph_expansion(temp_new,lat_new[:,0],lon_new[0,:]) 
            
            #rotate mueller                    
            mueller_rot=x2.yrotate(sph_coeffs,np.pi*theta_inc[j]/180)  
                                
            #transform mueller from spectral to spatial  
            mueller_rot_spatial[k,:,:]=x2.spectogrd(mueller_rot)
            
        else:                                        
            #rotate mueller                    
            mueller_rot=x_fine.yrotate(m_ijk.copy(),np.pi*theta_inc[j]/180)                

            #transform mueller from spectral to spatial  
            mueller_rot_spatial[k,:,:]=x_fine.spectogrd(mueller_rot)
            
            #ensure that (1,1) or in python (0,0) is always >=0
            if k==0:
                dummy=mueller_rot_spatial[k,:,:]
                logic0=dummy<0
                dummy[logic0]=0
                mueller_rot_spatial[k,:,:]=dummy
        

    #do the polarization transformation
    for u in range(0,ql[0]): #theta / polar angle
    
        for v in range(0,ql[1]): #phi / azimuth angle
    
            m_uv=mueller_rot_spatial[:,u,v]
            
            m_uv=np.reshape(m_uv,(4,4))
            
            #rotate from the incidence plane to scattering plane
            rot1=rotater_mat(PHI1mat[j,u,v])
            rot2=rotater_mat(-PHI2mat[j,u,v])               
               
            #the actual transformation from mueller matrix to phase matrix
            stokes_mat=np.dot(np.dot(rot2,m_uv), rot1)
              
            phase_mat_spatial[:,u,v]=stokes_mat.flatten()*lamb**2/(4*np.pi**2)

    #ensure that (1,1) or in python (0,0) is always >=0
    dummy=phase_mat_spatial[0,:,:]
    logic0=dummy<0
    dummy[logic0]=0
    phase_mat_spatial[0,:,:]=dummy
    

    #expand phase matrix into spherical harmonics        
    
    temp=phase_mat_spatial[:,:,:]
    
    input_data={}
    input_data['mueller_array']=np.moveaxis(temp, 0, -1)
    input_data['phis_grid']=xlons_fine[0,:]*180/np.pi
    input_data['thetas_grid']=xlats_fine[:,0]*180/np.pi
    
    [phase_mat_ijk,rel_error, max_len_coeffs_ij, max_ntrunc_ij
         ]=spha.expand_mueller_2_spharmonics(input_data,eps,diagnostics=False)
    
    print()    
    print('theta: '+str(theta_inc[j])+'deg')
    print('Yeah, it is done!')
    
    return phase_mat_ijk,rel_error,max_len_coeffs_ij, max_ntrunc_ij
 

#==============================================================================
#  mueller to phase matrix transformation
#==============================================================================


def mueller2phase_reexpand(output_data,grid_size_theta=180,eps=0,parallel=False,core_limit=16):
    """transforms the mueller matrices in spherical harmonics representation
       to stokes phase matrix in spherical harmonics representation.
       
       This transformation first transforms the spectral representation to the
       spatial representation and interpolate over the poles to reduce the error
       in the forward direction (poles) to high resolution spatial grid. 
       The mueller matrices on this HR grid are than expand to the full spectral       
       representation and rotated.Than the mueller matrices are transformed back 
       to the spatial representation and transformed to the phase matrix 
       representation on the spatial grid, which are the transformed to a spectral
       grid representation.
       
       grid_size_theta has to be even, otherwise it won't work
       eps is the accuracy of the phase matrix expansion
    """


    #some basic definition
    c0=299792458
    
    #wavelength
    lamb=c0/output_data['Frequency']
    
    #incidence angle
    theta_inc=output_data['theta_inc']
    
    #truncation number of spherical harmonics expansion
    max_ntrunc=output_data['max_ntrunc']

    if grid_size_theta % 2 != 0:
        grid_size_theta = grid_size_theta + 1

        print('Changed grid_size_theta to grid_size_theta+1')
        print('because, shtns does not like odd grid sizes for gaussian grids')
        print('grid_size_theta =',grid_size_theta)
    
    #create spherical object
    if max_ntrunc % 2 ==0:
        min_grid_size=max_ntrunc+2
    else:
        min_grid_size=max_ntrunc+1

    if min_grid_size<16:
        min_grid_size=16

    if grid_size_theta<=min_grid_size:
        grid_size_theta=min_grid_size


        
#    x = spha.Spharmt(min_grid_size*2+1,min_grid_size,max_ntrunc,1,gridtype='gaussian')    
#    x_fine = spha.Spharmt(grid_size_theta*2+1,grid_size_theta,max_ntrunc,1,gridtype='gaussian')    
#    
#    #spatial grids
#    xlons,xlats = np.meshgrid(x.lons, x.lats)
#    [xlons_fine,xlats_fine] = np.meshgrid(x_fine.lons, x_fine.lats)
    
    #create grids
    [lon_new,lat_new] = np.meshgrid(np.linspace(0,360,grid_size_theta*2+1)
        *np.pi/180, np.linspace(0,180,grid_size_theta+1)*np.pi/180)
    
    
    #get sizes
    qm=np.shape(output_data['mue_mat_horiz_avg_sph'])
    
    
    #calculate rotation angles      
    if parallel==True:
        [PHI1mat, PHI2mat]=calculate_rotation_angles_parallel(theta_inc, lat_new, lon_new, core_limit=16)
    
    elif parallel==False:
        [PHI1mat, PHI2mat]=calculate_rotation_angles(theta_inc, lat_new, lon_new)
    
    #allocate
    phase_mat=np.ones(qm,dtype=complex)*np.nan
    
    phase_mat_i=[None] * qm[0]
    
    max_ntrunc_output=np.zeros((qm[0],qm[1]))
    max_len_coeffs_output=np.zeros((qm[0],qm[1]))
    
#    parallel=False
    
    
    # do the transformation
    for i in range(0,qm[0]):#loop over the beta angles
        
        phase_mat_ij=[None] * qm[1]
        
        spawn_command('free -m')
        print(f'Before parallel: {memstring()}')
        print('now it\'s time parallize.../\/\\//\\/')
        print('                           ||||||||||')
        
        
        if parallel==True:
            print('n_beta: '+str(i))
            
            num_cores = multiprocessing.cpu_count()
    
            #uncomment if number of core should be limited
            if num_cores>core_limit:
                num_cores=core_limit
            
            #get mueller matrix of i-th beta angle           
            m_i=np.complex128(output_data['mue_mat_horiz_avg_sph'][i,:,:,:])   
            
            
            j_range=np.arange(0,qm[1])
                        
            #the core transformation in parallel 
            results = Parallel(n_jobs=num_cores,max_nbytes=None)(delayed(mueller2phase_core)
                (j,m_i,theta_inc,lamb,PHI1mat,PHI2mat,eps,lat_new,
                 lon_new,qm,min_grid_size,max_ntrunc,grid_size_theta) 
                 for j in j_range)
             

            for j in range(0,len(results)):

                phase_mat_ij[j]=results[j][0]
                max_len_coeffs_output[i,j]=results[j][2]
                max_ntrunc_output[i,j]=results[j][3]

                
        elif parallel==False:
    
            for j in range(0,qm[1]):#loop over the incidence angles
                          
            
                #get mueller matrix of i-th beta angle           
                m_i=np.complex128(output_data['mue_mat_horiz_avg_sph'][i,:,:,:])

                #the core transformation
                [phase_mat_ijk,rel_error, max_len_coeffs_ij, max_ntrunc_ij
                    ]=mueller2phase_core(j,m_i,theta_inc,lamb,PHI1mat,PHI2mat,eps,lat_new,
                     lon_new,qm,min_grid_size,max_ntrunc,grid_size_theta)

                
                phase_mat_ij[j]=phase_mat_ijk
                max_len_coeffs_output[i,j]=max_len_coeffs_ij
                max_ntrunc_output[i,j]=max_ntrunc_ij

            
        phase_mat_i[i]=phase_mat_ij        


    phase_mat=phase_mat_i
    return phase_mat, max_ntrunc_output, max_len_coeffs_output




#==============================================================================
# create phase matrix in spatial representation
#==============================================================================

# def create_regular_representation(output_data,grid_size_theta=90, gridtype='gaussian'):
#    """create phase matrix in spatial grid representation"""
#
#
#    #truncation number of spherical harmonics expansion
#    max_ntrunc=output_data['max_ntrunc']
#
#    phase_mat_sph=output_data['phase_matrix_sph']
#
#
#
#    if gridtype=='gaussian':
#
#        if max_ntrunc % 2 ==0:
#            min_grid_size=max_ntrunc+2
#        else:
#            min_grid_size=max_ntrunc+1
#
#        if min_grid_size<grid_size_theta:
#            min_grid_size=grid_size_theta
#        elif grid_size_theta<min_grid_size:
#            grid_size_theta=min_grid_size
#
#        if min_grid_size % 2 == 1:
#            min_grid_size=min_grid_size+1
#
#
#        #create spherical harmonics object
#        x = spha.Spharmt(min_grid_size*2+1,min_grid_size,max_ntrunc,1,gridtype='gaussian')
#
#    elif gridtype=='regular':
#
#        grid_size_theta=2*grid_size_theta
#
#        if max_ntrunc % 2 ==0:
#            min_grid_size=2*max_ntrunc+2
#        else:
#            min_grid_size=2*(max_ntrunc+1)
#
#        if min_grid_size<grid_size_theta:
#            min_grid_size=grid_size_theta
#        elif grid_size_theta<min_grid_size:
#            grid_size_theta=min_grid_size
#
#        if min_grid_size % 2 == 1:
#            min_grid_size=min_grid_size+1
#
#
#
#        x = spha.Spharmt(min_grid_size*2+1,min_grid_size,max_ntrunc,1,gridtype='regular')
#
#
#    #create spatial grid
#    phi_s,theta_s = np.meshgrid(x.lons, x.lats)
#    phi_s=phi_s*180/np.pi
#    theta_s=theta_s*180/np.pi
#
#    qm=np.shape(output_data['phase_matrix_sph'])
#    ql=np.shape(phi_s)
#
#    #allocate
#    phase_matrix_reg=np.ones((qm[0],qm[1],qm[2],ql[0],ql[1]))
#
#    #transform to spatial grid
#    for i in range(0,qm[0]):#loop over the beta angles
#
#        for j in range(0,qm[1]):#loop over the incidence angles
#
#            for k in range(0,qm[2]):# loop over the matrix elements
#
#                phase_matrix_reg[i,j,k,:,:]=x.spectogrd(phase_mat_sph[i,j,k,:])
#
#
#
#    if gridtype=='regular':
#
#          phase_matrix_reg=phase_matrix_reg[:,:,:,0::2,0::2]
#          phi_s=phi_s[0::2,0::2]
#          theta_s=theta_s[0::2,0::2]
#
#
#    return phase_matrix_reg,theta_s,phi_s
    
def create_regular_representation(output_data, max_ntrunc,
                                  grid_size_theta=90, gridtype='gaussian',full_phi=False):

    phase_mat_sph=output_data['phase_matrix_sph']

    if gridtype=='gaussian':

        if max_ntrunc % 2 ==0:
            min_grid_size=max_ntrunc+2
        else:
            min_grid_size=max_ntrunc+1

        if min_grid_size<grid_size_theta:
            min_grid_size=grid_size_theta
        elif grid_size_theta<min_grid_size:
            grid_size_theta=min_grid_size

        if min_grid_size % 2 == 1:
            min_grid_size=min_grid_size+1


        #create spherical harmonics object
        x = spha.Spharmt(min_grid_size*2,min_grid_size,max_ntrunc,1,gridtype='gaussian')

    elif gridtype=='regular':

        grid_size_theta_old=grid_size_theta

        if max_ntrunc % 2 ==0:
            min_grid_size=2*max_ntrunc+2
        else:
            min_grid_size=2*(max_ntrunc+1)

        if min_grid_size<grid_size_theta:
            min_grid_size=grid_size_theta
        elif grid_size_theta<min_grid_size:
            grid_size_theta=min_grid_size

        if min_grid_size % 2 == 1:
            min_grid_size=min_grid_size+1



        x = spha.Spharmt(min_grid_size*2,min_grid_size,max_ntrunc,1,gridtype='regular')


    #create spatial grid
    #phi_s,theta_s = np.meshgrid(x.lons, x.lats)
    phi_s=x.lons*180/np.pi
    theta_s=x.lats*180/np.pi

    qm=np.shape(phase_mat_sph)
    #ql=np.shape(phi_s)

    #allocate
    phase_matrix_reg=np.ones((qm[0],qm[1],qm[2],len(theta_s),len(phi_s)))


    #transform to spatial grid
    for i in range(0,qm[0]):#loop over the beta angles

        for j in range(0,qm[1]):#loop over the incidence angles

            for k in range(0,qm[2]):# loop over the matrix elements

                phase_matrix_reg[i,j,k,:,:]=x.spectogrd(np.complex128(phase_mat_sph[i,j,k,:]))




    if gridtype=='regular' and (
        grid_size_theta_old<grid_size_theta or grid_size_theta_old % 2==1):

        #to have uneven grid size or grid sizes that are smaller than the
        #minimum grid size, we need to interpolate.

        f_int=sp.interpolate.interp1d(theta_s,phase_matrix_reg,axis=3,kind='linear',
                                assume_sorted=True)

        theta_s=np.linspace(0,180,num=grid_size_theta_old)
        phase_matrix_reg=f_int(theta_s)


    if full_phi==True:

        if grid_size_theta_old % 2==1:
            phi_size=(grid_size_theta_old-1)*2+1
        else:
            phi_size = grid_size_theta_old*2 + 1

        phi_s_i=np.linspace(0,360.,num=phi_size)

        #add value of phi_s=0 for phi_s=360
        dump = phase_matrix_reg[:, :, :, :, 0]
        qdump = np.shape(dump)
        dump = np.reshape(dump, qdump + (1,))
        phase_matrix_reg = np.append(phase_matrix_reg, dump, axis=4)
        phi_s=np.append(phi_s,360)

    else:
        phi_s_i = np.linspace(0, 180., num=grid_size_theta_old)

    # interpolate to assure that azimuth grid always include 0 and 180 and if
    # the grid size are smaller than the minimum grid size.
    f_int = sp.interpolate.interp1d(phi_s, phase_matrix_reg, axis=4, kind='linear',
                                    assume_sorted=True)

    phase_matrix_reg=f_int(phi_s_i)
    phi_s=phi_s_i


    # #filter out negative values of the (1,1) or in python (0,0)
    # dummy=phase_matrix_reg[0,...]
    # logic0=dummy<0
    # dummy[logic0]=0
    # phase_matrix_reg[0,...]=dummy

    return phase_matrix_reg,theta_s,phi_s


#==============================================================================
#  plot mueller matrix
#==============================================================================

def plot_mueller(mueller_mat):

    qm=np.shape(mueller_mat)
    
    phi=np.linspace(0,2*np.pi,qm[1])
    theta=np.linspace(0,np.pi,qm[0])

    [PHI,THETA]=np.meshgrid(phi,theta)   
 
    
    fig = plt.figure()
    
    for j in range(0,16):
        
        
        
        plt.subplot(4, 4, j+1)
        h1=plt.pcolormesh(PHI,THETA,mueller_mat[:,:,j])
        
        if j==0:
            plt.ylabel('\theta [deg]')
            
        if j==15: 
            plt.xlabel('\theta [deg]')
        
        fig.colorbar(h1)



    return fig


#==============================================================================
#  create vertical polarisation vectors for each vertex
#==============================================================================

def create_vpol_vector(vertices):
    # Function to create unit vectors of the vertical polaristion according
    # to the incidence direction (vertex)
    
    if vertices.ndim==1:
        vertices=np.reshape(vertices,(1,len(vertices)))
        
    elif vertices.ndim>2:
        raise Exception('vertices must be an N x 3  array');
            
    

    # v= e_h x e_k = (e_z x e_k) x e_k    
    vpol=np.array([vertices[:,0]*vertices[:,2],vertices[:,1]*vertices[:,2],
                       -(vertices[:,0]**2 + vertices[:,1]**2)]).transpose()
    
    
    #define the k_i = e_z
    logicT=np.floor(np.sum(vertices==np.array([0,0,1]),axis=1)/3).astype('bool')
    vpol[logicT,:]=[[0,1,0]]
    
    #define the k_i = -e_z
    logicB=np.floor(np.sum(vertices==np.array([0,0,-1]),axis=1)/3).astype('bool')
    vpol[logicB,:]=[[0,-1,0]]
    
    #normalize vpol
    norm_V=np.sqrt( vpol[:,0]**2 + vpol[:,1]**2 + vpol[:,2]**2)    
    norm_V=np.tile(norm_V,(3,1)).transpose()
    vpol=vpol/norm_V


    return vpol

#==============================================================================
# define Polarizers 
#==============================================================================




def pol_changer(vertices, vpol, vpol_lab ,mueller_mat,ext_mat,sph_object):
    #function to transform the polarization from one base to another
    # see Appendix of TR2-workplan report for details.
    
    mueller_array_local=copy.deepcopy(mueller_mat)
    ext_mat_array_local=ext_mat.copy()
    
    #shape of mueller
#    qm=np.shape(mueller_array_local)
    

    #scattering angle in the scattering plane
    hpol_lab=np.array([np.cross(a,b) for a,b in zip(vpol_lab,vertices)])  
    
    x=np.array([np.dot(a,b) for a,b in zip(vpol,vpol_lab)])
    y=np.array([np.dot(a,b) for a,b in zip(vpol,hpol_lab)])
    
    phi_i=np.arctan2(y,x)
    phi_s=-phi_i
        
    
    #transform polarization of extinction matrix
    KxP1=np.array([np.dot(a,rotater_mat(b)) for a,b in zip(ext_mat_array_local,phi_i)])
    ext_mat_rot=np.array([np.dot(rotater_mat(a),b) for a,b in zip(phi_s,KxP1)])
    
    
    #transform polarization of mueller matrix, 
    #for the mueller matrix the the solution is different from the extinction matrix. 
    #Basically, we already have all the information we need. the rotation 
    #is actually just a shifting of the Mueller matrix by a specifice value in the azimuth.
    #Practically, this means we have just to rotate the sph coefficients 
    #around the Z axis of spherical harmonics by angle phi_i
           
    
#    sph_coeffs_null=np.zeros(max_len_coeffs,dtype=complex)
    
    if len(mueller_array_local)!=0:
        
        mueller_mat_rot=copy.deepcopy(mueller_array_local)
        
        
        for i in range(0,len(phi_i)):
            
            n_idx=np.array([len(j) for j in mueller_array_local[i]])
        
            for j in range(0,len(n_idx)):
        
                
                sph_coeffs=mueller_array_local[i][j]
                
#                #get truncation or lmax
#                ntrunc=int(1/2+np.sqrt(1/4+2*len(sph_coeffs))-2)
#
#                #create object                                
#                sph_object=shtns.sht(ntrunc, ntrunc, 1, shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)                
                                               
                temp2=sph_object.Zrotate(sph_coeffs,phi_i[i])         
        
                mueller_mat_rot[i][j]=np.complex64(temp2)
    

    else:
        mueller_mat_rot=np.empty(0)
    
    
    return mueller_mat_rot,ext_mat_rot
    

def construct_by_symmetry(mueller_mat,ext_mat,V_sym):
    #function to reconstruct mueller matrix and extinction matrix by using
    #mirror symmetry
    
    
    mueller_array_local=mueller_mat.copy()
    ext_mat_array_local=ext_mat.copy()
    
    #shape of mueller
    qm=np.shape(mueller_array_local)
    
    #rad per bin
    rpb=np.pi*2/qm[2]
    
    Sym_logic=np.abs((V_sym-1)/2)
    Sym_logic=Sym_logic.astype(bool)
    
    #symmetry matrix
    Ma=np.ones([2,2])
    Mb=-np.ones([2,2])
    M=np.vstack((np.hstack((Ma,Mb)),np.hstack((Mb,Ma))))
    
            
    
    if len(mueller_array_local)!=0:
        
        M_m=np.reshape(M,(1,1,qm[3]))
        M_m=np.tile(M_m,(qm[1],qm[2],1))
    
        #Construct mueller matrix
        for i in range(0,np.size(V_sym,axis=1)):
            
            if i==2:
                            
                #estimate how many bins we have to shift
                binshift=np.ndarray.astype(np.round( np.ndarray.astype(Sym_logic[:,i],'int')*np.pi/rpb ),'int')
            
                #shift the bins        
                mueller_array_local=np.array([np.roll(a,b,axis=1) for a,b in zip(mueller_array_local,binshift)])
            
            for j in range(0,np.size(V_sym,axis=0)):
            
                if Sym_logic[j,i]==True:
                
                    mueller_array_local[j,:,:,:]=np.flip(mueller_array_local[j,:,:,:],axis=1)*M_m
                    
        
 
            
    else:
            
        mueller_array_local=np.empty(0)
        


    #Construct extinction matrix
    
    
    #for extinction
    M_k=np.copy(M)
    
    #Number of symmetry operations
    Nsym=np.sum(Sym_logic,axis=1)
    
    #make symmetry operation        
    ext_mat_sym=np.array([a*(M_k**b) for a,b in zip(ext_mat_array_local,Nsym)])


    return mueller_array_local, ext_mat_sym
        
        
#==============================================================================
#  define average function
#==============================================================================

def orientational_avg( vertices, vpol, mueller_array, ext_mat_array, max_len_coeffs, max_ntrunc,
                      inc_dir, beta,
                      ref_lev, sym_XYobj=False, sym_XZobj=False,sym_YZobj=False):
    
    '''function to do the orientational average for one particle for a specific 
       incidence angle and a specific particle tilt (beta) angle'''

#    spawn_command('free -m')
#    print('  ')
#    print(f'starting or-avg: {memstring()}')
#    print('beta: '+str(beta)+'deg - theta: '+str(np.arccos(inc_dir[2])*180/np.pi)+'deg') 
    
    

    #get the shape
    qm=(16,max_len_coeffs)
    q_ext=np.shape(ext_mat_array)
    
    #create spherical harmonics object
    sph_object=shtns.sht(max_ntrunc, max_ntrunc, 1, shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)
    
#    # reshape mueller_array to common sph_expansion
#    for i in range(0,len(mueller_array)):
#                    
#        if i%100==0:
#            print('theta: '+str(np.arccos(inc_dir[2])*180/np.pi)+'deg - reshaping M: index '+str(i))
#        
#        n_idx=np.array([len(j) for j in mueller_array[i]])
#        
#        for j in range(0,len(n_idx)):
#        
#            mueller_array[i][j]=spha.map_sphexp(mueller_array[i][j],sph_object) 
    
    
    
    
    #zero matrix
    ZERO_EXT=np.zeros((q_ext[1],q_ext[2]))
    ZERO_MUE=np.complex64(np.zeros((qm[0],qm[1]),dtype=complex))

    mue_mat_alpha_temp=ZERO_MUE.copy()
    ext_mat_alpha_temp=ZERO_EXT.copy()
    
    
    spawn_command('free -m')
    print('  ')
    print(f'Before avg: {memstring()}')
    print('now do averaging...')
    print('beta: '+str(beta)+'deg - theta: '+str(np.arccos(inc_dir[2])*180/np.pi)+'deg')         
    print('   ')

    #get the number of quadrature points
    numQuadraturePoints=4*2**ref_lev
    if numQuadraturePoints>64:
        numQuadraturePoints=64

    #get the weights for gauss legendre quadrature
    [sample_points,gauss_weights]=np.polynomial.legendre.leggauss(numQuadraturePoints)
    
    #angles to be evaluated
    p_orientation_alpha=sample_points*180;
    p_orientation_gamma=sample_points*180;
       
 

    #counter alpha
    n_alpha=-1
    
    
    for a_i in range(0,len(p_orientation_alpha)):
        
        n_alpha=n_alpha+1
        alpha=p_orientation_alpha[a_i]

#        print('   ')
#        spawn_command('free -m')
#        print('  ')
#        print(f'Before avg: {memstring()}')
        print('I am averaging...alpha = '+str(alpha)+'deg')
        print('beta: '+str(beta)+'deg - theta: '+str(np.arccos(inc_dir[2])*180/np.pi)+'deg')
        print(datetime.now().time().isoformat())

        mue_mat_gamma_temp=ZERO_MUE.copy()
        ext_mat_gamma_temp=ZERO_EXT.copy()

        #counter gamma
        n_gamma=-1
#
#        if beta==0:
#            len_gamma=1
#        else:
        len_gamma=len(p_orientation_gamma)


        #sample gamma angles
        for g_i in range(0,len_gamma):

            n_gamma=n_gamma+1
            gamma=p_orientation_gamma[g_i]


#            print('n_alpha: '+str(n_alpha)+', n_gamma: '+str(n_gamma))
            
            #make copy
#            mueller_array_temp=copy.deepcopy(mueller_array)
#            ext_mat_array_temp=ext_mat_array.copy()
            
            #rotate and interpolate incidence directions
            vertices_rot=ico.rotate_vertices(vertices,alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)
            
            [w_int,idx]=ico.interpolation_weights_icosahedric_grid(vertices_rot,inc_dir)
            
            #create lab vertical polarization vectors
            vpol_lab=create_vpol_vector(inc_dir)
            vpol_lab=np.tile(vpol_lab,(len(idx),1))
            
#            vpol_lab=create_vpol_vector(vertices_rot[idx])
            
            #======================================================
            # Exploit symmetry
            
            idx_sym=idx.copy()
            
            N_sym=np.zeros(len(idx))
            V_sym=np.ones((len(idx),3))
            VTEST=np.zeros((len(idx),3))
            
            
            #find symmetries
            if sym_XYobj==True or sym_XZobj==True or sym_YZobj==True:
            
                for i in range(0,len(idx)):
                    
                    vt=np.copy(vertices[idx[i],:])
                    vt=vt.flatten()
                    
                    #Number of symmetries
#                    N_sym=0
                    
                    if sym_XYobj==True and vt[2]<0:
                        vt[2]=-vt[2]
                        N_sym[i]=N_sym[i]+1
                        V_sym[i,2]=-V_sym[i,2]
                        
                    if sym_XZobj==True and vt[1]<0:
                        vt[1]=-vt[1]
                        N_sym[i]=N_sym[i]+1
                        V_sym[i,1]=-V_sym[i,1]
                        
                    if sym_YZobj==True and vt[0]<0:
                        vt[0]=-vt[0]
                        N_sym[i]=N_sym[i]+1                    
                        V_sym[i,0]=-V_sym[i,0]
                    
                    VTEST[i,:]=vt
                    
                    [idx_sym[i],bla]=ico.find_nearest_Nvertices(vertices,vt,N=1)
                    
            logic=(idx_sym!=idx)
            use_symmetry=int(np.ceil(np.sum(logic)/len(logic)))
            
            
            
            if use_symmetry==1:
                
                raise Exception('Symmetry is still under construction...sorry!');                
                
#                #construct mueller and extinction by symmetry
#                [mueller_mat_sym,ext_mat_sym]=construct_by_symmetry(
#                        mueller_array_temp[idx_sym,:,:,:],
#                        ext_mat_array_temp[idx_sym,:,:],
#                        V_sym)
#                
#                #rotate original vpol
#                vpol_rot=ico.rotate_vertices(vpol[idx,:],alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180) 
#                
#                #map to new laboratory vpol
#                [mueller_mat_rot,ext_mat_rot]=pol_changer(vertices_rot[idx,:], vpol_rot,
#                        vpol_lab ,mueller_mat_sym,ext_mat_sym)
#
#
#                #interpolated value
#                EXTTEMP=ZERO_EXT.copy()
#                MUETEMP=ZERO_MUE.copy()
#                
#                for i in range(0,len(idx)):
#
#                    Ksym=ext_mat_rot[i,:,:];
#                    Msym=mueller_mat_rot[i,:,:,:]           
#                            
#                    EXTTEMP=w_int[i]*Ksym+EXTTEMP
#                    MUETEMP=w_int[i]*Msym+MUETEMP                                     
                
            
            # end of symmetry exploitation
            #======================================================
            else:            
                    
                
#                print('alpha = '+str(alpha))
#                print('beta  = '+str(beta))
#                print('gamma = '+str(gamma))
                
                
                #rotate original vpol
                vpol_rot=ico.rotate_vertices(vpol[idx,:],alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)   
                
                
                #map to new laboratory vpol
                mueller_array_temp_sub=[mueller_array[i] for i in idx]
                
                # reshape mueller_array to common sph_expansion
                for i in range(0,len(mueller_array_temp_sub)):
                                
                    
#                    print('theta: '+str(np.arccos(inc_dir[2])*180/np.pi)+'deg - reshaping M: index '+str(i))
                    
                    n_idx=np.array([len(j) for j in mueller_array_temp_sub[i]])
                    
                    for j in range(0,len(n_idx)):
                    
                        mueller_array_temp_sub[i][j]=spha.map_sphexp(mueller_array_temp_sub[i][j],sph_object) 
                
                

                [mueller_mat_rot,ext_mat_rot]=pol_changer(vertices_rot[idx,:],
                                vpol_rot,vpol_lab ,
                                mueller_array_temp_sub,ext_mat_array[idx,:,:],sph_object)
                
#                spha.visualize_mueller_matrix_sph(mueller_array_temp_sub[0][0],sph_object,max_len_coeffs,max_ntrunc)

                #interpolated value
                EXTTEMP=ZERO_EXT.copy()
                MUETEMP=ZERO_MUE.copy()
                
                            
                for i in range(0,len(idx)):
                    
                    n_idx=np.array([len(j) for j in mueller_mat_rot[i]])
                    
                    mueller_temp=ZERO_MUE.copy()
                    
#                    spha.visualize_mueller_matrix_sph(mueller_array_temp_sub[i][0],sph_object,max_len_coeffs,max_ntrunc)
                    
                    for j in range(0,len(n_idx)):
                    
                        mueller_temp[j,:]=mueller_mat_rot[i][j] 
#                        mueller_temp[j,:]=spha.map_sphexp(mueller_mat_rot[i][j],sph_object) 
                    
                                
                    EXTTEMP=w_int[i]*ext_mat_rot[i,:,:]+EXTTEMP
                    MUETEMP=w_int[i]*mueller_temp+MUETEMP


            #do the gamma summation
            ext_mat_gamma_temp=EXTTEMP*gauss_weights[g_i]/2+ext_mat_gamma_temp
            mue_mat_gamma_temp=MUETEMP*gauss_weights[g_i]/2+mue_mat_gamma_temp

            
        #do the alpha summation
        ext_mat_alpha_temp=ext_mat_gamma_temp*gauss_weights[a_i]/2+ext_mat_alpha_temp
        mue_mat_alpha_temp=mue_mat_gamma_temp*gauss_weights[a_i]/2+mue_mat_alpha_temp   
          


    print()    
    print('theta: '+str(np.arccos(inc_dir[2])*180/np.pi)+'deg')
    print('Yeah, it is done!')
    return [mue_mat_alpha_temp,ext_mat_alpha_temp]
#    return [1,2]



def rawdata2average(inputdatapath, incidence_angles, p_orientation_beta,
                    outputdatapath=None, alpha_0=0, beta_0=0, gamma_0=0, 
                    parallel=False,num_cores=8):
    #
    #function to do the orientational average for one particle for different 
    #   incidence angles and particle tilt (beta) angles
    #   inputdatapath is the fullpath to the ncfile with the results from a adda 
    #   calculation for oriented particles
    #   incidence_angles are a 1d numpy arrays and the angles have to be in degree
    #   p_orientation_beta are a 1d numpy arrays and the angles have to be in degree
    

    
    #Check if path exists    
    if outputdatapath is not None and not os.path.exists(outputdatapath):
            
#        if not os.path.exists(outputdatapath):
            os.makedirs(outputdatapath)
    
    
    
    
#    num_cores = multiprocessing.cpu_count()
#    
#    #uncomment if number of core should be limited
#    if num_cores>16:
#        num_cores=16
    
    
    #==============================================================================
    # load data
    #==============================================================================
    
    os.chdir(inputdatapath)

    
    #get datalist of overview file
    overviewlist = glob.glob(inputdatapath+"*overview_data.mat")
    
    
    if np.size(overviewlist)>1:
        print('There was more than one overview file!')
        print('This is bad, because I am confused now...')
        
        #insert error statement here
        raise RuntimeError('Therefore, I have to stop, sorry...')
        
    elif np.size(overviewlist)<1:
        print('There was no overview file!')
        print('This is bad, because I do not know how to go on.')
        
        #insert error statement here
        raise RuntimeError('Therefore, I have to stop, sorry...')
        
    
    
    #load overviewdata
    temp=acf.loadmat(overviewlist[0])
    overviewdata=temp['output_data']
    
    
    #extinction matrix array
    ext_mat_array=overviewdata['ext_mat_array']
    
    #vertex array
    vertices=overviewdata['vertices']
    
    #symmetries
    sym_XYobj=overviewdata['sym_XYobj']
    sym_XZobj=overviewdata['sym_XZobj']
    sym_YZobj=overviewdata['sym_YZobj']
#    sym_XYobj=True
#    sym_XZobj=True
#    sym_YZobj=True

    #maximum length of spherical harmonics coefficients per particle scan
    max_len_coeffs=overviewdata['max_len_coeffs']
    
    max_ntrunc=overviewdata['max_ntrunc']

    
    #load mueller matrix
    
    print('loading adda results...')
    mueller_array=pickle.load(open(inputdatapath+overviewdata['mueller_matrix_filename']+'.pkl','rb'))
    
    #reduce mueller_array from complex128 to complex64
    for i in range(len(mueller_array)):
                
        mueller_array[i]=[np.complex64(mueller_array[i][j]) for j in range(len(mueller_array[i]))]

    
    
    dump=np.zeros(len(mueller_array))
    for i in range(len(mueller_array)):
        dump[i]=sum([len(n) for n in mueller_array[i]])
        
    #==============================================================================
    # some preperations
    #==============================================================================
    
    #get shape
    q_ext=np.shape(ext_mat_array)
    
    
    #get refinement level
    ref_lev=np.size(overviewdata['eps_vec'])-sum(np.isnan(overviewdata['eps_vec']))            
                   
    #allocate                   
    ext_mat_horiz_avg=np.zeros((len(p_orientation_beta),len(incidence_angles),q_ext[1],q_ext[2])) 
    ext_mat_horiz_avg[:]=np.nan
    mue_mat_horiz_avg_sph=np.zeros((len(p_orientation_beta),len(incidence_angles),16,max_len_coeffs),dtype=complex) 
    
    #create vertical polarization vectors
    
    
    vpol=create_vpol_vector(vertices)

    
                     
    #rotate particle to a another basic orientation
    
    if (sym_XYobj==False and sym_XZobj==False and sym_YZobj==False and
        (alpha_0!=0 or beta_0!=0 or gamma_0!=0) ):
    
#        vertices_old=vertices.copy()
#        vpol_old=vpol.copy()

        #create spherical harmonics object
        sph_object=shtns.sht(max_ntrunc, max_ntrunc, 1, shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)
        
        vertices=ico.rotate_vertices(vertices,alpha_0*np.pi/180,beta_0*np.pi/180,gamma_0*np.pi/180)
        vpol=ico.rotate_vertices(vpol,alpha_0*np.pi/180,beta_0*np.pi/180,gamma_0*np.pi/180)
                

        #create lab vertical polarization vectors
        vpol_lab=create_vpol_vector(vertices)

        #map polarizationto new polarization basis
        [mueller_array,ext_mat_array]=pol_changer(
                vertices, vpol, vpol_lab, mueller_array,ext_mat_array,sph_object)
        
        #change vpol to the new vpol
        vpol=vpol_lab.copy()
        
    elif ((alpha_0!=0 or beta_0!=0 or gamma_0 !=0 ) and 
          (sym_XYobj==True or sym_XZobj==True or sym_YZobj==True)):
        print('As symmetry was used, there is no possibility to change')
        print('the basic orientation! Sorry.')
                
    
                     
    #==============================================================================
    #       ###do the averaging
    #==============================================================================          
    
    #counter beta
    n_beta=-1

    
    #Loop over different beta angles
    for beta in p_orientation_beta:
    
        n_beta=n_beta+1
    
        print()
        print('n_beta: '+str(n_beta))
        print('beta: '+str(beta)+' deg')
        
        
        with ty.utils.Timer():
       
               
#==============================================================================
#   parallel loop, 
#==============================================================================
        
            if parallel==True:
    
                spawn_command('free -m')
                print(f'Before parallel: {memstring()}')
                print('ncores: '+str(num_cores))
                print('now it\'s time parallize.../\/\\//\\/')
                print('                           ||||||||||')
                
     
                results = Parallel(n_jobs=num_cores,max_nbytes=None)(delayed(orientational_avg)
                             (vertices, vpol,mueller_array, ext_mat_array,
                              max_len_coeffs, max_ntrunc, 
                              np.array([np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)]), beta,
                                               ref_lev,sym_XYobj, sym_XZobj, sym_YZobj) 
                             for theta in incidence_angles)
                 
                for i in range(0,len(results)):
                    ext_mat_horiz_avg[n_beta,i,:,:]=results[i][1]
                    mue_mat_horiz_avg_sph[n_beta,i,:,:]=np.array([results[i][0]])
                    
     
    
#==============================================================================
#   serial loop,
#==============================================================================
     
            elif parallel==False:
        
                print('Parallel computing is not working!')
                print('Then I switch to serial computing.')
                spawn_command('free -m')
                print(f'Before ori-avg: {memstring()}')
                
                
                #counter theta
                n_theta=-1
         
                #Loop over incidence directions
                for theta in incidence_angles:
          
                    n_theta=n_theta+1
          
                    inc_dir=np.array([np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)])
          
                    [mue_mat_alpha_temp,ext_mat_alpha_temp]=orientational_avg( 
                              vertices, vpol, mueller_array, ext_mat_array, 
                              max_len_coeffs, max_ntrunc,
                              inc_dir, beta, 
                              ref_lev,sym_XYobj, sym_XZobj, sym_YZobj)
          
                    ext_mat_horiz_avg[n_beta,n_theta,:,:]=ext_mat_alpha_temp                
                    mue_mat_horiz_avg_sph[n_beta,n_theta,:,:]=mue_mat_alpha_temp

    
#==============================================================================
#        
#==============================================================================

    #Convert extmat from µm^2 to m^2
    ext_mat_horiz_avg=ext_mat_horiz_avg*1e-12;
    

    #prepare data for save as overview of run
    output_data={}
    output_data['theta_inc']=incidence_angles
    output_data['beta']=p_orientation_beta
    output_data['Temperature']=overviewdata['Temperature']
    output_data['Frequency']=overviewdata['Frequency']
    output_data['wavelength']=overviewdata['wavelength']
    output_data['diameter_volume_equ']=overviewdata['diameter_volume_equ']
    output_data['diameter_max']=overviewdata['diameter_max']
    output_data['refr_index']=overviewdata['refr_index']
    output_data['sym_XYobj']=overviewdata['sym_XYobj']
    output_data['sym_XZobj']=overviewdata['sym_XZobj']
    output_data['sym_YZobj']=overviewdata['sym_YZobj']
    output_data['ext_mat_horiz_avg']=ext_mat_horiz_avg
    output_data['mue_mat_horiz_avg_sph']=mue_mat_horiz_avg_sph
    output_data['eps_vec']=overviewdata['eps_vec']
    output_data['delta']=overviewdata['delta']
    output_data['limit_sph']=overviewdata['limit_sph']
    output_data['numQuadraturePoints']=overviewdata['numQuadraturePoints']
    output_data['vertices']=vertices
    output_data['triangles']=overviewdata['triangles']
    output_data['shape_name']=overviewdata['shape_name']
    output_data['ext_mat_array']=ext_mat_array
    output_data['alpha_0']=alpha_0
    output_data['beta_0']=beta_0
    output_data['gamma_0']=gamma_0
    output_data['max_len_coeffs']=overviewdata['max_len_coeffs']
    output_data['max_ntrunc']=overviewdata['max_ntrunc']
    
    #
    
    if outputdatapath is not None:
    
        ##Save output in mat format
        matfile_name=os.path.join(outputdatapath,overviewdata['shape_name']+'_horiz_avg.mat')
        
        print('   ')
        print('===============================================================')
        print('shape = '+overviewdata['shape_name'])
        print('Temperature = '+str(overviewdata['Temperature'])+' K')
        print('Frequency = '+str(overviewdata['Frequency'])+' GHz')
        print('Size = '+str(overviewdata['diameter_volume_equ'])+' m')
        print('===============================================================')
        print('   ')
        
        
        sio.savemat(matfile_name, {'output_data':output_data}, appendmat=True, format='5', long_field_names=True)
    else:
        
        
        
        print('No data was saved.')

    return output_data

