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
import adda_orientational_avg_spharmonics_functions as aoa
import typhon as ty
import scipy.io as sio
import spherical_harmonics_for_adda as spha
import sys

#==============================================================================
# symmetries, particle orientation angular grids and other stuff
#==============================================================================

#beta angles
p_orientation_beta=np.linspace(30,90.,num=1)

#incidence angles
incidence_angles=np.linspace(0,180.,num=5)

#create regular grid
regular_phasematrix=True

#plot data
plotting=True

#theta index for plotting
theta_idx=1

#grid size
grid_size=37

alpha_0=0
beta_0=0
gamma_0=0

#number of cores
ncores=16

#eps reexpand
eps=0.005
#==============================================================================
# paths
#==============================================================================

#
#rawdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/ellipsoid_spehere_ansatz/rawdata_thunder/'
#avgdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/ellipsoid_spehere_ansatz/orientational_avg_sphtest/'
#rawdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/ellipsoid_spehere_ansatz/rawdata/'
#avgdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/ellipsoid_spehere_ansatz/orientational_avg/'


#rawdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/ExampleAggregates/'
#avgdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/ExampleAggregates/orientational_avg_test/'

#rawdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/LiuShape887/'
#avgdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/LiuShape887/orientational_avg_test_37betas/'

#rawdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/HexPlaAggCrystal_Id20/'
#avgdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/HexPlaAggCrystal_Id20_horizavg/'
#
#rawdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/LiuSectorSnowflake_Id3/'
#avgdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/LiuSectorSnowflake_Id3_horizavg/'

#rawdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/HongColumn_Id7/'
#avgdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/HongColumn_Id7_horizavg/'

rawdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/HongPlate_Id9/'
avgdatapath='/scratch/uni/u237/user_data/mbrath/adda_results/HongPlate_Id9_horizavg/'

#==============================================================================
# preperations
#==============================================================================





#look after the adda output
folderlist = glob.glob(rawdatapath+'*_deq*')
folderlist.sort()

counter=-1
for folder in folderlist:
    
    counter=counter+1
    folder=os.path.basename(folder)
    print(folder+' ('+str(counter)+')')
    
print('======================================================================')    
print('choose data')    
print('type the number within in brackets')
print('or type "all" for all data.')
index = input(' ==>>  ')    
#index='all'

#index=sys.argv[1]

if index!='all':
    folderlist=[folderlist[int(index)]]


#folderlist=[folderlist[index]]


t=ty.utils.Timer().start()

counter=-1
for folder in folderlist:
    counter=counter+1
    
    folder=os.path.basename(folder)
    
    f_sep=folder.split('_')
    
#    if f_sep[0].lower()=='ellipsoid' and float(f_sep[1])<float(f_sep[2]):
#        beta_0t=90.
#    else:
#        beta_0t=beta_0
        
    beta_0t=beta_0    
    
    print(folder)
    print('beta = '+str(beta_0t))
    
    inputdatapath=rawdatapath+folder+'/'
    outputdatapath=avgdatapath+folder+'/'

    
    #do the averaging
    output_data=aoa.rawdata2average(inputdatapath, incidence_angles, p_orientation_beta,
                                outputdatapath,
                                alpha_0,beta_0t,gamma_0,parallel=True,num_cores=ncores)

#==============================================================================
#   faster transformation, but less accurate in forward direction  
#==============================================================================
#    #transform to phase matrix
#    #without reexpanding, faster but less accurate at the poles (forward direction)
#    phase_mat=aoa.mueller2phase(output_data) 
#    output_data['phase_matrix_sph']=phase_mat
# 
#==============================================================================

#==============================================================================
#   slower transformation, but more accurate in forward direction  
#==============================================================================

    #incidence angle
    theta_inc_mue=output_data['theta_inc']
    
    #truncation number of spherical harmonics expansion
    max_ntrunc_mue=output_data['max_ntrunc']
    
    #transform to phase matrix
    [phase_mat, max_ntrunc_pha, max_len_coeffs_pha]=aoa.mueller2phase_reexpand(output_data,grid_size_theta=46,eps=eps,parallel=True)
    
    
    
    phase_mat=spha.reshape_sphexp_2_common_exp(np.max(max_ntrunc_pha[:]),
                                               np.max(max_len_coeffs_pha[:]),phase_mat)
    
    
    
    
    output_data['eps_phase']=eps
    output_data['phase_matrix_sph']=phase_mat
    output_data['phase_matrix_sph_old']=phase_mat.copy()
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
    output_data['abs_vector_old']=abs_vector.copy()

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
    
    
    #save regular
    if regular_phasematrix==True:
        
        lmax = spha.get_lmax(output_data['phase_matrix_sph'][0,0,0,:])
        
        [phase_matrix_reg,theta_s,phi_s]=aoa.create_regular_representation(
                output_data,lmax,grid_size_theta=grid_size,gridtype='regular',full_phi=True)
        
        
        output_data['theta_s']=theta_s
        output_data['phi_s']=phi_s
        output_data['phase_matrix_reg']=np.single(phase_matrix_reg)
        
        
    #Check if path exists    
    if  not os.path.exists(outputdatapath):
            
        os.makedirs(outputdatapath)    
    
    ##Save output in mat format
    matfile_name=os.path.join(outputdatapath,output_data['shape_name']+'_horiz_avg_phasematrix.mat')
    
    print('   ')
    print('===============================================================')
    print('shape = '+output_data['shape_name'])
    print('Temperature = '+str(output_data['Temperature'])+' K')
    print('Frequency = '+str(output_data['Frequency'])+' GHz')
    print('Size = '+str(output_data['diameter_volume_equ'])+' m')
    print('===============================================================')
    print('   ')
    
    
#    sio.savemat(matfile_name, {'output_data':output_data}, appendmat=True, format='5', long_field_names=True)




    t.stop()
    
#==============================================================================
#       ###For development
#==============================================================================    

    if plotting==True:
    
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D        
        
        ext_mat_horiz_avg=output_data['ext_mat_horiz_avg']
        ext_mat_array=output_data['ext_mat_array']
        vertices=output_data['vertices']
        
        
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect("equal")
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=np.cos(u)*np.sin(v)*0.98
        y=np.sin(u)*np.sin(v)*0.98
        z=np.cos(v)*0.98
        ax.plot_wireframe(x, y, z, color="k", rstride=1, cstride=1,linewidth=0.5)
        
        plt.hold(True)
            
        h1=ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2],s=50, c=ext_mat_array[:,0,0], marker='s',
                       cmap='Blues')
        plt.hold(False)
        cbar = fig.colorbar(h1)
        plt.xlabel('x')
        plt.ylabel('y')
        
        
            
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect("equal")
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=np.cos(u)*np.sin(v)*0.98
        y=np.sin(u)*np.sin(v)*0.98
        z=np.cos(v)*0.98
        ax.plot_wireframe(x, y, z, color="k", rstride=1, cstride=1,linewidth=0.5)
        
        plt.hold(True)
        emax=np.abs(ext_mat_array[:,1,0].max())
        emin=np.abs(ext_mat_array[:,1,0].min())
        
        vmin=-np.max((emin,emax))
        vmax=np.max((emin,emax))
        
        
        h1=ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2],s=50, c=ext_mat_array[:,1,0], marker='s',
                       cmap='seismic',vmin=vmin,vmax=vmax)
        plt.hold(False)
        cbar = fig.colorbar(h1)
        plt.xlabel('x')
        plt.ylabel('y')
    
     
        
        
        fig = plt.figure()
        ytest=ext_mat_horiz_avg[0,:,0,0]   
        
        
        plt.subplot(3, 1, 1)
        plt.plot(incidence_angles,ytest,'bx-')      
        plt.xlabel('\theta [deg]')
        plt.ylabel('K(1,1) [m^2]')
        
        plt.subplot(3, 1, 2)
        ytest=ext_mat_horiz_avg[0,:,1,0]    
        plt.plot(incidence_angles,ytest,'gd-')      
        plt.xlabel('\theta [deg]')
        plt.ylabel('K(2,1) [m^2]')
        
        plt.subplot(3, 1, 3)
        ytest=ext_mat_horiz_avg[0,:,1,1]    
        plt.plot(incidence_angles,ytest,'rs-')      
        plt.xlabel('\theta [deg]')
        plt.ylabel('K(2,2) [m^2]')
        
            
        #incidence angle
        theta_inc=output_data['theta_inc']
        
        #truncation number of spherical harmonics expansion
        max_ntrunc=output_data['max_ntrunc']
    
    
        if max_ntrunc % 2 ==0:
            min_grid_size=max_ntrunc+2
        else:
            min_grid_size=max_ntrunc+1
            
        if min_grid_size<grid_size:
            min_grid_size=grid_size    
            
            

        x = spha.Spharmt(min_grid_size*2+1,min_grid_size,max_ntrunc,1,gridtype='gaussian')
        
        
        xlons,xlats = np.meshgrid(x.lons, x.lats)
        
        
        sph_coeffs=phase_mat[0,theta_idx,:,:]
        
        
        fig = plt.figure()
        
        for j in range(0,np.size(sph_coeffs,axis=0)):
            
            
            temp=sph_coeffs[j,:]  
            mjj=x.spectogrd(temp)
            
            plt.subplot(4, 4, j+1)
            h1=plt.pcolormesh(xlons,xlats,mjj, cmap=plt.cm.seismic)
        
            
            if j==0:
                plt.ylabel('theta [deg]')
                m11_max=np.max(mjj.flatten())
                
                
            if j==15: 
                plt.xlabel('phi [deg]')
            h1.set_clim(vmin=-m11_max, vmax=m11_max)
            fig.colorbar(h1)
            
        fig.suptitle('Incidence angle = '+str(theta_inc[theta_idx])+'˚')

        plt.show()