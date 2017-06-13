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
import numpy.matlib
import scipy.io as sio
import adda_control_functions as acf
import icosphere_grid as ico
from joblib import Parallel, delayed
import multiprocessing
#import typhon.plots 
#from netCDF4 import Dataset
import netCDF4
import matplotlib.pyplot as plt
import itertools
import typhon as ty
import spherical_harmonics_for_adda as spha
import shtns
import copy



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
    norm_V=np.matlib.repmat(norm_V,3,1).transpose()
    vpol=vpol/norm_V


    return vpol

#==============================================================================
# define Polarizers 
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
    #
    #function to do the orientational average for one particle for a specific 
    #   incidence angle and a specific particle tilt (beta) angle

    
    
    #get the shape
    qm=(16,max_len_coeffs)
    q_ext=np.shape(ext_mat_array)
    
    #create spherical harmonics object
    sph_object=shtns.sht(max_ntrunc, max_ntrunc, 1, shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)
    
    # reshape mueller_array to common sph_expansion
    for i in range(0,len(mueller_array)):
                    
        n_idx=np.array([len(j) for j in mueller_array[i]])
        
        for j in range(0,len(n_idx)):
        
            mueller_array[i][j]=spha.map_sphexp(mueller_array[i][j],sph_object) 
    
    
    
    
    #zero matrix
    ZERO_EXT=np.zeros((q_ext[1],q_ext[2]))
    ZERO_MUE=np.complex64(np.zeros((qm[0],qm[1]),dtype=complex))

    mue_mat_alpha_temp=ZERO_MUE.copy()
    ext_mat_alpha_temp=ZERO_EXT.copy()
    
    

    print()
    print('theta: '+str(inc_dir))

    #get the number of quadrature points
    numQuadraturePoints=2*2**ref_lev
    if numQuadraturePoints>16:
        numQuadraturePoints=16

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
            mueller_array_temp=copy.deepcopy(mueller_array)
            ext_mat_array_temp=ext_mat_array.copy()
            
            #rotate and interpolate incidence directions
            vertices_rot=ico.rotate_vertices(vertices,alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)
            
            [w_int,idx]=ico.interpolation_weights_icosahedric_grid(vertices_rot,inc_dir)
            
            #create lab vertical polarization vectors
            vpol_lab=create_vpol_vector(inc_dir)
            vpol_lab=np.matlib.repmat(vpol_lab,len(idx),1)
            
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
                    vt.flatten(1)
                    
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
                    
                #rotate original vpol
                vpol_rot=ico.rotate_vertices(vpol[idx,:],alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)   
                
                #map to new laboratory vpol
                mueller_array_temp_sub=[mueller_array_temp[i] for i in idx]

                [mueller_mat_rot,ext_mat_rot]=pol_changer(vertices_rot[idx,:],
                                vpol_rot,vpol_lab ,
                                mueller_array_temp_sub,ext_mat_array_temp[idx,:,:],sph_object)
                
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
          


    print('bla')
    return [mue_mat_alpha_temp,ext_mat_alpha_temp]
#    return [1,2]



def rawdata2average(inputdatapath, incidence_angles, p_orientation_beta,
                    outputdatapath=None, alpha_0=0, beta_0=0, gamma_0=0, 
                    num_x=73, num_y=36,parallel=False):
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
    
    
    
    
    num_cores = multiprocessing.cpu_count()
    
    #uncomment if number of core should be limited
#    if num_cores>16:
#        num_cores=16
    
    
    #==============================================================================
    # load data
    #==============================================================================
    
    os.chdir(inputdatapath)

    
    #get datalist of overview file
    overviewlist = glob.glob(inputdatapath+"*overview_data.mat")
    
    
    if np.size(overviewlist)!=1:
        print('There was more than one overview file!')
        print('This is bad, because I am confused now...')
        
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
    mueller_array=ty.arts.xml.load(inputdatapath+overviewdata['mueller_matrix_filename'])
    
#    mueller_array=data.variables['mueller_matrix_array'][:]
#    
#    phi_s=data.variables['phi_s'][:]
#    theta_s=data.variables['theta_s'][:]
    
    #==============================================================================
    # some preperations
    #==============================================================================
    
#    qm=mueller_array.shape
    q_ext=np.shape(ext_mat_array)
    
    
    #get refinement level
    ref_lev=np.size(overviewdata['eps_vec'])-sum(np.isnan(overviewdata['eps_vec']))            
                   
                   
    ext_mat_horiz_avg=np.zeros((len(p_orientation_beta),len(incidence_angles),q_ext[1],q_ext[2])) 
    ext_mat_horiz_avg[:]=np.nan
    
    mue_mat_horiz_avg_reg=np.zeros((len(p_orientation_beta),len(incidence_angles),num_y,num_x,16)) 
    mue_mat_horiz_avg_reg[:]=np.nan
    mue_mat_horiz_avg_sph=[None] * len(p_orientation_beta)
    
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
    
    #create spharmonics object for regular grid representation
    x = spha.Spharmt(num_x,num_y,max_ntrunc,1,gridtype='gaussian')
    phi_s,theta_s = np.meshgrid(x.lons, x.lats)
    
    
    
    #counter beta
    n_beta=-1

    
    #Loop over different beta angles
    for beta in p_orientation_beta:
    
        n_beta=n_beta+1
    
        print()
        print('n_beta: '+str(n_beta))
        
        
        mue_mat_horiz_avg_beta_sph=[None] * len(incidence_angles)
        
#==============================================================================
#   parallel loop, uncomment if parallel computing is possible
#==============================================================================
        
        if parallel==True:

 
            results = Parallel(n_jobs=num_cores,max_nbytes=None)(delayed(orientational_avg)
                         (vertices, vpol,mueller_array, ext_mat_array,
                          max_len_coeffs, max_ntrunc, 
                          np.array([np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)]), beta,
                                           ref_lev,sym_XYobj, sym_XZobj, sym_YZobj) 
                         for theta in incidence_angles)
             
            for i in range(0,len(results)):
                ext_mat_horiz_avg[n_beta,i,:,:]=results[i][1]
#                
                mue_mat_horiz_avg_beta_sph[i]=results[i][0]
                
                #regular grid representation     
#                mue_mat_horiz_avg_reg[n_beta,i,:,:,:]=x.spectogrd(results[i][0])
                
                for mii in range(0,16):
                           
                    mue_mat_horiz_avg_reg[n_beta,i,:,:,mii]=x.spectogrd(
                        np.complex128(results[i][0][mii,:]))
     

#==============================================================================
#   serial loop, uncomment if parallel computing is not possible
#==============================================================================
 
        elif parallel==False:
    
            print('Parallel computing is not working!')
            print('Then I switch to serial computing.')
            
            
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
                mue_mat_horiz_avg_beta_sph[n_theta]=mue_mat_alpha_temp

                #regular grid representation    

                for mii in range(0,16):
                           
                    mue_mat_horiz_avg_reg[n_beta,n_theta,:,:,mii]=x.spectogrd(
                        np.complex128(mue_mat_alpha_temp[mii,:]))
                
#==============================================================================
        
        mue_mat_horiz_avg_sph[n_beta]=mue_mat_horiz_avg_beta_sph

    
#==============================================================================
#        
#==============================================================================
 

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
    output_data['mue_mat_horiz_avg']=mue_mat_horiz_avg_reg
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
    output_data['phi_s']=phi_s[0,:]*180/np.pi
    output_data['theta_s']=theta_s[:,0]*180/np.pi
    
    #
    
    if outputdatapath is not None:
    
        ##Save output in mat format
        matfile_name=os.path.join(outputdatapath,overviewdata['shape_name']+'_horiz_avg.mat')
        
        print('   ')
        print('===============================================================')
        print('shape = '+overviewdata['shape_name'])
        print('Temperature = '+str(overviewdata['Temperature'])+' K')
        print('Frequency = '+str(overviewdata['Frequency'])+' GHz')
        print('Size = '+str(overviewdata['diameter_volume_equ'])+' µm')
        print('===============================================================')
        print('   ')
        
        
        sio.savemat(matfile_name, {'output_data':output_data}, appendmat=True, format='5', long_field_names=True)
    else:
        
        
        
        print('No data was saved.')

    return output_data

