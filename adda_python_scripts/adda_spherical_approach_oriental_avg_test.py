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
import typhon.plots 
from netCDF4 import Dataset

#==============================================================================
# define Polarizers 
#==============================================================================

def rotater_mat(eta):
    #Important eta has to be in rad!!

#    P=np.ones((4,4))
#
#    P[1,0]=np.cos(2*eta)
#    P[0,1]=np.cos(2*eta)
#    P[3,0]=np.cos(2*eta)
#    P[0,3]=np.cos(2*eta)    
#    P[2,1]=np.cos(2*eta)    
#    P[1,2]=np.cos(2*eta)
#    P[3,2]=np.cos(2*eta)    
#    P[2,3]=np.cos(2*eta)    


    P=np.zeros((4,4))
    P[0,0]=1
    P[1,1]=np.cos(2*eta)
    P[2,1]=np.sin(2*eta)
    P[1,2]=-P[2,1]
    P[2,2]=P[1,1]
    P[3,3]=1

    return P


##special treatment of the poles

def rotatepoles(vertices, mueller_array, ext_mat_array,alpha):
    #Important alpha has to be in rad!!
    
    mueller_array_local=mueller_array.copy()
    ext_mat_array_local=ext_mat_array.copy()
    
    
    #shape of mueller and extinction matrix
    qm=np.shape(mueller_array_local)
    q_ext=np.shape(ext_mat_array_local)
    
    #Top pole
    Tp=np.array([0.,0.,1.])
    
    #bottom pole
    Bp=np.array([0.,0.,-1.])
    
    
    #get the vertex of the top pole
    [idx_Tp,dist1]=ico.find_nearest_Nvertices(vertices, Tp)
    
    #get the vertex of the bottom pole
    [idx_Bp,dist2]=ico.find_nearest_Nvertices(vertices, Bp)

    if dist1<1e-7 and dist2<1e-7:

        #for the extinction matrix, the problem is similar to rotate the stokes vector
        
        
        # pole matrices
        ext_mat_array_poles=np.vstack((ext_mat_array_local[idx_Tp,:,:],ext_mat_array_local[idx_Bp,:,:]))    
        mueller_array_poles=np.vstack((mueller_array_local[idx_Tp,:,:,:],mueller_array_local[idx_Bp,:,:,:]))
    
        #polarzer matrix
        P1=rotater_mat(alpha)
        P2=rotater_mat(-alpha)
    
        
    #        mueller_temp=np.reshape(mueller_array_poles,(2,qm[1],qm[2],4,4))
    #        mueller_temp=np.dot(mueller_temp,P1)    
    #        mueller_temp=np.transpose(np.dot(np.transpose(mueller_temp,(0,1,2,4,3)),P2.T),(0,1,2,4,3))    
    #        mueller_array_poles=np.reshape(mueller_temp,(2,qm[1],qm[2],qm[3]))
    #        
        ext_mat_array_poles=np.dot(ext_mat_array_poles,P1)   
        BA = np.transpose(np.dot(np.transpose(ext_mat_array_poles,(0,2,1)),P2.T),(0,2,1))
        ext_mat_array_poles=BA
        
        
        #for the mueller matrix the the solution is different from the extinction matrix. 
        #Basically, we already have all the information we need. the rotation around the poles
        #is actually just a shifting of the Muellermatrix by a specifice value in the azimuth.
        #Practically, this means we have just to shift the order in the azimuth dimension of
        #the mueller array. 
        #Important assumption the Mueller matrix contains the full azimuth direction meaning
        #from 0 to 360 deg (0 to 2*pi). If not this method will not work!!!
        #azimuth dimension is the third dimension of mueller array
               
        
        #rad per bin
        rpb=np.pi*2/qm[2]       
    
        #estimate how many bins we have to shift
        binshift=int(np.round( alpha/rpb ))
        
        #shift the bins        
        mueller_array_poles=np.roll(mueller_array_poles,binshift,axis=2)
        
        #COMMENT!!!
        #It would be more accurate but also slower to interpolate the mueller
        #matrix to the new coordinates, instead of simply shifting the bins. 
        #I think it is not needed, because we are interested in azimuthal averages
        #and the error that we made by shifting the bins should cancel out due to the
        #averaging
    
        mueller_array_local[idx_Tp,:,:,:]=mueller_array_poles[0,:,:,:]
        mueller_array_local[idx_Bp,:,:,:]=mueller_array_poles[1,:,:,:]
        
        ext_mat_array_local[idx_Tp,:,:]=ext_mat_array_poles[0,:,:]
        ext_mat_array_local[idx_Bp,:,:]=ext_mat_array_poles[1,:,:]
    
    
    return mueller_array_local, ext_mat_array_local     



#==============================================================================
#  define average function
#==============================================================================

def orientational_avg( vertices, mueller_array, ext_mat_array, inc_dir, beta,
                      ref_lev, sym_XYobj=False, sym_XZobj=False,sym_YZobj=False):
    #
    #function to do the orientational average for one particle for a specific 
    #   incidence angle and a specific particle tilt (beta) angle
    
                                                     
    print()
    print('theta: '+str(inc_dir))                                                     

    #get the number of quadrature points
    numQuadraturePoints=16#4*2**ref_lev

    #get the weights for gauss legendre quadrature
    [sample_points,gauss_weights]=np.polynomial.legendre.leggauss(numQuadraturePoints)
    
    #angles to be evaluated
    p_orientation_alpha=sample_points*180;
    p_orientation_gamma=sample_points*180;
    
    
    
    #get the shape
    qm=np.shape(mueller_array)
    q_ext=np.shape(ext_mat_array)

    
    #symmetry matrices
    Ma=np.ones([2,2])
    Mb=-np.ones([2,2])
    M=np.vstack((np.hstack((Ma,Mb)),np.hstack((Mb,Ma))))
    
    #for extinction
    M_k=np.copy(M)
    
    #for mueller
    M_m=np.reshape(M,(1,1,qm[3]))
    M_m=np.tile(M_m,(qm[1],qm[2],1))
    
    
    
    #zero matrix
    ZERO_EXT=np.zeros((q_ext[1],q_ext[2]))
    ZERO_MUE=np.zeros((qm[1],qm[2],qm[3]))

    mue_mat_alpha_temp=ZERO_MUE
    ext_mat_alpha_temp=ZERO_EXT
    
    
    ext_mat_alter=np.zeros((q_ext[1],q_ext[2],numQuadraturePoints,numQuadraturePoints))

    #counter alpha
    n_alpha=-1
    
    test=np.zeros((5,len(p_orientation_alpha)))
    
    for a_i in range(0,len(p_orientation_alpha)):
        
          
        ext_mat_galter=np.zeros((q_ext[1],q_ext[2],numQuadraturePoints))

        n_alpha=n_alpha+1
        alpha=p_orientation_alpha[a_i]


        mue_mat_gamma_temp=ZERO_MUE
        ext_mat_gamma_temp=ZERO_EXT

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
            
#            print()
#            print('n_alpha: '+str(n_alpha)+', n_gamma: '+str(n_gamma))
            

            #rotate and interpolate adda sphere
            vertices_rot=ico.rotate_vertices(vertices,alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)
            
            #the poles are treated special, due to the ambiguity with the azimuth direction,
            #therfore the muellermatrix and the extinction matrix of the poles are mutiplied with
            #a polarization matrix to consider the the fact that though the direction is not chaging
            #when beta is zero and only alpha rotation happens the polarization changes.
            [mueller_array_temp, ext_mat_array_temp]=rotatepoles(vertices, mueller_array, ext_mat_array,alpha*np.pi/180+gamma*np.pi/180)
            
 
            [w_int,idx]=ico.interpolation_weights_icosahedric_grid(vertices_rot,inc_dir)

            test[0,g_i]=ext_mat_array_temp[idx[0],0,0]
            test[1,g_i]=ext_mat_array_temp[idx[0],0,1]
            test[2,g_i]=idx[0]
            test[3,g_i]=gamma
            test[4,g_i]=w_int[0]
            
            #interpolated value
            EXTTEMP=ZERO_EXT.copy()
            MUETEMP=ZERO_MUE.copy()
            for i in range(0,len(idx)):
                
                #======================================================
                #    Exploit symmetry
                if ((vertices[idx[i],2]<0  and  sym_XYobj==True) or 
                    (vertices[idx[i],1]<0  and  sym_XZobj==True) or
                    (vertices[idx[i],0]<0  and  sym_YZobj==True)):    

                    vt=np.copy(vertices[idx[i],:])
                    vt.flatten(1)
                    
                    #Number of symmetries
                    N_sym=0
                    
                    if sym_XYobj==True and vt[2]<0:
                        vt[2]=-vt[2]
                        N_sym=N_sym+1
                    if sym_XZobj==True and vt[1]<0:
                        vt[1]=-vt[1]
                        N_sym=N_sym+1
                    if sym_YZobj==True and vt[0]<0:
                        vt[0]=-vt[0]
                        N_sym=N_sym+1
                        
                        
                    [idx_sym,bla]=ico.find_nearest_Nvertices(vertices,vt,N=1)
                    
                    Ksym=ext_mat_array_temp[idx_sym[0],:,:]*M_k**N_sym
                    Msym=mueller_array_temp[idx_sym[0],:,:,:]*M_m**N_sym
#                            
                    EXTTEMP=w_int[i]*Ksym+EXTTEMP
                    MUETEMP=w_int[i]*Msym+MUETEMP                                     
                    
                
                # end of symmetry exploitation
                #======================================================
                else:
                
                    EXTTEMP=w_int[i]*ext_mat_array_temp[idx[i],:,:]+EXTTEMP
                    MUETEMP=w_int[i]*mueller_array_temp[idx[i],:,:,:]+MUETEMP
                                 
            ext_mat_alter[:,:,a_i,g_i]=EXTTEMP
            ext_mat_galter[:,:,g_i]=EXTTEMP

            #do the gamma summation
            ext_mat_gamma_temp=EXTTEMP*gauss_weights[g_i]/2+ext_mat_gamma_temp
            mue_mat_gamma_temp=MUETEMP*gauss_weights[g_i]/2+mue_mat_gamma_temp

            
        #do the alpha summation
        ext_mat_alpha_temp=ext_mat_gamma_temp*gauss_weights[a_i]/2+ext_mat_alpha_temp
        mue_mat_alpha_temp=mue_mat_gamma_temp*gauss_weights[a_i]/2+mue_mat_alpha_temp   
          
                                                                   
        ext_mat2=np.mean(np.mean(ext_mat_alter,axis=3),axis=2)

    print('bla')
    return [mue_mat_alpha_temp,ext_mat_alpha_temp]
#    return [1,2]



def rawdata2average(inputdatapath, incidence_angles, p_orientation_beta,
                    outputdatapath=None, alpha_0=0, beta_0=0, gamma_0=0):
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

    
    #load mueller matrix
    data=Dataset(inputdatapath+overviewdata['mueller_matrix_filename'])
    
    mueller_array=data.variables['mueller_matrix_array'][:]
    
    
    
    #==============================================================================
    # some preperations
    #==============================================================================
    
    qm=np.shape(mueller_array)
    q_ext=np.shape(ext_mat_array)
    
    
    #get refinement level
    ref_lev=np.size(overviewdata['eps_vec'])-sum(np.isnan(overviewdata['eps_vec']))-1            
                   
                   
    ext_mat_horiz_avg=np.zeros((len(p_orientation_beta),len(incidence_angles),q_ext[1],q_ext[2])) 
    ext_mat_horiz_avg[:]=np.nan
    
    mue_mat_horiz_avg=np.zeros((len(p_orientation_beta),len(incidence_angles),qm[1],qm[2],qm[3])) 
    mue_mat_horiz_avg[:]=np.nan
    
                     
    #rotate particle to a another basic orientation
    
    if sym_XYobj==False and sym_XZobj==False and sym_YZobj==False:
    
        vertices=ico.rotate_vertices(vertices,alpha_0*np.pi/180,beta_0*np.pi/180,gamma_0*np.pi/180)
        
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

##==============================================================================
##   serial loop, uncomment if parallel computing is not possible
##==============================================================================
# 
#        #counter theta
#        n_theta=-1
# 
#        #Loop over incidence directions
#        for theta in incidence_angles:
#  
#            n_theta=n_theta+1
#  
#            inc_dir=[np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)]
#  
#            [mue_mat_alpha_temp,ext_mat_alpha_temp]=orientational_avg( vertices,
#                      mueller_array, ext_mat_array, inc_dir, beta, 
#                      ref_lev,sym_XYobj, sym_XZobj, sym_YZobj)
#  
#            ext_mat_horiz_avg[n_beta,n_theta,:,:]=ext_mat_alpha_temp
#            mue_mat_horiz_avg[n_beta,n_theta,:,:,:]=mue_mat_alpha_temp
# 
#                               
##==============================================================================

#==============================================================================
#   parallel loop, uncomment if parallel computing is possible
#==============================================================================
         
 
        results = Parallel(n_jobs=num_cores,max_nbytes=None)(delayed(orientational_avg)
                     (vertices,mueller_array, ext_mat_array, 
                      [np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)], beta,
                                       ref_lev,sym_XYobj, sym_XZobj, sym_YZobj) 
                     for theta in incidence_angles)
         
        for i in range(0,len(results)):
            ext_mat_horiz_avg[n_beta,i,:,:]=results[i][1]
            mue_mat_horiz_avg[n_beta,i,:,:,:]=results[i][0]
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
    output_data['mue_mat_horiz_avg']=mue_mat_horiz_avg
    output_data['eps_vec']=overviewdata['eps_vec']
    output_data['delta']=overviewdata['delta']
    output_data['limit_sph']=overviewdata['limit_sph']
    output_data['numQuadraturePoints']=overviewdata['numQuadraturePoints']
    output_data['vertices']=vertices
    output_data['triangles']=overviewdata['triangles']
    output_data['shape_name']=overviewdata['shape_name']
    output_data['ext_mat_array']=ext_mat_array
    #
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








    
#==============================================================================
# symmetries and particle orientation angular grids
#==============================================================================


#only beta, the number of alpha and gamma angles will be set according to
# refinement level 
p_orientation_beta=np.linspace(0,180.,num=1)

incidence_angles=np.linspace(0.,180.,num=19)


#==============================================================================
# storage preperations
#==============================================================================



inputdatapath=('/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/rawdata/ellipsoid_1.0_0.9999990000010001_0-0-0_deq02607um_freq183.0GHz_T0253K/')
#inputdatapath=('/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/rawdata/ellipsoid_1.0_0.5_0-90-0_deq00095um_freq100.0GHz_T0253K/')
#inputdatapath=('/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/rawdata/ellipsoid_1.0_0.5_0-0-0_deq00200um_freq183.0GHz_T0253K/')


outputdatapath=('/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/orientational_avg/ellipsoid_1.0_0.9999990000010001_0-0-0_deq02607um_freq183.0GHz_T0253K/')
#outputdatapath=('/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/orientational_avg/ellipsoid_1.0_0.5_0-90-0_deq00095um_freq100.0GHz_T0253K/')
#outputdatapath=('/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/orientational_avg/ellipsoid_1.0_0.5_0-0-0_deq00200um_freq183.0GHz_T0253K/')



num_cores = multiprocessing.cpu_count()

if num_cores>8:
    num_cores=16

alpha_0=0
beta_0=0
gamma_0=0


output_data=rawdata2average(inputdatapath, incidence_angles, p_orientation_beta,
                            outputdatapath,
                            alpha_0,beta_0,gamma_0)


    
#==============================================================================
#       ###For development
#==============================================================================    

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


ext_mat_horiz_avg=output_data['ext_mat_horiz_avg']
mue_mat_horiz_avg=output_data['mue_mat_horiz_avg']
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
h1=ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2], c=ext_mat_array[:,0,0], marker='o',
               cmap='density')
plt.hold(False)
cbar = fig.colorbar(h1)



fig = plt.figure()
ytest=ext_mat_horiz_avg[0,:,0,0]   


plt.subplot(2, 1, 1)
plt.plot(incidence_angles,ytest,'bx-')      
plt.xlabel('\theta [deg]')
plt.ylabel('K(1,1) [m^2]')

plt.subplot(2, 1, 2)
ytest=ext_mat_horiz_avg[0,:,1,0]    
plt.plot(incidence_angles,ytest,'gd-')      
plt.xlabel('\theta [deg]')
plt.ylabel('K(2,1) [m^2]')


fig = plt.figure()
ytest=mue_mat_horiz_avg[0,:,0,0,0]   
plt.plot(incidence_angles,ytest,'rd-')      
plt.xlabel('\theta [deg]')
plt.ylabel('M(1,1)')
    

#==============================================================================
#       ### END For development
#==============================================================================    

