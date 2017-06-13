#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:58:25 2015

@author: u242031
"""
#==============================================================================
# import modules
#==============================================================================

#import subprocess as sub
import glob
import os
#import scipy as sp
import numpy as np
import scipy.io as sio

#pdb.set_trace()
#from adda_config import *
import adda_control_functions as acf
import icosphere_grid as ico

import zipfile

from netCDF4 import Dataset






#==============================================================================
# symmetries and particle orientation angular grids
#==============================================================================


#only beta, the number of alpha and gamma angles will be set according to
# refinement level 
p_orientation_beta=np.linspace(0.,180.,num=1)

incidence_angles=np.linspace(0.,180.,num=19)


#==============================================================================
# storage preperations
#==============================================================================



inputdatapath=('/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/rawdata/ellipsoid_1.0_2.0_deq00095um_freq100.0GHz_T0253K/')


outputdatapath=('/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/oriental_average/ellipsoid_1.0_2.0_deq00095um_freq100.0GHz_T0253K/')

if not os.path.exists(outputdatapath):
        os.makedirs(outputdatapath)


#==============================================================================
# load data
#==============================================================================

os.chdir(inputdatapath)

#get datalist of vertices
vertexlist = glob.glob(inputdatapath+"*vertex*.zip")

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

#convergence vector
eps_vec=overviewdata['eps_vec'] 

#symmetries
sym_XYobj=overviewdata['sym_XYobj']
sym_XZobj=overviewdata['sym_XZobj']
sym_YZobj=overviewdata['sym_YZobj']

#load mueller matrix
data=Dataset(inputdatapath+overviewdata['mueller_matrix_filename'])

mueller_array=data.variables['mueller_matrix_array']
phi_s_grid=data.variables['phi_s']
theta_s_grid=data.variables['theta_s']
matrix_index=data.variables['matrix_index']
vertex_index=data.variables['vertex_index']





#allocate_flag=False
#for vidx in range(0,len(vertexlist)):
#    with zipfile.ZipFile(vertexlist[vidx]) as zf:
#        with zf.open('mueller_scatgrid') as mueller_file:
#            
#            print()
#            
#            data_temp=np.loadtxt(mueller_file, skiprows=1)
#            phis_grid=data_temp[:,1]
#            thetas_grid=data_temp[:,0]
#            
#            #make it unique
#            phis_grid=np.unique(phis_grid)
#            thetas_grid=np.unique(thetas_grid)
#
#
#
#            #Mueller matrix
#            mueller_raw_array=data_temp[:,2:]
#            qmm=mueller_raw_array.shape
#            mueller_raw_array=np.reshape(mueller_raw_array,(thetas_grid.size,
#                                                phis_grid.size,qmm[1]))
#
#
#            
#            if allocate_flag==False:
#                
#                qm=mueller_raw_array.shape
#                qv=len(vertexlist)
#                
#                mueller_array=np.empty((qv,)+qm)
#        
#                allocate_flag==True
#                
#            mueller_array[vidx,:,:,:]=mueller_raw_array




#==============================================================================
# some preperations
#==============================================================================

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


#get refinement level
ref_lev=np.size(overviewdata['eps_vec'])-sum(np.isnan(overviewdata['eps_vec']))-1            
               
               
ext_mat_horiz_avg=np.zeros((len(p_orientation_beta),len(incidence_angles),q_ext[1],q_ext[2])) 
ext_mat_horiz_avg[:]=np.nan

mue_mat_horiz_avg=np.zeros((len(p_orientation_beta),len(incidence_angles),qm[1],qm[2],qm[3])) 
mue_mat_horiz_avg[:]=np.nan

                 
#==============================================================================
#       ###do the averaging
#==============================================================================    




numQuadraturePoints=4*2**ref_lev

[sample_points,gauss_weights]=np.polynomial.legendre.leggauss(numQuadraturePoints)

p_orientation_alpha=sample_points*180;
p_orientation_gamma=sample_points*180;

#counter beta
n_beta=-1



#Loop over different beta angles
for beta in p_orientation_beta:

    n_beta=n_beta+1

    #counter theta
    n_theta=-1

    #Loop over incidence directions
    for theta in incidence_angles:

        n_theta=n_theta+1

        inc_dir=[np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)]


        #do the horizontal averaging=======================================

        mue_mat_alpha_temp=np.zeros((qm[1],qm[2],qm[3]))
        ext_mat_alpha_temp=np.zeros((q_ext[1],q_ext[2]))

        #counter alpha
        n_alpha=-1

        
        for a_i in range(0,len(p_orientation_alpha)):

            n_alpha=n_alpha+1
            alpha=p_orientation_alpha[a_i]


            mue_mat_gamma_temp=np.zeros((qm[1],qm[2],qm[3]))
            ext_mat_gamma_temp=np.zeros((q_ext[1],q_ext[2]))

            #counter gamma
            n_gamma=-1

            #sample gamma angles
            for g_i in range(0,len(p_orientation_gamma)):

                n_gamma=n_gamma+1
                gamma=p_orientation_alpha[g_i]
                
                print()
                print('n_alpha: '+str(n_alpha)+', n_gamma: '+str(n_gamma))
                

                #rotate and interpolate adda sphere
                vertices_rot=ico.rotate_vertices(vertices,alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)

                [w_int,idx]=ico.interpolation_weights_icosahedric_grid(vertices_rot,inc_dir)

                
                #interpolated value
                EXTTEMP=np.zeros((q_ext[1],q_ext[2]))
                MUETEMP=np.zeros((qm[1],qm[2],qm[3]))
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
                        
                        Ksym=ext_mat_array[idx_sym[0],:,:]*M_k**N_sym
                        Msym=mueller_array[idx_sym[0],:,:,:]*M_m**N_sym
#                            
                        EXTTEMP=w_int[i]*Ksym+EXTTEMP
                        MUETEMP=w_int[i]*Msym+MUETEMP                                     
                        
                    
                    # end of symmetry exploitation
                    #======================================================
                    else:
                    
                        EXTTEMP=w_int[i]*ext_mat_array[idx[i],:,:]+EXTTEMP
                        MUETEMP=w_int[i]*mueller_array[idx[i],:,:,:]+MUETEMP

                #do the gamma summation
                ext_mat_gamma_temp=EXTTEMP*gauss_weights[g_i]/2+ext_mat_gamma_temp
                mue_mat_gamma_temp=MUETEMP*gauss_weights[g_i]/2+mue_mat_gamma_temp


                
#            #copy
#            ext_mat_gamma=ext_mat_gamma_temp


            #do the alpha summation
            ext_mat_alpha_temp=ext_mat_gamma_temp*gauss_weights[a_i]/2+ext_mat_alpha_temp
            mue_mat_alpha_temp=mue_mat_gamma_temp*gauss_weights[a_i]/2+mue_mat_alpha_temp                                                               

            

        ext_mat_horiz_avg[n_beta,n_theta,:,:]=ext_mat_alpha_temp
        mue_mat_horiz_avg[n_beta,n_theta,:,:,:]=mue_mat_alpha_temp

    
#==============================================================================
#       ###For development
#==============================================================================    

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("equal")

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x=np.cos(u)*np.sin(v)*0.98
y=np.sin(u)*np.sin(v)*0.98
z=np.cos(v)*0.98
ax.plot_wireframe(x, y, z, color="k", rstride=1, cstride=1,linewidth=0.5)

plt.hold(True)
ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2], c=ext_mat_array[:,0,0], marker='o')
plt.hold(False)



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
output_data['vertices']=overviewdata['vertices']
output_data['triangles']=overviewdata['triangles']
output_data['shape_name']=overviewdata['shape_name']
#
#
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




















