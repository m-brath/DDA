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
#import glob
import os
#import scipy as sp
import numpy as np
import scipy.io as sio
import sys

#pdb.set_trace()
#from adda_config import *
import adda_control_functions as acf
import icosphere_grid as ico
import adda_orientational_sph_avg_functions as aoa
from netCDF4 import Dataset



##==============================================================================
## Get strings from sys.argv (debug)
##==============================================================================
#
##Temperature index
#Tindex=0
#
##Frequency index
#findex=0
#
##Size index
#Dindex=0
#
##Configuration file
#adda_config='adda_config_box'
##adda_config='adda_config_LiuThickPlateTEST'
#
##from adda_config import *
#print(adda_config)
#
##index = raw_input('input something!: ')
#
#exec('from '+adda_config+' import *')



#==============================================================================
# Get strings from sys.argv
#==============================================================================

#Temperature index
Tindex=int(sys.argv[1])

#Frequency index
findex=int(sys.argv[2])

#Size index
Dindex=int(sys.argv[3])

#Configuration file
adda_config=sys.argv[4]

#from adda_config import *
print(adda_config)

#index = raw_input('input something!: ')

exec('from '+adda_config+' import *')

#==============================================================================
#  constants/paths
#==============================================================================



#adda command
main_command=['mpiexec','-n',str(ncores),'adda_mpi']





#==============================================================================
# symmetries and particle orientation angular grids
#==============================================================================


#only beta, the number of alpha and gamma angles will be set according to
# refinement level 
p_orientation_beta=np.linspace(0.,180.,num=number_of_betas)



#==============================================================================
# storage preperations
#==============================================================================


#Temperature
T=temperatures[Tindex]

#Wavelength in micrometer
lamb=c0/frequencies[findex]*1e6 #[1e-6 m]

#Size
D=Sizes[Dindex]

dmax=Dmax[Dindex]


#generate storage folders
filename=(shape_name+'_deq{DEQ:05.0f}um_freq{FRQ:05.1f}GHz_T{TEMPERATURE:04.0f}K').format
strings={'DEQ':(D), 'FRQ':(c0/(lamb)*1e-3),'TEMPERATURE':T }

storage_path=os.path.join(main_path+subpath,filename(**strings))

if not os.path.exists(storage_path):
        os.makedirs(storage_path)

if not os.path.exists(ramdrive):
        os.makedirs(ramdrive)


#==============================================================================
# start computation
#==============================================================================


#
if np.size(shape_str)==0:
    read_shape_str=['-shape','read',shapelist[Dindex]]

    static_clo=read_shape_str+static_clo



#permittivity of Ice
eps_ice=acf.eps_ice_matzler06(frequencies[findex],T)

#refractive Index
m=np.sqrt(eps_ice)


#symmetry matrix
Ma=np.ones([2,2])
Mb=-np.ones([2,2])
M=np.vstack((np.hstack((Ma,Mb)),np.hstack((Mb,Ma))))

#define start refinement level
ref_lev=0
flag=False
converged=False

epsilon_old=1

#allocate
ext_mat_horiz_avg_old=np.zeros((len(p_orientation_beta),len(incidence_angles),4,4))#*np.inf
eps_mat=np.zeros(limit_sph+1)*np.nan
measure_mat=np.zeros(limit_sph+1)
test_mat=np.zeros(limit_sph+1)
ext_mat_array_old=np.ones((0,4,4))
vertices_old=np.ones((0,3))
triangles_old=np.ones((0,3))


#prepare
zipname_vid=('vertex_X{XV:07.0f}_Y{YV:07.0f}_Z{ZV:07.0f}').format
zip_fid='_test_sph.zip'
             


while (ref_lev<=limit_sph and converged==False) or ref_lev<3 :
    
    
    
    #define/refine grid
    if ref_lev==0:
        [vertices, triangles]=ico.make_icosahedric_grid()
    else:
        [vertices, triangles]=ico.refine_icosahedric_grid(vertices, triangles, 1)
    
            
    vpol=aoa.create_vpol_vector(vertices)
        
        
   



    allocate_flag=False

    for point_i in range(0,np.size(vertices,axis=0)):
        
    
        ##Name of zipfile
        zipstrings={'XV':(vertices[point_i,0]*1e5), 'YV':(vertices[point_i,1]*1e5),
                    'ZV':(vertices[point_i,2]*1e5) }

        zipname=os.path.join(storage_path,shape+'_'+zipname_vid(**zipstrings)+zip_fid)
        
    
        #define input_data dictionary
        input_data={}

        #wavelength
        input_data['lambda']=lamb

        #Particle Size
        input_data['Size']=D/2.# Sizes is defined as diameter but adda needs radius

        #refractive Index
        input_data['refrative_index']=m

        #incidence angle
        input_data['theta']=vertices[point_i,:]

        #alpha
        input_data['alpha']=alpha_0

        #beta angle
        input_data['beta']=beta_0

        #gamma angle
        input_data['gamma']=gamma_0

        #precision
        input_data['eps']=eps


        #ramdrive
        input_data['ramdrive']=ramdrive

        #main command for adda
        input_data['main_command']=main_command

        #Adda's static command line options
        input_data['static_clo']=static_clo

        #Name of the zipname
        input_data['zipname']=zipname

        input_data['zipping']=ZIPPING



        #Use symmetry if existent 
        if ((vertices[point_i,2]<0-1e-10  and  sym_XYobj==True) or 
            (vertices[point_i,1]<0-1e-10  and  sym_XZobj==True) or
            (vertices[point_i,0]<0-1e-10  and  sym_YZobj==True)):
            
#            print('symmetry used')
            
            continue
#        else:
#            print('symmetry not used')
            
        
        if ref_lev==0:
            
            #remove existing zip            
            if os.path.exists(zipname):
                os.remove(zipname)
        
            #Call Adda
            output_data=acf.adda_call(input_data)
            
           
        
            #Allocate extinction matrix array
            if allocate_flag==False:

                q=np.shape(output_data['ext_matrix'])
                qm=np.shape(output_data['mueller_array'])
                
                ext_mat_array=np.empty((np.size(vertices,axis=0),q[0],q[1]))
                ext_mat_array[:] = np.NAN
                ext_mat_array_old=ext_mat_array.copy()
                             
                mueller_array=np.empty((np.size(vertices,axis=0),)+qm) 
                mueller_array[:] = np.NAN  
                mueller_array_old = mueller_array.copy()                         
                
                allocate_flag=True
                
            ext_mat_array[point_i,:,:]=output_data['ext_matrix']    
            mueller_array[point_i,:,:,:]=output_data['mueller_array']
            phis_grid=output_data['phis_grid']
            thetas_grid=output_data['thetas_grid']
    
        else:
            
            #Allocate extinction matrix array
            if allocate_flag==False:

                q=np.shape(output_data['ext_matrix'])
                qm=np.shape(output_data['mueller_array'])
                
                ext_mat_array=np.empty((np.size(vertices,axis=0),q[0],q[1]))
                ext_mat_array[:] = np.NAN
                             
                mueller_array=np.empty( (np.size(vertices,axis=0),)+qm )
                mueller_array[:] = np.NAN   
                
                allocate_flag=True
            
            
            #check if values already exist
            logic_temp=vertices[point_i]==vertices_old
            logic=np.sum(logic_temp,axis=1)==np.size(vertices_old,axis=1)
            
            if sum(logic)==1:
                ext_mat_array[point_i,:,:]=ext_mat_array_old[logic,:,:]
                mueller_array[point_i,:,:,:]=mueller_array_old[logic,:,:,:]

        
            else:
                
                #remove existing zip            
                if os.path.exists(zipname):
                    os.remove(zipname)                
                
                #Call Adda
                output_data=acf.adda_call(input_data)

                

                ext_mat_array[point_i,:,:]=output_data['ext_matrix']
                mueller_array[point_i,:,:,:]=output_data['mueller_array']


    q_ext=ext_mat_array.shape

        

    

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

            inc_dir=np.array([np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)])


            #do the horizontal averaging=======================================

            ext_mat_alpha_temp=np.zeros((q_ext[1],q_ext[2]))

            #counter alpha
            n_alpha=-1

            
            for a_i in range(0,len(p_orientation_alpha)):

                n_alpha=n_alpha+1
                alpha=p_orientation_alpha[a_i]

#                ext_mat_gamma_temp=np.zeros((q_ext[1],q_ext[2],len(p_orientation_gamma)))
                ext_mat_gamma_temp=np.zeros((q_ext[1],q_ext[2]))

                #counter gamma
                n_gamma=-1

                #sample gamma angles
                for g_i in range(0,len(p_orientation_gamma)):

                    n_gamma=n_gamma+1
                    gamma=p_orientation_alpha[g_i]

                    #rotate and interpolate adda sphere
                    vertices_rot=ico.rotate_vertices(vertices,alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)
                    
                    ext_mat_array_temp=ext_mat_array.copy()
 
                    [w_int,idx]=ico.interpolation_weights_icosahedric_grid(vertices_rot,inc_dir)
                    
                    #create lab vertical polarization vectors
                    vpol_lab=aoa.create_vpol_vector(inc_dir)
                    vpol_lab=np.matlib.repmat(vpol_lab,len(idx),1)
                
                    #rotate original vpol
                    vpol_rot=ico.rotate_vertices(vpol[idx,:],alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)   
                    
                    #map to new laboratory vpol
                    [dummy,ext_mat_rot]=aoa.pol_changer(vertices_rot[idx,:], vpol_rot,
                            vpol_lab ,np.empty(0),ext_mat_array_temp[idx,:,:])
            
                    

                    [w_int,idx]=ico.interpolation_weights_icosahedric_grid(vertices_rot,inc_dir)

                    
                    #interpolated value
                    EXTTEMP=np.zeros((q_ext[1],q_ext[2]))
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
                            
                            Ksym=ext_mat_array_temp[idx_sym[0],:,:]*M**N_sym
#                            
                            EXTTEMP=w_int[i]*Ksym+EXTTEMP
                            
                        
                        # end of symmetry exploitation
                        #======================================================
                        else:
                        
#                            EXTTEMP=w_int[i]*ext_mat_array_temp[idx[i],:,:]+EXTTEMP
                            EXTTEMP=w_int[i]*ext_mat_rot[i,:,:]+EXTTEMP

                    #do the gamma summation
                    ext_mat_gamma_temp=EXTTEMP*gauss_weights[g_i]/2+ext_mat_gamma_temp


                    
                #copy
                ext_mat_gamma=ext_mat_gamma_temp


                #do the alpha summation
                ext_mat_alpha_temp=ext_mat_gamma*gauss_weights[a_i]/2+ext_mat_alpha_temp


            #copy
            ext_mat_alpha=ext_mat_alpha_temp

            if ref_lev<1 and flag==False:
                ext_mat_horiz_avg_new=(np.ones((len(p_orientation_beta),
                                         len(incidence_angles),4,4))*np.NaN)
                
                flag=True
                

            ext_mat_horiz_avg_new[n_beta,n_theta,:,:]=ext_mat_alpha

    
    #check convergence
    epsilon_temp=(np.abs(ext_mat_horiz_avg_new[:,:,0:2,0:2]-ext_mat_horiz_avg_old[:,:,0:2,0:2])**2)
    
    
    epsilon_temp=np.mean(np.mean(epsilon_temp,axis=0),axis=0)
    
    old_value_temp=np.mean(np.mean(ext_mat_horiz_avg_old[:,:,0:2,0:2]**2,axis=0),axis=0)
    
    
#    normaliser=ext_mat_horiz_avg_old[:,:,0,0]
    
#    epsilon_temp=((np.mean(np.mean(epsilon_temp,axis=3),axis=2))/
#                    normaliser**2)
    
#    epsilon_temp=((np.max(np.max(epsilon_temp,axis=3),axis=2))/
#                  normaliser**2)
    

#    epsilon=np.sqrt(np.max(epsilon_temp.flatten()))
    epsilon=np.sqrt(np.mean(epsilon_temp.flatten()))
    old_value=np.sqrt(np.mean(old_value_temp.flatten()))

    
    
    if ref_lev>0:        

        converged=epsilon<=delta*old_value
        
        eps_mat[ref_lev]=epsilon/old_value

    else:
        converged==False
        


    #prepare next round
    ref_lev=ref_lev+1
    ext_mat_horiz_avg_old=np.copy(ext_mat_horiz_avg_new)
    epsilon_old=epsilon
    ext_mat_array_old=np.copy(ext_mat_array)
    mueller_array_old=np.copy(mueller_array)
    vertices_old=np.copy(vertices)
    triangles_old=np.copy(triangles)
    

    

    
    print('feddich')

#fig = plt.figure()
#plt.plot(np.arange(0,len(eps_mat)),eps_mat,'x')


#prepare save names
matfile_name=os.path.join(storage_path,shape_name+'_overview_data.mat')
netcdf_name=os.path.join(storage_path,shape_name+'_mueller_data.nc')

#==============================================================================
#       ###Save output in mat format
#==============================================================================

#prepare data for save as overview of run
output_data={}
output_data['theta_inc']=incidence_angles
output_data['beta']=p_orientation_beta
output_data['Temperature']=T
output_data['Frequency']=frequencies[findex]
output_data['wavelength']=lamb;
output_data['diameter_volume_equ']=D*1e-6
output_data['diameter_max']=dmax*1e-6
output_data['refr_index']=m
output_data['sym_XYobj']=sym_XYobj
output_data['sym_XZobj']=sym_XZobj
output_data['sym_YZobj']=sym_YZobj
output_data['ext_mat_array']=ext_mat_array
output_data['ext_mat_horiz_avg']=ext_mat_horiz_avg_new
output_data['eps_vec']=eps_mat
output_data['delta']=delta
output_data['limit_sph']=limit_sph
output_data['numQuadraturePoints']=numQuadraturePoints
output_data['vertices']=vertices
output_data['triangles']=triangles
output_data['mueller_matrix_filename']=shape_name+'_mueller_data.nc'
output_data['shape_name']=shape_name
           
## save overview data          
           
sio.savemat(matfile_name, {'output_data':output_data}, appendmat=True, format='5', long_field_names=True)


#==============================================================================
#       ### save mueller matrix as netcdf
#==============================================================================


#prepare netcdf file
dataset = Dataset(netcdf_name, 'w', format='NETCDF4_CLASSIC')

dataset.createDimension('phi', len(phis_grid))
dataset.createDimension('theta', len(thetas_grid))
dataset.createDimension('midx', qm[2])
dataset.createDimension('vidx',len(vertices))

phi_s=dataset.createVariable('phi_s', np.float32,('phi',))
theta_s=dataset.createVariable('theta_s', np.float32,('theta',))
matrix_index=dataset.createVariable('matrix_index', np.float32,('midx',))
vertex_index=dataset.createVariable('vertex_index', np.float32,('vidx',))

mueller_matrix_array = dataset.createVariable('mueller_matrix_array', np.float32,
                                              ('vidx','theta','phi','midx'),zlib=True)
#assign
mueller_matrix_array[:]=mueller_array
phi_s[:]=phis_grid
theta_s[:]=thetas_grid
matrix_index[:]=np.arange(0,qm[2],1)      
vertex_index[:]=np.arange(0,len(vertices),1) 

#close and write
dataset.close()   
 

#==============================================================================
#       ### info display
#==============================================================================
                   

print('   ')
print('===============================================================')
print('shape = '+shape_name)
print('Temperature = '+str(T)+' K')
print('Frequency = '+str(frequencies[findex]*1e-9)+' GHz')
print('Size = '+str(D)+' µm')
print('===============================================================')
print('   ')























