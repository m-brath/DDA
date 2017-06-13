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

import adda_orientational_sph_avg_functions as aoa
import typhon as ty


#==============================================================================
# symmetries, particle orientation angular grids and other stuff
#==============================================================================


#only beta, the number of alpha and gamma angles will be set according to
# refinement level 
p_orientation_beta=np.linspace(0,180.,num=1)

incidence_angles=np.linspace(0,180.,num=19)



alpha_0=0
beta_0=90
gamma_0=0

#
rawdatapath='/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/rawdata_eps3/'
avgdatapath='/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/orientational_avg_eps3/'

#rawdatapath='/scratch/uni/u237/users/mbrath/adda_results/ExampleAggregates/'
#avgdatapath='/scratch/uni/u237/users/mbrath/adda_results/ExampleAggregates/orientational_avg_test/'

#rawdatapath='/scratch/uni/u237/users/mbrath/adda_results/LiuShape887/'
#avgdatapath='/scratch/uni/u237/users/mbrath/adda_results/LiuShape887/orientational_avg_test_37betas/'

#==============================================================================
# storage preperations
#==============================================================================





#look after the adda output
folderlist = glob.glob(rawdatapath+'*_deq*')

counter=-1
for folder in folderlist:
    
    counter=counter+1
    folder=os.path.basename(folder)
    print(folder+' ('+str(counter)+')')
    
print('======================================================================')    
print('choose data')    
print('type the number within in brackets')
print('or type "all" for all data.')
#index = input(' ==>>  ')    
#index='3'



#if index!='all':
#    folderlist=[folderlist[int(index)]]


folderlist=[folderlist[5]]


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

    
    output_data=aoa.rawdata2average(inputdatapath, incidence_angles, p_orientation_beta,
                                outputdatapath,
                                alpha_0,beta_0t,gamma_0,parallel=True)

    t.stop()
    
#==============================================================================
#       ###For development
#==============================================================================    

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.tri as mtri
    
    
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

    
    
    
    
    
#    x=vertices[:,0]
#    y=vertices[:,1]
#    z=vertices[:,2]
#    
#    r=np.sqrt(x**2+y**2+z**2)
#    theta=np.arccos(z/r)
#    phi=np.arctan2(y,x)
#    triangles = mtri.Triangulation(theta, phi).triangles
#    colors = ext_mat_array[:,0,0]
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    cmap = plt.get_cmap('Blues')
#    triang = mtri.Triangulation(x, y, triangles)
#    collec = ax.plot_trisurf(triang, z, cmap=cmap, shade=True, linewidth=0.)
#    collec.set_array(colors)
#    cbar = fig.colorbar(collec)
#    collec.autoscale()
#    plt.show()
    
    
    
    
    
    
    
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
    
    
    
    phi=output_data['phi_s']
    theta=output_data['theta_s']

    [PHI,THETA]=np.meshgrid(phi,theta)
    
    mueller_mat=mue_mat_horiz_avg[0,0,:,:,:]
    
    fig = plt.figure()
    
    for j in range(1,17):
        
        
        
        plt.subplot(4, 4, j)
        h1=plt.pcolormesh(PHI,THETA,mueller_mat[:,:,j-1])
        
        if j==1:
            plt.ylabel('\theta [deg]')
            
        if j==16: 
            plt.xlabel('\theta [deg]')
        
        cbar = fig.colorbar(h1)
    
    


