# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:08:27 2016

@author: Manfred Brath
"""
#Functions to create an icosahedric grid and to refine it
#
#   The vertices/grid points are given by an N x 3 matrix with N the number of
#   vertices/grid point. Each grid point lies on unit sphere. The columns are
#   the cartesian x, y and z coordinate of each grid point/vertex
#   There is an additional M x 3 aray with the vertex index of each triangle of
#   icosahedric grid.




import numpy as np
import icosphere_grid as ico
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

[vertices, triangles]=ico.make_icosahedric_grid()


test=ico.rotate_vertices(np.array([[0,0,1]]),0*np.pi/180,90*np.pi/180,0*np.pi/180)

test2=ico.rotate_vertices(np.array([[0,0,-1]]),0*np.pi/180,90*np.pi/180,0*np.pi/180)

test3=ico.rotate_vertices(np.array([[6e-17,0,-1]]),0*np.pi/180,90*np.pi/180,0*np.pi/180)

#numRefinements=2
#[vertices, triangles]=ico.refine_icosahedric_grid(vertices, triangles, numRefinements)
#
#
#
### do the interpolation
#
#r_v=np.sqrt(np.sum(vertices**2,axis=1));
#theta_v=np.arccos(vertices[:,2]/r_v);
#phi_v=np.arctan2(vertices[:,1],vertices[:,0]);
#
#f=np.sin(4*theta_v)*np.exp(-(theta_v-np.pi/2)**2)*np.exp(-phi_v**2)*np.sinc(phi_v);
#
#
#
#
##incidence angles
#theta_inc=np.linspace(0,180,91)
#phi_inc=np.linspace(-180,180,181)
#
#[PHI_INC,THETA_INC]=np.meshgrid(phi_inc,theta_inc)
#
##true data'
#f_true=(np.sin(4*THETA_INC*np.pi/180)*np.exp(-(THETA_INC*np.pi/180-np.pi/2)**2)*
#        np.exp(-(PHI_INC*np.pi/180)**2)*np.sinc(PHI_INC*np.pi/180));
#
#
##Allocate
#F_int=PHI_INC.copy()*np.nan
#
#for v in range(0,np.size(phi_inc)):
#    for u in range(0,np.size(theta_inc)):
#
##        inc_dir=np.array([np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)])
#
#
##        if v==90:
##            if u==45:
##                print('bla')
#
#        inc_dir=np.array([np.sin(THETA_INC[u,v]*np.pi/180)*np.cos(PHI_INC[u,v]*np.pi/180),
#                          np.sin(THETA_INC[u,v]*np.pi/180)*np.sin(PHI_INC[u,v]*np.pi/180),
#                          np.cos(THETA_INC[u,v]*np.pi/180)]);
#
#        [w,idx]=ico.interpolation_weights_icosahedric_grid(vertices,inc_dir)
#
#        #interpolated value
#        FTEMP=0
#        for i in range(0,len(idx)):
#            FTEMP=w[i]*f[idx[i]]+FTEMP
#
#
#        F_int[u,v]=FTEMP;
##        F_int[u,v]=f_true[u,v]
#
#
#
#
##vertices_sph=ico.vertices_cart2sphere(vertices)
##vertices_rot=ico.rotate_vertices(vertices,45*np.pi/180,0*np.pi/180,0*np.pi/180)
#
##
#fig = plt.figure()
#
#plt.scatter(phi_v*180/np.pi,theta_v*180/np.pi,s=30,c=f)
##
#fig = plt.figure()
#plt.scatter(np.ravel(PHI_INC),np.ravel(THETA_INC),s=30,c=np.ravel(F_int))
#
#fig = plt.figure()
#plt.scatter(np.ravel(PHI_INC),np.ravel(THETA_INC),s=30,c=f_true)
#
##plt.show()
###
###draw sphere
##fig = plt.figure()
##ax = fig.add_subplot(111, projection='2d')
##u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
##x=np.cos(u)*np.sin(v)
##y=np.sin(u)*np.sin(v)
##z=np.cos(v)
##ax.plot_wireframe(x, y, z, color="g")
##
##ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'bx',markersize=12,markeredgewidth=3)
###ax.plot(vertices_rot[:,0],vertices_rot[:,1],vertices_rot[:,2],'r+',markersize=12,markeredgewidth=3)
##ax.set_aspect("equal")
###
###numRefinements=1
##[vertices, triangles]=ico.refine_icosahedric_grid(vertices_rot, triangles, numRefinements)
##
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
##draw sphere
#u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
#x=np.cos(u)*np.sin(v)
#y=np.sin(u)*np.sin(v)
#z=np.cos(v)
##ax.plot_wireframe(x, y, z, color="r")
#ax.set_aspect("equal")
#
##ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'bx',markersize=6,markeredgewidth=2)
#ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2], c=vertices[:,0], marker='o')