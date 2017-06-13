# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.special import sph_harm
from scipy.special import lpmv
import scipy as sp
import matplotlib.pyplot as plt
import spherical_harmonics_for_adda as spha
import typhon as ty
import glob
import os

import adda_control_functions as acf
from netCDF4 import Dataset
import shtns

##==============================================================================
##  create test data
##==============================================================================
#
##incidence angles
#theta=np.linspace(0,180,91)*np.pi/180
#phi=np.linspace(-180,180,181)*np.pi/180
#
#[PHI,THETA]=np.meshgrid(phi,theta)
#
##true data'
#f_true=5*(np.sin(4*THETA)*np.exp(-10*(THETA-np.pi/2)**2)*
#        np.exp(-(PHI)**2)*np.sinc(PHI));
##f_true=np.sin(1*THETA)*np.cos(1*PHI)**2+np.abs(PHI);
##f_true=20*np.exp(-10*( (THETA-np.pi/2)**2+PHI**2))+10*np.sin(1*THETA)*np.cos(1*PHI)**2
##f_true=THETA*0+1.
#
##f_sq=(spha.twoD_int(f_true**2*np.sin(THETA),THETA[:,0],PHI[0,:]))
##f_lin=(spha.twoD_int(f_true*np.sin(THETA),THETA[:,0],PHI[0,:]))
#
#fig, ax = plt.subplots()
#pc=ax.pcolormesh(PHI,THETA,f_true)
#ax.set_xlabel('Phi')
#ax.set_ylabel('Theta')
#ax.set_aspect('equal')
#
#cbar = fig.colorbar(pc)
#
#
#[sph_coeffs, sph_obj, f_int, delta]=spha.sph_expansion(f_true,theta,phi,eps=0.01,diagnostics=True)
#
#
#lons,lats = np.meshgrid(sph_obj.lons, sph_obj.lats)
#
##transform spherical harmonics to field
#
##sph_coeffs[1:-1]=0+0j
#f_test=sph_obj.spectogrd(sph_coeffs)
#
#
#
#fig, axarr = plt.subplots(2,1)
#pc1=axarr[0].pcolormesh(lons,lats,np.real(f_test))
#axarr[0].set_xlabel('Phi')
#axarr[0].set_ylabel('Theta')
#axarr[0].set_aspect('equal')
#
#cbar = fig.colorbar(pc1,ax=axarr[0])
#
#pc2=axarr[1].pcolormesh(PHI,THETA,np.sqrt(np.abs(np.real(f_test)-f_int)**2))
#axarr[1].set_xlabel('Phi')
#axarr[1].set_ylabel('Theta')
#axarr[1].set_aspect('equal')
#
#cbar = fig.colorbar(pc2,ax=axarr[1])








rawdatapath='/scratch/uni/u237/users/mbrath/adda_results/ellipsoid_spehere_ansatz/rawdata_eps3/'
#rawdatapath='/scratch/uni/u237/users/mbrath/adda_results/ExampleAggregates/'
#rawdatapath='/scratch/uni/u237/users/mbrath/adda_results/HongPlate_Id9/'

#look after the adda output
folderlist = glob.glob(rawdatapath+'*deq*K')

counter=-1
for folder in folderlist:
    
    counter=counter+1
    folder=os.path.basename(folder)
    print(folder+' ('+str(counter)+')')
    
print('======================================================================')    
print('choose data')    
print('type the number within in brackets')
print('or type "all" for all data.')





folderlist=[folderlist[7]]

folder=os.path.basename(folderlist[0])

inputdatapath=rawdatapath+folder+'/'

print(inputdatapath)

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

mueller_array=data.variables['mueller_matrix_array'][0,:,:,:]

phi_s=data.variables['phi_s'][:]
theta_s=data.variables['theta_s'][:]






#==============================================================================
# start expansion
#==============================================================================


phi_s_rad=phi_s*np.pi/180
theta_s_rad=theta_s*np.pi/180

[PHI,THETA]=np.meshgrid(phi_s_rad,theta_s_rad)


#f_old=mueller_array[:,:,0]

#f_true=np.roll(f_old,int(len(theta_s_rad)/2),axis=0)
f_true=mueller_array[:,:,1]

n=np.arange(0,np.size(THETA,axis=0))
win=(1-np.cos(2*np.pi*n/(np.size(THETA,axis=0)-1)))/2
win=np.matlib.repmat(win,np.size(THETA,axis=1),1)
win=win.transpose()



M11=mueller_array[:,:,0]
m11_int=np.sqrt(spha.twoD_int(M11**2*np.sin(THETA),THETA[:,0],PHI[0,:]))

#eps_m=np.percentile(np.abs(mueller_array[:]),100)
#eps_m=np.percentile(np.abs(f_true[:]),100)
eps_m=m11_int*0.001



fig, ax = plt.subplots()
pc=ax.pcolormesh(PHI,THETA,f_true)
ax.set_xlabel('Phi')
ax.set_ylabel('Theta')
ax.set_aspect('equal')

cbar = fig.colorbar(pc)


[sph_coeffs, sph_obj, f_int, delta]=spha.sph_expansion(f_true,theta_s_rad,phi_s_rad,eps=eps_m,diagnostics=True)


lons,lats = np.meshgrid(sph_obj.lons, sph_obj.lats)

#transform spherical harmonics to field
f_test=sph_obj.spectogrd(sph_coeffs)



fig, axarr = plt.subplots(2,1)
pc1=axarr[0].pcolormesh(lons,lats,f_test)
axarr[0].set_xlabel('Phi')
axarr[0].set_ylabel('Theta')
axarr[0].set_aspect('equal')

cbar = fig.colorbar(pc1,ax=axarr[0])

pc2=axarr[1].pcolormesh(PHI,THETA,np.abs(f_test-f_int))
axarr[1].set_xlabel('Phi')
axarr[1].set_ylabel('Theta')
axarr[1].set_aspect('equal')

cbar = fig.colorbar(pc2,ax=axarr[1])




#==============================================================================
# reconstruct with a different grid
#==============================================================================

if sph_obj.ntrunc % 2 ==0:
    min_grid_size=sph_obj.ntrunc+2
else:
    min_grid_size=sph_obj.ntrunc+1
    
if min_grid_size<32:
    min_grid_size=32    

x = spha.Spharmt(min_grid_size*2,min_grid_size,sph_obj.ntrunc,1,gridtype='gaussian')

f_horst=x.spectogrd(sph_coeffs)


#create spherical harmonics object
sph_object=shtns.sht(sph_obj.ntrunc, sph_obj.ntrunc, 1, shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)

#sph_coeffs2=sph_coeffs.copy()
#sph_coeffs2[10:-1]=0
#sph_rot=sph_object.Zrotate(sph_coeffs2,-0.74)
#sph_rot[10:-1]=0


sph_rot=x.zrotate(sph_coeffs,np.pi/2)
f_rot=x.spectogrd(sph_rot)


##cutted
#sph_coeffs_cut=sph_coeffs.copy()
#sph_coeffs_cut[26:-1]=0+0j
#
#f_cut=x.spectogrd(sph_coeffs_cut)
#
xlons,xlats = np.meshgrid(x.lons, x.lats)
#
##xlons=np.hstack((xlons,xlons[:,[0]]*0+np.pi*2))
##xlats=np.hstack((xlats,xlats[:,[0]]))
##f_horst=np.hstack((f_horst,f_horst[:,[0]]))
#
#
#
fig, axarr = plt.subplots(2,1)
pc1=axarr[0].pcolormesh(xlons,xlats,f_horst)
axarr[0].set_xlabel('Phi')
axarr[0].set_ylabel('Theta')
axarr[0].set_aspect('equal')

cbar = fig.colorbar(pc1,ax=axarr[0])

pc2=axarr[1].pcolormesh(xlons,xlats,f_rot)
axarr[1].set_xlabel('Phi')
axarr[1].set_ylabel('Theta')
axarr[1].set_aspect('equal')

cbar = fig.colorbar(pc2,ax=axarr[1])


#==============================================================================
# 
#==============================================================================

from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D



# The Cartesian coordinates of the unit sphere
x = np.sin(xlats) * np.cos(xlons)
y = np.sin(xlats) * np.sin(xlons)
z = np.cos(xlats)


#f_horst=np.log10(f_horst)

norm = colors.Normalize()
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(14,10))
m = cm.ScalarMappable(cmap=cm.inferno)
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.inferno(norm(f_horst)))
m.set_array(f_horst)
fig.colorbar(m, shrink=0.8);



##==============================================================================
## 
##==============================================================================
#
##maximum length of spherical harmonics coefficients per particle scan
#max_len_coeffs=overviewdata['max_len_coeffs']
#
#max_ntrunc=overviewdata['max_ntrunc']
#
#
##load mueller matrix
#mueller_array=ty.arts.xml.load(inputdatapath+overviewdata['mueller_matrix_filename'])
#
#
#
#if max_ntrunc % 2 ==0:
#    min_grid_size=max_ntrunc+2
#else:
#    min_grid_size=max_ntrunc+1
#    
#if min_grid_size<90:
#    min_grid_size=90    
#
#x = spha.Spharmt(min_grid_size*2+1,min_grid_size,max_ntrunc,1,gridtype='gaussian')
#
#sph_object=shtns.sht(max_ntrunc, max_ntrunc, 1, shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)
#
#sph_coeffs=mueller_array[0]
#
#
#
#
#
#sph_coeffs_null=np.zeros(max_len_coeffs,dtype=complex)
#
#xlons,xlats = np.meshgrid(x.lons, x.lats)
#
#
#fig = plt.figure()
#
#for j in range(0,np.size(sph_coeffs,axis=0)):
#            
##    temp=sph_coeffs_null.copy()
##    temp[0:len(sph_coeffs[j])]=np.complex128(sph_coeffs[j])
#
#    temp=spha.map_sphexp(sph_coeffs[j],sph_object)
#    
#    
#    mjj=x.spectogrd(temp)
#    
#    plt.subplot(4, 4, j+1)
#    h1=plt.pcolormesh(xlons,xlats,mjj)
#    
#    if j==0:
#        plt.ylabel('\theta [deg]')
#        
#    if j==15: 
#        plt.xlabel('\phi [deg]')
#    
#    fig.colorbar(h1)
#
#
