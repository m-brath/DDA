#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26, 2021

@author: Manfred Brath
"""
import os
import utils
import sph
import numpy as np
import glob

from matplotlib import pyplot as plt

#==============================================================================
#  additional functions
#==============================================================================

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

#==============================================================================
#%%  path/constants 
#==============================================================================

#Particle shape
shape='PlateType1_Id9'

# orientation angle
beta=0

#Desired particle size in 10^-6 m
particle_size_desired=735


main_folder=('/scratch/uni/u237/user_data/mbrath/adda_results/SSD/'+
             f'AzimuthallyRandom/ice/Pristine/{shape}/AzimuthallyRandom_beta{beta:03.0f}deg')

# =============================================================================
#%% get data
# =============================================================================

print('loading data...\n')

# get list of particles
particle_list=glob.glob(os.path.join(main_folder,'*.nc'))
particle_list=[os.path.basename(particle) for particle in particle_list]

# get volume equiv. diameter in 10^-6 m      
dveq=[ float(particle.split('_')[1][4:9]) for particle in particle_list]

# get particle 
particle_size, particle_index=find_nearest(dveq, particle_size_desired)

#load dataset
ssd_dataset=utils.ssdb_read_ncfile(os.path.join(main_folder,particle_list[particle_index]))


print('loading data done!\n')


# =============================================================================
#%% select frequency and temperature
# =============================================================================

#Desired frequency in Hz
freq_desired=671e9

#Desired temperature in K
T_desired=230

# =============================================================================
#%% Extract assymetry parameter
# =============================================================================

freq_grid = ssd_dataset[1]
T_grid = ssd_dataset[2]


T, T_index=find_nearest(T_grid, T_desired)
freq, f_index=find_nearest(freq_grid, freq_desired)



#select data
ssd=ssd_dataset[0][f_index,T_index]

#Incidence angles
theta_inc=ssd['SingleScatteringData']['za_inc']['data']

#Phase function in spherical harmonics
phase_real=ssd['SingleScatteringData']['phaMat_data_real']['data'][0,0,:,:]
phase_imag=ssd['SingleScatteringData']['phaMat_data_imag']['data'][0,0,:,:]
phase_sph=np.complex128(phase_real+1j*phase_imag)

#dummy grid size
grid_size_theta=90

#allocate
g=np.zeros(len(theta_inc))

for i in range(len(theta_inc)):
    
    print(f'theta ={theta_inc[i]:4.1f} deg')
    
    #get truncation level of spherical harmonics series
    max_ntrunc = sph.get_lmax(phase_sph[i,:])
    
    
    # some preparation stuff for shtns. I know, that I need this, but right now
    # I don't know why...I have to think about it. Nonetheless the results below
    # are correct.
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
    
    # set up spherical harmonics object
    sph_object = sph.Spharmt(min_grid_size*2,min_grid_size,max_ntrunc,1,gridtype='regular')
    
    # rotate spherical harmonics series, so that forward direction points toward northpole
    # and backward direction towards southpole.
    phase_sph_rot = sph_object.yrotate(phase_sph[i,:], -theta_inc[i]*np.pi/180)
    
    #assymetry parameter
    g[i]=np.real(phase_sph_rot[1]/phase_sph_rot[0]/np.sqrt(4*np.pi)*2)
    

g=np.array(g)
    
# =============================================================================
#%% plot assymetry parameter     
# =============================================================================

width_in_cm=29.7
height_in_cm=20.9

fig, ax = plt.subplots(1,1,dpi=150)
fig.set_size_inches(width_in_cm / 2.54, h=height_in_cm / 2.54)

ax.plot(theta_inc,g)
ax.set_xlabel(r'$\theta_{inc} \, [^\circ]$')
ax.set_ylabel(r'$g$')
ax.set_title(f'{shape} - {particle_size}$\mu$m - {T:3.0f}K - {freq/1e9:3.1f}GHz - {beta:.0f} deg')
ax.set_ylim([0,1])

