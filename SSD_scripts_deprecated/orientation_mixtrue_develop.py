#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:14:42 2018

@author: u242031
"""

import typhon.arts.xml as xml
import glob
import numpy as np
import sys

import os
from copy import deepcopy

# =============================================================================
# constants
# =============================================================================

#habitname='HexPlaAggCrystal_Id20'
#habitname='PlateType1_Id9'
habitname = sys.argv[1]

scatterer_path='/scratch/uni/u237/user_data/mbrath/adda_results/SSD_XML_MultiFreq/'+habitname

base_outpath='/scratch/uni/u237/user_data/mbrath/adda_results/SSD_ori_mixtures_MultiFreq/'+habitname+'/'


#number of mixtures
nom=7

#define different sigma for the different orientation mixtures
a=np.array([0])
#b=np.logspace(np.log10(5),np.log10(90),nom-2)
b=5*2**np.linspace(0,4,nom-2)
c=np.array([np.inf])
sigma=np.concatenate((a,b,c));


# =============================================================================
# aux functions
# =============================================================================

def find_nearest(array, value):
    '''
    Function that find the neareast value within a given array
    :param array: the array where to look after
    :param value: the search value
    :return  the nearest value within the array and the corresponding index
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def gauss(x,x0,sigma):

    rho=1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((x-x0)/(sigma))**2*1/2);

    return rho

def bi_gauss(x,m1,m2,sigma1,sigma2):
    
    rho=gauss(x,m1,sigma1)/2+gauss(x,m2,sigma2)/2
    
    return rho


# =============================================================================
# get the orienations
# =============================================================================

orientationlist = glob.glob(scatterer_path + '/*')


#number of existing orientations
noe=len(orientationlist)


orientations = np.ones(noe)
for i in range(noe):
    beta = os.path.basename(orientationlist[i])
    strings = beta.split('_')

    orientations[i] = float(strings[2][4:7])

#sort it!
o_idx=np.argsort(orientations);
orientationlist=[orientationlist[idx] for idx in o_idx]

#tilt angles
betas=orientations[o_idx]
beta_max=np.max(betas)

# =============================================================================
# do the averaging
# =============================================================================



#define uniform distribution
rho_0=1/noe*np.ones((noe));


#allocate
rho=np.zeros((nom,noe));

# loop over the different mixtures
for i in range(nom):


    if i==0: #exact horizontal
        
        kernel=np.zeros(noe);
        [dummy,nidx]=find_nearest(betas,0);
        kernel[nidx]=1;
        
        if beta_max==180:            
            [dummy,nidx]=find_nearest(betas,180);
            kernel[nidx]=1;
       
    elif i==nom-1: #total random 
        kernel=rho_0*np.sin(betas*np.pi/180);
        kernel=kernel/np.sum(kernel);

    else: 
        
        if beta_max==90:        
            # for a particle with beta max = 90 deg it is enough to integrate from 
            # 0 to 90 deg.
            kernel=gauss(betas,0,sigma[i])*np.sin(betas*np.pi/180);
        elif beta_max==180:
            # in our case of particle, where the particle fall speed is independent of if
            # top side or the bottom side is pointing down, then we have to integrate from
            # 0 to 180 deg and the probability distribution is bimodal with modes of 0 and 180. 
            kernel=bi_gauss(betas,0,180,sigma[i],sigma[i])*np.sin(betas*np.pi/180);                
        else:
            raise Exception('beta max must be exactly 90 deg or 180 deg')    
    
    kernel=kernel/np.sum(kernel);


    rho[i,:]=kernel;

    #load meta data
    scattererlist_meta = glob.glob(orientationlist[0] + '/*' + habitname + '*meta.xml')
    scat_meta_temp=xml.load(scattererlist_meta[0])
    smt=scat_meta_temp[0]

    #modify description in meta data
    for k in range(len(smt)):
        smt[k].description='orientation mixture -- sigma_beta = '+str(sigma[i])+' deg'

    #do the orientation mixture
    for j in range(noe):

        #load the data
        scattererlist_data = glob.glob(orientationlist[j] + '/*' + habitname + '*data.xml')
        print('...reading data:')
        print(scattererlist_data[0])
        print('   ')
        scat_data_temp=xml.load(scattererlist_data[0])

        #do the mixture
        sdt_temp = scat_data_temp[0]

        if j==0:
            sdt=deepcopy(sdt_temp)

            for k in range(len(sdt)):
                sdt[k].pha_mat_data=sdt[k].pha_mat_data*0.
                sdt[k].ext_mat_data=sdt[k].ext_mat_data*0.
                sdt[k].abs_vec_data=sdt[k].abs_vec_data*0.


        for k in range(len(sdt)):

            sdt[k].pha_mat_data=sdt_temp[k].pha_mat_data*rho[i,j]+sdt[k].pha_mat_data
            sdt[k].ext_mat_data=sdt_temp[k].ext_mat_data*rho[i,j]+sdt[k].ext_mat_data
            sdt[k].abs_vec_data=sdt_temp[k].abs_vec_data*rho[i,j]+sdt[k].abs_vec_data


    scat_data_temp = [sdt]
    scat_meta_temp = [smt]

    if np.isinf(sigma[i]):
        outpath=base_outpath+'sigma_%02.1f_deg' %(9999)
    else:
        outpath=base_outpath+'sigma_%02.1f_deg' %(sigma[i])

     #Check if path exists
    if  not os.path.exists(outpath):
        os.makedirs(outpath)

    print('saving the data...')
    filename = '%s/%s' %(outpath,habitname)
    print(filename)
    xml.save(scat_data_temp,'%s.scat_data.%s' %(filename,'xml'), format='binary')
    xml.save(scat_meta_temp,'%s.scat_meta.%s' %(filename,'xml'), format='binary')

    print('i = '+str(i))

print('...and done')