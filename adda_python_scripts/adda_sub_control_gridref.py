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
import sys
#import pdb
#import shutil

#pdb.set_trace()
#from adda_config import *
import adda_control_functions as acf


import zipfile
try:
#    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

modes = { zipfile.ZIP_DEFLATED: 'deflated',
          zipfile.ZIP_STORED:   'stored'}

##==============================================================================
## Get strings from sys.argv (debug)
##==============================================================================

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
#adda_config='adda_config_ellipsoid_test'
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
# storage preperations
#==============================================================================


##storage
storage_path=main_path+subpath

if not os.path.exists(storage_path):
    os.mkdir(storage_path)


#check ramdrive
if not os.path.exists(ramdrive):
    os.makedirs(ramdrive)

#==============================================================================
# symmetries and particle orientation angular grids
#==============================================================================

## ==>> all angles in degree <<==

#Particle orientation alpha (lab z-axis)
if sym_Zlab==True:

    p_orientation_alpha=np.array([0.,180.])    #np.array([0.])#

else:

    p_orientation_alpha=np.array([0.,360.])    #np.array([0.])#



#Particle orientation beta (particle y-axis)
if sym_XYobj==True:

    nob=int(np.ceil(number_of_betas/2.))
    p_orientation_beta=np.linspace(0.,90.,num=nob)      #np.array([0.])#


else:

    p_orientation_beta=np.linspace(0.,180.,num=number_of_betas)      #np.array([0.])#




#Particle orientation gamma (particle z-axis)
if np.isinf(sym_Zobj):

    p_orientation_gamma=np.array([0.])

else:

    p_orientation_gamma=np.array([0.,360./sym_Zobj])



#Incidence angles,
angle_range=int(max(incidence_angles)-min(incidence_angles))

if sym_XZobj==True and sym_YZobj==True and angle_range==180:

    nob=int(np.ceil(len(incidence_angles)/2.))
    incidence_angles=np.linspace(0.,90.,num=nob)      #np.array([0.])#






#==============================================================================
# start computation
#==============================================================================


#Temperature
T=temperatures[Tindex]

#Wavelength in micrometer
lamb=c0/frequencies[findex]*1e6 #[1e-6 m]

#Size
D=Sizes[Dindex]

dmax=Dmax[Dindex]

#
if np.size(shape_str)==0:
    read_shape_str=['-shape','read',shapelist[Dindex]]

    static_clo=read_shape_str+static_clo



#permittivity of Ice
eps_ice=acf.eps_ice_matzler06(frequencies[findex],T)

#refractive Index
m=np.sqrt(eps_ice)


#Name of zipfile
zipname=storage_path+shape+'-'+str(Tindex)+'-'+str(findex)+'-'+str(Dindex)+'_test.zip'

#Clean up, if zip-file is already existing
if os.path.exists(zipname):
    os.remove(zipname)


theta_index=-1
for theta in incidence_angles:

    for beta in p_orientation_beta:

        #define input_data dictionary
        input_data={}

        #wavelength
        input_data['lambda']=lamb

        #Particle Size
        input_data['Size']=D/2.# Sizes is defined as diameter but adda needs radius

        #refractive Index
        input_data['refrative_ndex']=m

        #incidence angle
        input_data['theta']=theta

        #alpha
        input_data['alpha_ab']=np.array([p_orientation_alpha[0],p_orientation_alpha[-1]])

        #beta angle
        input_data['beta']=beta

        #gamma angle
        if len(p_orientation_gamma)>1:
            input_data['gamma_ab']=np.array([p_orientation_gamma[0],p_orientation_gamma[-1]])
        else:
            input_data['gamma_ab']=np.array([p_orientation_gamma[0]])

        #precision
        input_data['eps']=eps


        #ramdrive
        input_data['ramdrive']=ramdrive

        #main command for adda
        input_data['main_command']=main_command

        #Adda's static command line options
        input_data['static_clo']=static_clo

        #simulation basename
#        input_data['simbasename']=simbasename

        #Name of the zipname
        input_data['zipname']=zipname

        #Limits
        input_data['limit_alpha']=limit_alpha
        input_data['limit_gamma']=limit_gamma
        input_data['delta']=delta

#        pdb.set_trace()

        #Call Adda inclusive integration over alpha and gamma angles
        output_data=acf.romberg_alpha(input_data)

        #a little bit of cleaning up
#        path_name=ramdrive+'/'+simbasename
#        shutil.rmtree(path_name)

        #put incidence angla and beta angle into output data
        output_data['theta_inc']=theta
        output_data['beta']=beta
        output_data['Temperature']=T
        output_data['Frequency']=frequencies[findex]
        output_data['diameter_volume_equ']=D*1e-6
        output_data['diameter_max']=dmax*1e-6
        output_data['refr_index']=m
        output_data['sym_Zobj']=sym_Zobj
        output_data['sym_XYobj']=sym_XYobj
        output_data['sym_Zlab']=sym_Zlab

        #Save output in mat format
        mat_temp_name=ramdrive+'ADDA_averaged_alpha_gamma_'+'theta_'+str(theta)+'beta_'+str(beta)+'.mat'

#        pdb.set_trace()

        print('   ')
        print('===============================================================')
        print('theta = '+str(theta))
        print('beta = '+str(beta))
        print('Temperature = '+str(T)+' K')
        print('Frequency = '+str(frequencies[findex]*1e-9)+' GHz')
        print('Size = '+str(D)+' µm')
        print('===============================================================')
        print('   ')

        import scipy.io as sio
        sio.savemat(mat_temp_name, {'output_data':output_data}, appendmat=True, format='5', long_field_names=True)





        with zipfile.ZipFile(zipname, mode='a',allowZip64=True) as zf:


            #look after the adda output
            filelist = glob.glob(mat_temp_name)

#            pdb.set_trace()
            #and now put it into a zipfile
            for filename in filelist:


                folder_structure=os.path.join('theta_'+str(theta),
                'beta_'+str(beta),os.path.basename(filename))

                zf.write(filename, folder_structure, compress_type=compression)

                #clean up
                os.remove(filename)




















