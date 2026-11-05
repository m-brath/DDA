#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:58:25 2015

@author: u242031
"""
#==============================================================================
# import modules
#==============================================================================

#print("Hello World")

import subprocess as sub
import glob
import functools
import os
#import scipy as sp
import numpy as np
import scipy.io as sio
import sys
import itertools
import pickle
import typhon as ty
#import zipfile

#pdb.set_trace()
#from adda_config import *
import adda_control_functions as acf
import icosphere_grid as ico
#import adda_orientational_sph_avg_functions as aoa
import adda_orientational_avg_spharmonics_functions as aoa
import spherical_harmonics_for_adda as spha
#from netCDF4 import Dataset
from memory import memstring

import copy

from posix_spawn import spawn_command


# Force flush, otherwise output can disappear from slurm log file
print = functools.partial(print, flush=True)

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
##adda_config='adda_config_HexPlaAggCrystal_Id20'
#adda_config='adda_config_LiuSectorSnowflake_Id3'
#
##from adda_config import *
#print(adda_config)
#
##index = raw_input('input something!: ')
#
#exec('from '+adda_config+' import *')


print("Starting job")
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
#raise RuntimeError("Alles doof")

#index = raw_input('input something!: ')

exec('from '+adda_config+' import *')

#==============================================================================
#  constants/paths
#==============================================================================


if 'calc_type' not in locals():

    calc_type='thunder'


#==============================================================================
# ADDA executable + launcher — config-driven (Phase 3)
#   adda_binary : path (or command name) of the ADDA executable
#   adda_launcher: list of launcher tokens prefixed to it ('' = run directly)
#   adda_version: provenance string, e.g. '1.2' or '1.5.0-alpha3'
# Defaults preserve the historical per-cluster behaviour (ADDA 1.2).
#==============================================================================


if 'adda_binary' not in locals():

    if calc_type=='mistral':

        adda_binary=prefix+'/adda/adda_1.2/src/mpi/adda_mpi'

    else:

        adda_binary='adda_mpi'


if 'adda_launcher' not in locals():

    if calc_type=='mistral':

        adda_launcher=['srun','-n',str(ncores)]

    elif calc_type=='local':

        adda_launcher=[]

    else:

        adda_launcher=['mpiexec','-n',str(ncores)]


if 'adda_version' not in locals():

    adda_version='1.2'


#adda command
main_command=list(adda_launcher)+[adda_binary]





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

# ramdrive may be None in local mode (system temp dir is used then)
if ramdrive is not None and not os.path.exists(ramdrive):
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

# initial placeholders for the refinement-loop outputs; they are always
# (re-)allocated inside the loop, but binding them here prevents NameError
# if the loop body ever exits before the allocation branch ran
ext_mat_array=np.empty((0,0,0))
mueller_array=[]
rel_err_array=np.zeros((0,16))
max_len_array=np.zeros(0)
max_ntrunc_array=np.zeros(0)
ext_mat_horiz_avg_new=np.full((len(p_orientation_beta),len(incidence_angles),4,4),np.nan)


#prepare
zipname_vid=('vertex_X{XV:07.0f}_Y{YV:07.0f}_Z{ZV:07.0f}').format
zip_fid='_temp.zip'

#netcdf_tempname=os.path.join(storage_path,shape_name+'_temp_data.nc')
netcdf_tempname=os.path.join(storage_path,shape_name+'_temp_data_')
print(f"NetCDF path: {netcdf_tempname}")

filenum=0

# Phase 4: mirror-symmetry write-time fill-in.
# A particle mirror maps the just-computed raw ADDA grid of a run-vertex to
# all its mirror-image vertices on the SAME propagation-attached angle grid
# (transform empirically verified by analysis/phase4_grid_symmetry.py, and
# the icosahedron grid is exactly closed under coordinate mirrors). Filled
# vertices get identical NetCDF cache entries, so all later refinement
# passes, averaging and post-processing see a complete dataset and need no
# symmetry branch at all. Mirrors only need to be RUN for one symmetry-
# reduced octant of the grid (see the _runfirst ordering below).
_mirror_flags=[f for f,s in (('XY',sym_XYobj),('XZ',sym_XZobj),('YZ',sym_YZobj))
               if s]


def mirror_fill(point_i, particle_id, output_data, vertices, ext_mat_array,
                mueller_array, rel_err_array, max_len_array, max_ntrunc_array):

    if not _mirror_flags or 'mueller_array' not in output_data:
        return

    for nsub in range(1, len(_mirror_flags)+1):

        for subset in itertools.combinations(_mirror_flags, nsub):

            img=vertices[point_i,:].copy()
            for f in subset:
                img[acf.MIRROR_AXES[f]]=-img[acf.MIRROR_AXES[f]]

            r_near=ico.find_nearest_Nvertices(vertices,img,N=1)
            jj=int(np.asarray(r_near[0])[0])
            snap=float(np.asarray(r_near[1])[0])

            if jj==point_i or mueller_array[jj] is not None:
                continue

            if snap>1e-3:
                print('WARNING: mirror snap %.1e rad (subset %s)'
                      %(snap,'+'.join(subset)))

            jmap=acf.mirror_phi_index_map(output_data['phis_grid'],subset)
            mir_data={}
            mir_data['mueller_array']=output_data['mueller_array'][:,jmap,:]\
                                      *acf.mirror_signs(len(subset))
            mir_data['thetas_grid']=output_data['thetas_grid']
            mir_data['phis_grid']=output_data['phis_grid']

            extm=output_data['ext_matrix']\
                 *acf.mirror_signs(len(subset)).reshape((4,4))

            [sphj,relj,lenj,ntrj]=spha.expand_mueller_2_spharmonics(
                    mir_data,eps_sphexp,diagnostics=False)

            pid_j=zipname_vid(**{'XV':(vertices[jj,0]*1e5),
                                 'YV':(vertices[jj,1]*1e5),
                                 'ZV':(vertices[jj,2]*1e5)})

            acf.write_data_to_separate_netcdf(netcdf_tempname, pid_j,
                                              sphj, relj, lenj, ntrj, extm)

            ext_mat_array[jj,:,:]=extm
            mueller_array[jj]=sphj
            rel_err_array[jj,:]=relj
            max_len_array[jj]=lenj
            max_ntrunc_array[jj]=ntrj

            print('  mirror-filled %s (%s) from %s'
                  %(pid_j,'+'.join(subset),particle_id))


while (ref_lev<=limit_sph and converged==False) or ref_lev<=2 :



    #define/refine grid
    if ref_lev==0:
        [vertices, triangles]=ico.make_icosahedric_grid()
    else:
        [vertices, triangles]=ico.refine_icosahedric_grid(vertices, triangles, 1)


    vpol=aoa.create_vpol_vector(vertices)


    allocate_flag=False

    # Phase 4: with mirror symmetries enabled, process the run-vertices (all
    # flag-coordinates non-negative) FIRST, so every skipped vertex has already
    # been mirror-filled by the time the loop reaches it
    if _mirror_flags:

        def _runfirst(v):
            for f in _mirror_flags:
                if v[acf.MIRROR_AXES[f]]<0:
                    return 1
            return 0

        _order=sorted(range(0,np.size(vertices,axis=0)),
                      key=lambda i: _runfirst(vertices[i,:]))
    else:
        _order=range(0,np.size(vertices,axis=0))

    for point_i in _order:

        ##Name of zipfile
        zipstrings={'XV':(vertices[point_i,0]*1e5), 'YV':(vertices[point_i,1]*1e5),
                    'ZV':(vertices[point_i,2]*1e5) }

        zipname=os.path.join(storage_path,shape+'_'+zipname_vid(**zipstrings)+zip_fid)
#        zipname=os.path.join(storage_path,shape+zip_fid)

        particle_id=zipname_vid(**zipstrings)

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



        #Use symmetry if existent (Phase 4: skip only when the mirror fill-in
        #has already provided this vertex; otherwise run ADDA normally, so a
        #missing fill can never produce NaN data silently)
        if ((vertices[point_i,2]<0-1e-10  and  sym_XYobj==True) or
            (vertices[point_i,1]<0-1e-10  and  sym_XZobj==True) or
            (vertices[point_i,0]<0-1e-10  and  sym_YZobj==True)):

            if allocate_flag and mueller_array[point_i] is not None:
                continue


        filenum+=1
        print(f"Particle number {filenum}, {memstring()}")

        if ref_lev==0:


            #Check if vertex was already simulated
            #if yes, read from netcdf
            #if not, run adda

            with ty.utils.Timer():
                print(f"Checking cached data")
                output_data = acf.read_data_from_separate_netcdf(netcdf_tempname, particle_id)



            if output_data is None:

                #Call Adda
                print("Running adda")
                spawn_command('free -m')
                with ty.utils.Timer():
                    output_data=acf.adda_call(input_data)

                #Expand Mueller matrix to spherical harmonics
                print("Expanding mueller to spharmonics")
                with ty.utils.Timer():
                    [mueller_matrix_sph,rel_error, max_len_coeffs, max_ntrunc]=spha.expand_mueller_2_spharmonics(
                        output_data,eps_sphexp,diagnostics=False)



                acf.write_data_to_separate_netcdf(netcdf_tempname, particle_id,
                                          mueller_matrix_sph,
                                          rel_error, max_len_coeffs, max_ntrunc,
                                          output_data['ext_matrix'])

            else:
                mueller_matrix_sph=output_data['mueller_sph']
                rel_error=output_data['relerror']
                max_len_coeffs=output_data['max_len_coeffs']
                max_ntrunc=output_data['max_ntrunc']






            #Allocate extinction matrix array
            if allocate_flag==False:

                q=np.shape(output_data['ext_matrix'])
#                qm=np.shape(output_data['mueller_array'])

                ext_mat_array=np.empty((np.size(vertices,axis=0),q[0],q[1]))
                ext_mat_array[:] = np.nan
                ext_mat_array_old=ext_mat_array.copy()

                mueller_array = [None] * np.size(vertices,axis=0)
                mueller_array_old = copy.deepcopy(mueller_array)

                #relative error of sph expansion
                rel_err_array = np.zeros((np.size(vertices,axis=0),16))
                rel_err_array_old = rel_err_array.copy()

                #max number coefficients
                max_len_array=np.zeros((np.size(vertices,axis=0)))
                max_len_array_old=max_len_array.copy()

                #max number of truncation
                max_ntrunc_array=np.zeros((np.size(vertices,axis=0)))
                max_ntrunc_array_old=max_ntrunc_array.copy()

                allocate_flag=True

            ext_mat_array[point_i,:,:]=output_data['ext_matrix']
            mueller_array[point_i]=mueller_matrix_sph

            #relative error of sph expansion
            rel_err_array[point_i,:]=rel_error

            #max number coefficients
            max_len_array[point_i]=max_len_coeffs

            #max number of truncation
            max_ntrunc_array[point_i]=max_ntrunc

            # ---- Phase 4: mirror-symmetry write-time fill-in ---------------
            mirror_fill(point_i, particle_id, output_data, vertices,
                        ext_mat_array, mueller_array, rel_err_array,
                        max_len_array, max_ntrunc_array)

#            phis_grid=output_data['phis_grid']
#            thetas_grid=output_data['thetas_grid']

        else:

            #Allocate extinction matrix array
            if allocate_flag==False:

                q=np.shape(output_data['ext_matrix'])
#                qm=np.shape(output_data['mueller_array'])

                ext_mat_array=np.empty((np.size(vertices,axis=0),q[0],q[1]))
                ext_mat_array[:] = np.nan

                mueller_array=[None] * np.size(vertices,axis=0)

                #relative error of sph expansion
                rel_err_array = np.zeros((np.size(vertices,axis=0),16))

                #max number coefficients
                max_len_array=np.zeros((np.size(vertices,axis=0)))

                #max number of truncation
                max_ntrunc_array=np.zeros((np.size(vertices,axis=0)))

                allocate_flag=True


            #check if values already exist
            logic_temp=vertices[point_i]==vertices_old
            logic=np.sum(logic_temp,axis=1)==np.size(vertices_old,axis=1)

            if sum(logic)==1:
                # numpy 2 no longer converts a shape-(1,) boolean-selection
                # result into a scalar implicitly -> index directly
                r_idx=int(np.flatnonzero(logic)[0])

                ext_mat_array[point_i,:,:]=ext_mat_array_old[r_idx,:,:]

                mueller_array[point_i]=mueller_array_old[r_idx]

                #relative error of sph expansion
                rel_err_array[point_i,:]=rel_err_array_old[r_idx,:]

                #max number coefficients
                max_len_array[point_i]=max_len_array_old[r_idx]

                #max number of truncation
                max_ntrunc_array[point_i]=max_ntrunc_array_old[r_idx]
            else:

                #Check if vertex was already simulated
                #if yes, read from netcdf
                #if not, run adda

                with ty.utils.Timer():
                    print(f"Checking cached data")
                    output_data = acf.read_data_from_separate_netcdf(netcdf_tempname, particle_id)



                if output_data is None:

                    #Call Adda
                    print("Running adda")
                    spawn_command('free -m')
                    print(f'Before adda: {memstring()}')
                    with ty.utils.Timer():
                        output_data=acf.adda_call(input_data)

                    #Expand Mueller matrix to spherical harmonics
                    print(f'Before spharmonics: {memstring()}')
                    #print("Expanding mueller to spharmonics")
                    with ty.utils.Timer():
                        [mueller_matrix_sph,rel_error, max_len_coeffs, max_ntrunc]=spha.expand_mueller_2_spharmonics(
                            output_data,eps_sphexp,diagnostics=False)


                    print(f'Before writing cache data: {memstring()}')
                    #print("Writing cached data")
                    with ty.utils.Timer():
                        acf.write_data_to_separate_netcdf(netcdf_tempname, particle_id,
                                                  mueller_matrix_sph,
                                                  rel_error, max_len_coeffs, max_ntrunc,
                                                  output_data['ext_matrix'])
                else:
                    mueller_matrix_sph=output_data['mueller_sph']
                    rel_error=output_data['relerror']
                    max_len_coeffs=output_data['max_len_coeffs']
                    max_ntrunc=output_data['max_ntrunc']


                ext_mat_array[point_i,:,:]=output_data['ext_matrix']
                mueller_array[point_i]=mueller_matrix_sph

                #relative error of sph expansion
                rel_err_array[point_i,:]=rel_error

                #max number coefficients
                max_len_array[point_i]=max_len_coeffs

                #max number of truncation
                max_ntrunc_array[point_i]=max_ntrunc

                # Phase 4: mirror-symmetry write-time fill-in (refined passes)
                mirror_fill(point_i, particle_id, output_data, vertices,
                            ext_mat_array, mueller_array, rel_err_array,
                            max_len_array, max_ntrunc_array)

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
                    gamma=p_orientation_gamma[g_i]

                    #rotate and interpolate adda sphere
                    vertices_rot=ico.rotate_vertices(vertices,alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)

                    ext_mat_array_temp=ext_mat_array.copy()

                    [w_int,idx]=ico.interpolation_weights_icosahedric_grid(vertices_rot,inc_dir)

                    #create lab vertical polarization vectors
                    vpol_lab=aoa.create_vpol_vector(inc_dir)
                    vpol_lab=np.tile(vpol_lab,(len(idx),1))

                    #======================================================
                    # Exploit symmetry

                    idx_sym=idx.copy()

                    N_sym=np.zeros(len(idx))
                    V_sym=np.ones((len(idx),3))



                    #find symmetries
                    if sym_XYobj==True or sym_XZobj==True or sym_YZobj==True:

                        for i in range(0,len(idx)):

                            vt=np.copy(vertices[idx[i],:])
                            vt=vt.flatten()

                            #Number of symmetries
        #                    N_sym=0

                            if sym_XYobj==True and vt[2]<0:
                                vt[2]=-vt[2]
#                                N_sym[i]=N_sym[i]+1
                                V_sym[i,2]=-V_sym[i,2]

                            if sym_XZobj==True and vt[1]<0:
                                vt[1]=-vt[1]
#                                N_sym[i]=N_sym[i]+1
                                V_sym[i,1]=-V_sym[i,1]

                            if sym_YZobj==True and vt[0]<0:
                                vt[0]=-vt[0]
#                                N_sym[i]=N_sym[i]+1
                                V_sym[i,0]=-V_sym[i,0]


                            # numpy 2: find_nearest returns a size-1 array -> index it
                            r_idx=ico.find_nearest_Nvertices(vertices,vt,N=1)
                            idx_sym[i]=int(np.asarray(r_idx[0])[0])

                    logic=(idx_sym!=idx)
                    use_symmetry=int(np.ceil(np.sum(logic)/len(logic)))



                    if use_symmetry==1:

                        #construct mueller and extinction by symmetry
                        [mueller_mat_sym,ext_mat_sym]=aoa.construct_by_symmetry(
                                np.empty(0),
                                ext_mat_array_temp[idx_sym,:,:],
                                V_sym)

                        #rotate original vpol
                        vpol_rot=ico.rotate_vertices(vpol[idx,:],alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)

                        #map to new laboratory vpol
                        [mueller_mat_rot,ext_mat_rot]=aoa.pol_changer(vertices_rot[idx,:], vpol_rot,
                                vpol_lab ,np.empty(0),ext_mat_sym,np.empty(0))


                        #interpolated value
                        EXTTEMP=np.zeros((q_ext[1],q_ext[2]))


                        for i in range(0,len(idx)):

                            Ksym=ext_mat_rot[i,:,:];

                            EXTTEMP=w_int[i]*Ksym+EXTTEMP



                    # end of symmetry exploitation
                    #======================================================
                    else:

                        #rotate original vpol
                        vpol_rot=ico.rotate_vertices(vpol[idx,:],alpha*np.pi/180,beta*np.pi/180,gamma*np.pi/180)

                        #map to new laboratory vpol
                        [_,ext_mat_rot]=aoa.pol_changer(vertices_rot[idx,:], vpol_rot,
                                vpol_lab ,np.empty(0),ext_mat_array_temp[idx,:,:],np.empty(0))

                        #interpolated value
                        EXTTEMP=np.zeros((q_ext[1],q_ext[2]))


                        for i in range(0,len(idx)):

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
                                         len(incidence_angles),4,4))*np.nan)

                flag=True


            ext_mat_horiz_avg_new[n_beta,n_theta,:,:]=ext_mat_alpha


    #check convergence
    epsilon_temp=(np.abs(ext_mat_horiz_avg_new[:,:,0:2,0:2]-ext_mat_horiz_avg_old[:,:,0:2,0:2])**2)

    epsilon_temp=np.mean(np.mean(epsilon_temp,axis=0),axis=0)

    old_value_temp=np.mean(np.mean(ext_mat_horiz_avg_old[:,:,0:2,0:2]**2,axis=0),axis=0)

#    epsilon=np.sqrt(np.max(epsilon_temp.flatten()))
    epsilon=np.sqrt(np.mean(epsilon_temp.flatten()))
    old_value=np.sqrt(np.mean(old_value_temp.flatten()))



    if ref_lev>0:

        print(f'Convergence epsilon: {epsilon} <= delta*old_value: {delta*old_value}')
        converged=epsilon<=delta*old_value

        eps_mat[ref_lev]=epsilon/old_value

    else:
        converged=False



    #prepare next round
    ref_lev=ref_lev+1
    ext_mat_horiz_avg_old=np.copy(ext_mat_horiz_avg_new)
    epsilon_old=epsilon
    ext_mat_array_old=np.copy(ext_mat_array)
    mueller_array_old=copy.deepcopy(mueller_array)
    rel_err_array_old=copy.deepcopy(rel_err_array)
    max_len_array_old=max_len_array.copy()
    max_ntrunc_array_old=max_ntrunc_array.copy()

    vertices_old=np.copy(vertices)
    triangles_old=np.copy(triangles)



    print('done')


#prepare save names
matfile_name=os.path.join(storage_path,shape_name+'_overview_data.mat')
netcdf_name=os.path.join(storage_path,shape_name+'_mueller_data.nc')
xml_name=os.path.join(storage_path,shape_name+'_mueller_data.pkl')

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
output_data['adda_version']=adda_version
output_data['adda_binary']=adda_binary
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
output_data['mueller_matrix_filename']=shape_name+'_mueller_data'
output_data['shape_name']=shape_name
output_data['max_len_coeffs']=int(np.max(max_len_array))
output_data['max_ntrunc']=int(np.max(max_ntrunc_array))

## save overview data

print(f'Saving overview data in {matfile_name}')
sio.savemat(matfile_name, {'output_data':output_data}, appendmat=True, format='5', long_field_names=True)


#==============================================================================
#       ### save mueller matrix (pickle)
#       formerly ARTS binary XML via typhon (ty.arts.xml) — that module was
#       dropped in typhon >0.8 and PyARTS is unavailable on osx-arm64. The
#       file is a pure internal artifact between sub_control and post-
#       processing; the ARTS-facing layer is the SSDB NetCDF export.
#==============================================================================


print(f'Saving mueller array in {xml_name}')
with open(xml_name, 'wb') as fh:
    pickle.dump(mueller_array, fh, protocol=pickle.HIGHEST_PROTOCOL)


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

#==============================================================================
# remove tempdata
#==============================================================================

#os.rename(netcdf_tempname,netcdf_name)
#os.remove(netcdf_tempname)

for ncfilename in glob.glob(netcdf_tempname + 'vertex_*.nc'):
    os.remove(ncfilename)
