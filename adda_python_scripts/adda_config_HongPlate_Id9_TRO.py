# -*- coding: utf-8 -*-
"""
TRO (totally random orientation) config: HongPlate_Id9_TOR_sizegrid
(Phase 5, plan step 21 — replaces the archived old_configs/*_RndOrient.py
pattern with the Phase-3 config contract + temperature-carrying liste).

Strategy decision (report Phase-5 §5.2): ADDA 1.4.0 `-orient avg` delivers
cross sections AND the orientation-averaged angular Mueller/amplitude product
in one Romberg-integrated call (verified 2026-11-06: exit 0, `mueller` +
`CrossSec` written, alpha-integration error ~1e-13; only -scat_grid/-Csca/
-alldir combinations are rejected). Own orientation sampling is therefore NOT
needed for TRO products. The avg input exploits the voxel-exact XY mirror of
this size grid (beta 0-90; adda_aux_input/HongPlate_Id9_TOR_sizegrid/).

Run via adda_main_control_rndorient.py (local branch = direct execution).
`liste` uses the 9-row format: row 8 = temperature (the T####K folder labels
of the old 8-row runs were flat indices — see report §1.6-10).
"""

#==============================================================================
# import modules
#==============================================================================

import numpy as np
import glob
import socket
import os
import sys
import adda_control_functions as acf

#==============================================================================
#  constants
#==============================================================================

c0=299792458 #[m/s]

#==============================================================================
# Names/paths
#==============================================================================

hereiam=socket.gethostname()

if hereiam[0:6]=='mlogin' or hereiam[0:7]=='mistral' or hereiam[0:2]=='m2':

    prefix='/work/um0878/users/mbrath/'
    prefix_data='/work/um0878/user_data/mbrath/'

    calc_type='mistral'
    ncores=36
    Partition='compute2'

elif sys.platform.startswith('darwin'):

    # local (no-queue) mode: paths relative to this repo
    _here=os.path.dirname(os.path.abspath(__file__))+'/'
    _repo=os.path.dirname(_here[:-1])+'/'
    prefix=_repo
    prefix_data=_repo
    calc_type='local'
    Partition='n/a'
    ncores=4

else:
    prefix='/scratch/uni/u237/users/mbrath/'
    prefix_data='/scratch/uni/u237/user_data/mbrath/'

    calc_type='thunder'
    ncores=32


#==============================================================================
# ADDA checkout, provenance and binary (Phase 3 contract)
#==============================================================================

if calc_type=='local':
    adda_prefix=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
else:
    adda_prefix=prefix+'adda/'

if calc_type=='mistral':
    cluster_name='DKRZ: mistral-compute2'
elif calc_type=='local':
    cluster_name='local: '+hereiam
else:
    cluster_name='TCUB: thunder'

# PRODUCTION binary (Phase-3 decision): 1.4.0
adda_version='1.4.0'

if calc_type=='mistral':
    adda_binary=adda_prefix+'adda_1.4/src/mpi/adda_mpi'
    adda_launcher=['srun','-n',str(ncores)]
elif calc_type=='local':
    adda_binary=adda_prefix+'adda_1.4/src/seq/adda'
    adda_launcher=[]
else:
    adda_binary='adda_mpi'
    adda_launcher=['mpiexec','-n',str(ncores)]

adda_version=os.environ.get('ADDA_VERSION',adda_version)
adda_binary=os.environ.get('ADDA_BINARY',adda_binary)

#ramdrive
if calc_type=='mistral':
    ramdrive="/dev/shm/"
elif calc_type=='local':
    ramdrive=None
else:
    ramdrive="/dev/shm/u237/mbrath/"

#orientation-averaging input (XY-mirror-reduced beta; see file header)
avg_param_path=adda_prefix+'adda_aux_input/HongPlate_Id9_TOR_sizegrid/avg_params.dat'

#main path
main_path=prefix_data+'adda_results/'

#shape name
shape_name='HongPlate_Id9_TOR'

#shape folder_name (the size grid on disk)
shape_foldername='HongPlate_Id9_TOR_sizegrid'

#shape mainpath
shape_mainpath=adda_prefix+'adda_shape_files/'

#shape_path
shape_path=shape_mainpath+shape_foldername+'/'

#subfolder
subpath=shape_foldername+'/'

#no -shape tokens baked into static_clo (shape comes per-liste-column)
shape_str=[]


#==============================================================================
# Particle definition (size grid with 5 resolution classes per size)
#==============================================================================

shapelist = glob.glob(shape_path+"*adda")
shapelist.sort()

shapelist_mat = glob.glob(shape_path+"*.mat")
shapelist_mat.sort()

Sizes=np.zeros([np.size(shapelist_mat)])
Dmax=np.zeros([np.size(shapelist_mat)])
fmin=np.zeros([np.size(shapelist_mat)])

for i in range(np.size(shapelist_mat)):

    dummy=acf.loadmat(shapelist_mat[i])
    meta_data=dummy['meta']
    Sizes[i]=meta_data['diameter_vol_eq']*1e6 #[µm]
    Dmax[i]=meta_data['diameter_max']*1e6 #[µm]
    fmin[i]=meta_data['frequency_min'] #[Hz]


#==============================================================================
#  Input parameters
#==============================================================================

# same production channel list as the ARO habit configs
frequencies=np.array([1,1.4, 3, 5, 7., 9, 10, 10.65, 13, 15, 19, 24, 31, 32, 36, 50, 58, 89,
                      94, 115, 122, 164, 167, 175, 191, 228, 247, 314, 336,
                      439, 457, 657, 671, 862, 886.])*1e9

temperatures=np.array([190., 230., 270.])

#size parameter cap x = pi*D_veq/lambda (identical rule to adda_main_control.py)
size_parameter_limit=10

#precision of adda
eps=3

#estimated batch time [minutes] (mistral batch path only)
batch_time=60.

#ADDA scattering-matrix output type; with '-orient avg' ADDA Romberg-integrates
# cross sections AND the orientation-averaged angular Mueller matrix (1.4/1.5;
# verified end-to-end 2026-11-06). -Csca/-alldir/-scat_grid are ILLEGAL with
# -orient avg in 1.4/1.5 and were never parsed anyway (Phase 3).
scat_mat='-scat_matr'
scat_mat_type='both'
orient_input=['-orient','avg']

scatoutput_str=[scat_mat,scat_mat_type]+orient_input+[avg_param_path]

static_clo=shape_str+scatoutput_str


#==============================================================================
#  liste: flat job table (9 rows; row 8 = temperature)
#==============================================================================
#   0 shape_name | 1 eq radius [m] | 2 eps | 3 m_re | 4 m_im | 5 lambda [m]
#   6 shape file basename | 7 batch_time | 8 temperature [K]
#
# Resolution selection identical to adda_main_control.py: for each (size,f)
# pick the shape file with the SMALLEST fmin >= f (least fine discretization
# that is still resolved at f); sizes with no fmin >= f are skipped there.

def build_liste(freqs, temps, size_keep=None):

    if size_keep is None:
        size_keep=np.ones(np.shape(Sizes),dtype=bool)

    cols=[]

    for j,f in enumerate(freqs):
        for k,T in enumerate(temps):

            m=np.sqrt(acf.eps_ice_matzler06(f,T))

            for D in np.unique(Sizes[size_keep]):

                r_idx=np.where((Sizes==D)&size_keep)[0]

                df=fmin[r_idx]-f
                df[df<0]=np.inf

                if not np.isfinite(df).any():
                    continue

                Dindex=r_idx[np.argmin(df)]

                #size parameter skip (x = pi * D_veq / lambda)
                if np.pi*D*1e-6*(f/c0) > size_parameter_limit:
                    continue

                lamb=c0/f

                col=[np.array([shape_name],dtype=object),
                     np.array([[D/2./1e6]]),          # eq RADIUS in m
                     np.array([[eps]]),
                     np.array([[np.real(m)]]),
                     np.array([[np.imag(m)]]),
                     np.array([[lamb]]),              # m
                     np.array([os.path.basename(shapelist[Dindex])]),
                     np.array([[batch_time]]),
                     np.array([[T]])]

                cols.append(col)

    liste=np.zeros((9,len(cols)),dtype=object)
    for c,col in enumerate(cols):
        for r in range(9):
            liste[r,c]=col[r]

    return liste


liste=build_liste(frequencies,temperatures)


#==============================================================================
# post-processing / chunking defaults
#==============================================================================

num_chunks=48   # per-(f,T,size) runs are minutes-cheap with orient-avg Romberg
