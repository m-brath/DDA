#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-6 GATE 2 config — tiny oblate ellipsoid (x=0.3), one size/frequency/T.

Used by analysis/phase6_backscatter.py to check the exact-backscatter Mueller
matrix against the pipeline's own phase matrix near Theta_s = 180 degree
(convention scan + assertion). Clone of adda_config_phase3_test.py with
x=0.3 and dpl=300 (at x=0.1 a dpl of 100 would leave <4 voxels across the
particle axis — the asphericity that pins the Stokes-basis convention would
be discretization-lost; see report Phase 6).

    ADDA_BINARY=<adda-exe> ADDA_VERSION=<ver> \
        python adda_sub_control_spharmonics.py 0 0 0 adda_config_phase6_gate2

NOT a production habit config.
"""

import numpy as np
import socket
import os
import sys

#==============================================================================
#  constants / cluster detection (same structure as production configs)
#==============================================================================

c0=299792458 #[m/s]

hereiam=socket.gethostname()

if hereiam[0:6]=='mlogin' or hereiam[0:7]=='mistral' or hereiam[0:2]=='m2': #we are on mistral

    prefix='/work/um0878/users/mbrath/'
    prefix_data='/work/um0878/user_data/mbrath/'

    calc_type='mistral'

    ncores=36
    Partition='compute2'

elif sys.platform.startswith('darwin'):

    # local (no-queue) mode (Phase 3): paths relative to this repo
    _here=os.path.dirname(os.path.abspath(__file__))+'/'
    _repo=os.path.dirname(_here[:-1])+'/'
    prefix=_here
    prefix_data=_repo
    calc_type='local'
    Partition='n/a'
    #number of cores
    ncores=4

else:
    prefix='/scratch/uni/u237/users/mbrath/'
    prefix_data='/scratch/uni/u237/user_data/mbrath/'

    calc_type='thunder'

    #number of cores
    ncores=16


#==============================================================================
# ADDA checkout, provenance and binary (Phase 3 - configurable binary)
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

adda_version='1.2'

if calc_type=='mistral':
    adda_binary=adda_prefix+'adda_1.2/src/mpi/adda_mpi'
    adda_launcher=['srun','-n',str(ncores)]
elif calc_type=='local':
    adda_binary=adda_prefix+'adda_1.2/src/seq/adda'
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
    ramdrive=None      # system temp dir
else:
    ramdrive="/dev/shm/u242031/"

#==============================================================================
#  paths
#==============================================================================

scat_param_path=adda_prefix+'baselines/scat_params_baseline.dat'

main_path=prefix_data+'adda_results/'

subpath='phase6_gate2/rawdata/'

force_new=True

#==============================================================================
#  Input parameters (single tiny case)
#==============================================================================

frequencies=np.array([183.])*1e9 #[Hz]

lamb=c0/frequencies

temperatures=np.array([230.]) #[K]

#size parameter (x=0.3: smooth near the backscatter pole, but asphericity
#still resolved at dpl 300 — required to pin the convention, see docstring)
x=np.array([0.3])

part_ar=2 #aspect ratio (oblate)

#particle(grid) size [um]
Sizes=lamb*x/np.pi*1e6
Dmax=Sizes*(part_ar)**(1/3)

size_parameter_limit=10

#orientation sampling (tiny: 3 betas x 5 incidence angles)
number_of_betas=3

incidence_angles=np.linspace(0.,180.,num=5)

#limits and break criteria
delta=0.02
limit_sph=2
eps=3
eps_sphexp=0.02

ZIPPING=False

#==============================================================================
#  Particle definition (analytic oblate ellipsoid)
#==============================================================================

alpha_0=0
beta_0=0
gamma_0=0

shape='ellipsoid'
shape_str=['-shape',shape,'1.0',str(1/part_ar)]
shape_name=(shape+'_'+shape_str[2]+'_'+shape_str[3]+'_'+
            str(alpha_0)+'-'+str(beta_0)+'-'+str(gamma_0))

#symmetries (deliberately off — plain averaging path)
sym_Zobj=np.inf
sym_XYobj=False
sym_XZobj=False
sym_YZobj=False
sym_Zlab=False

#==============================================================================
#  Adda inputs
#==============================================================================

scat_output='-store_scat_grid'
scat_input='-scat_grid_inp'
scat_mat='-scat_matr'
scat_mat_type='both'
dpl_input='-dpl'
dpl_str='300'

scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type,
                dpl_input,dpl_str]

static_clo=shape_str+scatoutput_str
