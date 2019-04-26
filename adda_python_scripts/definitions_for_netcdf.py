#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:16:03 2017

@author: u242031
"""

import numpy as np
from datetime import datetime
from subprocess import check_output

#==============================================================================
# Definitions general
#==============================================================================

version='beta_0.9'

groupnames=['SingleScatteringData','ShapeData','CalculationData']



#definitions 
fixed_values={}

fixed_values['aa_inc']=0
fixed_values['phaMat_index']=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
fixed_values['extMat_index']=np.array([[1,2,0,0],[2,1,0,0],[0,0,1,3],[0,0,-3,1]])
fixed_values['absVec_index']=np.array([1,2,0,0])

refrIndex_model='Maetzler2016'

#==============================================================================
# SingleScatteringData definitions
#==============================================================================

SSD_general_attribute='azimuthally_random'

# SingleScatteringData dimension lengths
dimensions=np.array([-1,
           1,
           -1,
           4,
           4,
           16,
           3,
           2])

# SingleScatteringData dimension names
SSD_dimnames=[ 'sph_coeffs_scat',       
       'aa_inc',
       'za_inc',
       'scatMat_row',
       'scatMat_Col',
       'phaMatElem',
       'extMatElem',
       'absVecElem']


SSD_definition={}

# SingleScatteringData variable names
SSD_definition['variables']=['frequency',      #0
            'temperature',      #1
            'sph_coeffs_scat',  #2
            'aa_inc',           #3
            'za_inc',           #4
            'phaMat_index',     #5
            'extMat_index',     #6
            'absVec_index',     #7
            'phaMat_data_real', #8
            'phaMat_data_imag', #9
            'extMat_data',      #10
            'absVec_data']      #11
              
SSD_definition['datatypes']=['f4','f4','i4','f4','f4','i1','i1','i1','f4','f4','f4','f4']              
              
SSD_definition['fill_values']=[np.NaN,
                 np.NaN,
                 -1,
                 np.NaN,
                 np.NaN,
                 -1,
                 -1,
                 -1,
                 np.NaN,
                 np.NaN,
                 np.NaN,
                 np.NaN]              
              

SSD_definition['dimensions']=[None,
                None,
                (SSD_dimnames[0],),
                (SSD_dimnames[1],),
                (SSD_dimnames[2],),
                (SSD_dimnames[4],SSD_dimnames[3]),
                (SSD_dimnames[4],SSD_dimnames[3]),
                (SSD_dimnames[3],),
                (SSD_dimnames[5],SSD_dimnames[1],SSD_dimnames[2],SSD_dimnames[0]),
                (SSD_dimnames[5],SSD_dimnames[1],SSD_dimnames[2],SSD_dimnames[0]),
                (SSD_dimnames[6],SSD_dimnames[1],SSD_dimnames[2]),
                (SSD_dimnames[7],SSD_dimnames[1],SSD_dimnames[2])]
                


SSD_definition['RAW_variables']=['Frequency',
              'Temperature',
              'max_len_coeffs',
              '',
              'theta_inc',
              '',
              '',
              '',
              'phase_matrix_sph',
              'phase_matrix_sph',
              'ext_mat_horiz_avg',
              'abs_vector']


SSD_definition['units']=['Hz','K','','degree','degree','','','','m^2','m^2','m^2','m^2']




SSD_definition['descriptions']=[
        'Frequency',
        'Temperature',
        'Spherical harmonics index',
        'Azimuth incident angle array',
        'Zenith incident angle array',
        'Phase matrix index',
        'Extinction matrix index',
        'Absorption vector index',
        'Phase matrix data',
        'Phase matrix data',
        'Extinction matrix data',
        'Absorption vector data']
        

SSD_attributes=['units','descriptions']


#==============================================================================
# ShapeDataDefinitions
#==============================================================================


ShD_definition={}

ShD_definition['variables']=[
        'diameter_max',
        'diameter_vol_eq',
        'diameter_area_eq_aerodynamical',
        'mass',
        'dpl',
        'N_dipoles',
        'refrIndex_real',
        'refrIndex_imag',
        'alpha',
        'beta',
        'gamma'
        ]

ShD_definition['units']=[
        'm',
        'm',
        'm',
        'kg',
        '',
        '',
        'Real part of refractive index',
        'Imaginary part of refractive index',
        'degree',
        'degree',
        'degree',
        ]


ShD_definition['descriptions']=[
        'Maximimum diameter, calculated as the minimum circumsphere diameter',
        'Volume equivalent diameter',
        '',
        '',
        'Dipoles per wavelength',
        'Number of dipoles',
        'Value only given for homogenous particles',
        'Value only given for homogenous particles',
        'Initial rotation angle alpha (zyz-notation)',
        'Initial rotation angle beta (zyz-notation)',
        'Initial rotation angle gamma (zyz-notation)'
        ]

ShD_definition['fill_values']=[
        np.NaN,
        np.NaN,
        np.NaN,
        np.NaN,
        -1,
        -1,
        np.NaN,
        np.NaN,
        np.NaN,
        np.NaN,
        np.NaN,
        ]      

ShD_definition['datatypes']=['f4','f4','f4','f4','i4','i4','f4','f4','f4','f4','f4']     


ShD_attributes=['units','descriptions']


ShD_definition['descriptions_general']=[
        'description',
        'source',
        'shape_file',
        'refrIndex_model',
        'habit_file_id',
        'habit_id',
        'phase',
        'refrIndex_homogenous_bool',
        'density_homogenous_bool'
        ]

ShD_definition['meta_description_general']=[
        'description',
        'source',
        '',
        '',
        'habit_name',
        'habit_id',
        'medium',
        'refrIndex_homogenous_bool',
        'density_homogenous_bool'
        ]


#==============================================================================
# CalculationData Definitions
#==============================================================================

CaD_definition={}

CaD_definition['descriptions_general']=[
        'method',
        'software',
        'software_version',
        'system',
        'n_nodes',
        'n_cores',
        'date_completion',
        'ADDA_eps',
        'ADDA_avgParam_file',
        'ADDA_scatParam_file',
        'Refinement_delta',
        'Maximum allowed refinement level',
        'Mueller_sph_eps',
        'Phase_sph_eps',
        'sym_XYobj',
        'sym_XZobj',
        'sym_YZobj',
        'Spherical harmonics toolbox',
        ]

CaD_definition['values']=[
        'DDA',
        'ADDA',
        '1.2',
        'DKRZ: mistral-compute2',
        'n/a',
        '36',
        datetime.now().isoformat(),
        ['eps'],
        'n/a',
        ['scat_param_path'],
        ['delta'],
        ['limit_sph'],
        ['eps_sphexp'],
        ['Phase_sph_eps'],
        ['sym_XYobj'],
        ['sym_XZobj'],
        ['sym_YZobj'],
        (''.join(check_output(['python', '-c', 'import shtns; shtns.print_version()']
        ).decode('UTF-8').split()))#'SHTns 2.8 built Jul 13 2017, 14:54:27, id: avx'
        ]




