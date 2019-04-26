#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:10:37 2017

@author: u242031
"""

#import numpy as np
import os
import netCDF4



def ssdb_read_ncfile(nc_file):
    #2016-11-10 Robin Ekelund
    """Reads input netCDF file for data.

    Parameters
    ----------
    nc_file: netCDF file name (incl. path).

    Returns
    -------
    Single scattering data (dict)
    """
    def get_ncData(dataSet): # Recurse data extraction
        data = {}
        # Extracts attributes
        for attribute in dataSet.ncattrs():
            data[attribute] = getattr(dataSet, attribute)
        # Extract variables
        for variable in dataSet.variables.keys():
            
            if variable=='phaMat_index':
                print('halt!')
            
            tmp_dict = {'data': dataSet.variables[variable][:],#.data,
                        'dim': dataSet.variables[variable].dimensions}
            data[variable] = tmp_dict
        # Extract groups
        for group in dataSet.groups.keys():
            data[group] = get_ncData(dataSet.groups[group])
        return data

    # Read data
    assert(os.path.exists(nc_file)), \
      'NetCDF data file does not exist:\n%s\n' %nc_file
    data = netCDF4.Dataset(nc_file, 'r', 'NETCDF4')
    SSD = get_ncData(data)
    data.close()

    return SSD


#==============================================================================
# ## paths
#==============================================================================

filename=('/scratch/uni/u237/user_data/mbrath/adda_results/SSD_1st_try/'+
          'Azimuthally_random/ice/SingleCrystals/Pristine/HongPlate_Id9/'+
          'Dmax00013um_Dveq00010um_Mass4.79983e-13kg/beta000deg/'+
          'Freq0164.000GHz_T190K.nc')


SSD=ssdb_read_ncfile(filename)



