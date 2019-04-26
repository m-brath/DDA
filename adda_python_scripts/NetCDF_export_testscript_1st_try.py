#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:10:37 2017

@author: u242031
"""

import numpy as np
import adda_control_functions as acf
from netCDF4 import Dataset


data_main_folder='/scratch/uni/u237/user_data/mbrath/adda_results/HongPlate_Id9_horizavg/'
data_folder='HongPlate_Id9_deq00010um_freq001.0GHz_T0190K/'
data_name='HongPlate_Id9_horiz_avg_phasematrix.mat'



# load test data
filename=data_main_folder+data_folder+data_name
data=acf.loadmat(filename)

output_data=data['output_data']


        



