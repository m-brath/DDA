#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:15:52 2017

@author: u242031
"""

from joblib import Parallel, delayed
#import multiprocessing

# what are your inputs, and what operation do you want to
# perform on each input. For example...
inputs = range(10)

def processInput(i,b):
    return i * i+b

num_cores = 3#multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,2) for i in inputs)