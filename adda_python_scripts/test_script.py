# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:18:59 2015

@author: u242031
"""
import numpy as np




def test_funktion(x):
    y=x**2
    z=np.array([x,y])
    
    return y, z
    
    
    
    
    
    
x=2.

[a,b]=test_funktion(x)    