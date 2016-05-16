# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:17:32 2015

@author: good-cat
"""

# Imported libraries for math and plots
#------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import signal

#------------------------------------------------#

def combine(data,delta,step):
    out = np.zeros((data.shape[1],data.shape[1]))
    for i in range(data.shape[0]/delta-3):
        #print i
        #print out[i*step:i*step+delta,:].shape
        #print data[i*delta:i*delta+delta,:].shape
        out[i*step:i*step + delta,:] += (1.0*step/delta)*data[i*delta:i*delta + delta,:]
        
    return out[40:-40,:]
        