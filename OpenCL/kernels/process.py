# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:17:36 2015

@author: good-cat
"""

import numpy as np
from deconv_slice import deconv_slice

refr = np.loadtxt('/home/good-cat/Documents/SLIT/python/refr_ind.txt')


def
    dist = np.zeros((refr.shape[0],refr.shape[1]))
    outp = np.zeros((refr.shape[0],refr.shape[1]))
    
    delta = 40
    
    for i in range(refr.shape[0]/delta):
        dist[i*delta:i*delta+delta,:],outp[i*delta:i*delta+delta,:] =\
                                deconv(refr[i*delta:i*delta+delta,:])
