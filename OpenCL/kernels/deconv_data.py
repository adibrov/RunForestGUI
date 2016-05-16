# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:17:36 2015

@author: good-cat
"""

import numpy as np
from deconv_slice import deconv_slice

refr = np.loadtxt('/home/good-cat/Documents/SLIT/python/refr_ind.txt')

def deconv(data,delta=40,step=40,rho=10.5):
    
    dist = np.zeros((data.shape[0], data.shape[1]))
    outp = np.zeros((data.shape[0], data.shape[1]))
    
    #delta = 40
    
    for i in range(data.shape[0]/step):
        if i*step+delta <=data.shape[0]:
            M = (1.0*step/delta)
                
            dist[i*step:i*step+delta,:] += M*deconv_slice(data[i*step:i*step+delta,:],rho)[0]
            outp[i*step:i*step+delta,:]+= M*deconv_slice(data[i*step:i*step+delta,:],rho)[1]
            
    for i in range(delta/step):
        dist[i*step:i*step+step,:] *= (1.0*delta/step)/ (i+1)
        outp[i*step:i*step+step,:] *= (1.0*delta/step)/ (i+1)
        j = data.shape[0]/step - i - 1
        dist[j*step:j*step+step,:] *= (1.0*delta/step)/ (i+1)
        outp[j*step:j*step+step,:] *= (1.0*delta/step)/ (i+1)
    return dist, outp
