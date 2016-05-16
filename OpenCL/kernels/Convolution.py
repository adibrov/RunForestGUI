# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:23:57 2015

@author: good-cat
"""
###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###

import numpy as np

def conv(data,kernel,patch_size=3,dim=1):
    """
    Function to perform a 2d-convolution with the defined kernel.
    """
###--------------------------------------------------------###
###--------------Kernel-dependent conditions---------------###
###--------------------------------------------------------###

    if (patch_size%2 == 0):
        raise ValueError("Patch size dimension has to be odd.")  
    if (kernel.__name__ == 'hessian'):
        dim = 8
    if (kernel.__name__ == 'proj'):
        dim = 6
        patch_size = 19
    if (kernel.__name__ == 'structure'):
        dim = 2
###--------------------------------------------------------###      
###----------------------Main part-------------------------###
###--------------------------------------------------------###      

    N = patch_size # short variable name
    delta = (N-1)/2 # deviation from the center of the patch
    out = np.zeros((data.shape[0],data.shape[1],dim)) 
    d = np.zeros((data.shape[0]+2*N,data.shape[1]+2*N))
    d[N:-N,N:-N] = data
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = kernel(d[i+N-delta:i+N+delta+1,j+N-delta:j+N+delta+1])
    
    return out
    
###--------------------------------------------------------###