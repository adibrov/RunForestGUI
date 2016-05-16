# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 21:56:42 2015

@author: good-cat
"""

def ks_loop(x, alpha, D) :
    import numpy as np
    ''' 
    Length of the output signal must be larger than the length of the input signal,
    that is, D must be larger than 1 
    '''
    if D < 1:
        print('Duration D must be greater than 1')
        
    # Make sure the input is a row-vector
    if x.ndim != 1:
        print('The array entered is of the wrong size')
        return None
    
    # Number of input samples
    M = len(x)
    
    # N umber of output samples
    size_y = D*M
    
    # Initialize with random input x
    y = np.zeros((size_y,1))
    for i in range(M):
        y[i] = x[i]
    
    for index in range(M,size_y):
        y[index] = float(alpha * y[index - M])
    
    return y