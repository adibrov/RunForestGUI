# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:53:19 2015

@author: good-cat
"""
###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###

import numpy as np

def bilateral(Nx,Ny, sigma_dist=3.0, sigma_int=50.0):
    """ 
    The kernel for Bilateral filter. Arguments: data ((2n+1)x(2n+1) 2d-array).
    """

        
###--------------------------------------------------------###
###----------------------Main part-------------------------###
###--------------------------------------------------------###
        
        
 
    w = np.zeros((Nx,Ny))  # initializing kernel with zeros
    norm = 0.0
    
    
    # programming every array-unit of the kernel
    for i in range(Nx): 
        for j in range(Ny):
            w[i,j] = np.exp(-0.5*((i-N/2)**2+(j-N/2)**2)/(sigma_dist**2)\
                    - 0.5*((data[i,j]-data[N/2,N/2])**2)/(sigma_int**2))
            norm = norm + w[i,j]

                    
    # return the sum to the convolution procedure ---> !(number)!
    return (1.0/norm)*(w*data).sum()
    
###--------------------------------------------------------###
###----------------------Testing part----------------------###
###--------------------------------------------------------###
    

def test_bilateral():
    data = np.ones((3,3))
    data1 = np.zeros((5,5))
    z = bilateral(data)
    z1 = bilateral(data1)
    
    assert (type(z) == np.float64)
    assert ((z<=data.sum()) and (z>=data.min()))

    assert (type(z1) == np.float64)
    assert (z1 == 0.0)
    assert ((z1<=data1.max()) and (z1>=data1.min()))

if __name__ == "__main__":       
    test_bilateral()
    
###--------------------------------------------------------###
