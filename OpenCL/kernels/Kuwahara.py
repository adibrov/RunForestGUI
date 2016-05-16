# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:40:56 2015

@author: good-cat
"""
###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###

import numpy as np

def kuwahara(data):
    """ 
    The kernel for Kuwahara filter. Arguments: data ((2n+1)x(2n+1) 2d-array).
    """
    
###--------------------------------------------------------###
###-------------------Possible errors----------------------###
###--------------------------------------------------------###
    
    if data.ndim !=2:
        raise ValueError("Data has to be a 2d-array.")    
    
    if ((data.shape[0]%2==0) or (data.shape[1]%2==0)):       
        raise ValueError("Data has to be a (2n+1)x(2n+1) array.")
    
    
###--------------------------------------------------------###
###----------------------Main part-------------------------###
###--------------------------------------------------------###
    
    N = data.shape[0]
    
    aux = np.zeros((1+N/2,1+N/2,4))
    aux[:,:,0] = data[:N/2+1,:N/2+1]
    aux[:,:,1] = data[:N/2+1,N/2:]
    aux[:,:,2] = data[N/2:,:N/2+1]
    aux[:,:,3] = data[N/2:,N/2:]
    sigma = np.zeros(4)
    for i in range(4):
        sigma[i] = np.std(aux[:,:,i])
        
    out = 0
    sigma_min = min(sigma)
    for i in range(4):
        if (sigma[i]==sigma_min):
            out = np.mean(aux[:,:,i])
     # return the sum to the convolution procedure ---> !(number)!
    return out
    
###--------------------------------------------------------###
###----------------------Testing part----------------------###
###--------------------------------------------------------###


def test_kuwahara():
    data = np.ones((3,3))
    z = kuwahara(data)  
    assert (type(z) == np.float64)



if __name__ == "__main__":
       
    test_kuwahara()