# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:02:32 2015

@author: good-cat
"""

###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###

import numpy as np

def gauss(Nx=10, Ny=10, sigma_x = 4.0, sigma_y = 4.0):
    """ 
    The kernel for Gaussian filter. Arguments: data (2d-array).
    """

###--------------------------------------------------------###
###-------------------Possible errors----------------------###
###--------------------------------------------------------###
    

        
###--------------------------------------------------------###
###----------------------Main part-------------------------###
###--------------------------------------------------------###        
        
   
    x = np.linspace(0,Nx-1,Nx)
    y = np.linspace(0,Nx-1,Nx)
    
    X,Y = np.meshgrid(x,y)
    
    
    G = np.exp(-0.5*(((X-x[Nx/2])/sigma_x)**2 + ((Y-y[Ny/2])/sigma_y)**2))
    
    return (1/G.sum())*G

 
    
        
###--------------------------------------------------------###
###----------------------Testing part----------------------###
###--------------------------------------------------------###

def test_gauss():
    data = np.ones((10,10))
    z = gauss()*data  
    
    
   # assert (type(z) == np.array)
    assert ((z.sum()<=data.sum()) and (z.sum()>=-data.sum()))
    

if __name__ == "__main__":
       
    test_gauss()