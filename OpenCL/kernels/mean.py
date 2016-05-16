# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:49:21 2015

@author: good-cat
"""

import numpy as np

def Mean(Nx=10,Ny=10):
    """ 
    The kernel for mean filter. Arguments: data (2d-array).
    """
    return (1.0/(Nx*Ny))*np.ones((Nx,Ny))
    
   
def test_Mean():
    data = np.ones((100,100))
    z = data*Mean(100,100)*10000  
    
    assert (data == z).all()
    assert (type(z.sum()) == np.float64)
    
    print "mean tested"


if __name__ == "__main__":
       
    test_Mean()
