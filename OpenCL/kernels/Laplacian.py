# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:02:32 2015

@author: good-cat
"""

###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###

import numpy as np

def laplacian():
    """ 
    The kernel for Laplacian filter. Arguments: data (2d-array).
    """

###--------------------------------------------------------###
###-------------------Possible errors----------------------###
###--------------------------------------------------------###
    

        
###--------------------------------------------------------###
###----------------------Main part-------------------------###
###--------------------------------------------------------###        
        
   

    
    L = np.zeros((3,3))
    L[1,1] = -4.0
    L[0,0] = 1.0
    L[0,2] = 1.0
    L[2,0] = 1.0
    L[2,2] = 1.0
    
    
    return L

 
    
        
###--------------------------------------------------------###
###----------------------Testing part----------------------###
###--------------------------------------------------------###

def test_laplacian():
    data = np.ones((3,3))
    z = laplacian()*data  
    
    
    assert (type(z) == np.ndarray)
   # assert ((z.sum()<=data.sum()) and (z.sum()>=-data.sum()))
    

if __name__ == "__main__":
       
    test_laplacian()