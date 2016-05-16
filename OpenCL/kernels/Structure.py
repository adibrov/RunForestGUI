# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:22:32 2015

@author: good-cat
"""

###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###

import numpy as np

def structure(data):
    """ 
    The kernel for Structure filter. Arguments: data (3x3 2d-array).
    """
    
###--------------------------------------------------------###
###-------------------Possible errors----------------------###
###--------------------------------------------------------### 
    
    if data.ndim !=2:
        raise ValueError("Data has to be a 2d-array.")    
    
    if data.shape !=(3,3):       
        raise ValueError("Data has to be a 3x3 array.")
        
###--------------------------------------------------------###
###----------------------Main part-------------------------###
###--------------------------------------------------------###        
    
    # Calculate the components of the Structure tensor: S = (a b)
   #                                                        (c d)
    
    a = 0.25*(1.0*data[1,2] -  1.0*data[1,0])**2
    b = 0.25*(1.0*data[1,2] -  1.0*data[1,0])*(1.0*data[2,1] -  1.0*data[0,1])
    
    d = 0.25*(1.0*data[2,1] -  1.0*data[0,1])**2
    
    
 
    first_eig = 0.5*(a + d) + np.sqrt(0.5*(4*b**2 + (a - d)**2))
    second_eig = 0.5*(a + d) - np.sqrt(0.5*(4*b**2 + (a - d)**2))
    t = [first_eig,second_eig]
    
    out = np.array([min(t),max(t)])
    
    # return the array of features to the convolution procedure ---> !(array of 2 numbers)!    
    return out
    
###--------------------------------------------------------###
###----------------------Testing part----------------------###
###--------------------------------------------------------###



def test_structure():
    data = np.ones((3,3))
    z = structure(data)  
    
    assert (type(z[0]) == np.float64)



if __name__ == "__main__":
       
    test_structure()