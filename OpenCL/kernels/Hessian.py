# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:45:58 2015

@author: good-cat
"""

###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###
import numpy as np

def hessian(data):
    """ 
    The kernel for Hessian filter. Arguments: data (3x3 2d-array).
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
    
    # Calculate the components of the Hessian matrix: H = (a b)
   #                                                      (c d)

    a = 1.0*data[1,0] - 2.0*data[1,1] + 1.0*data[1,2]    
    b = .25*(data[2,2] - data[2,1] - data[1,2] + 2.0*data[1,1] - data[0,1] - data[1,0] + data[0,0])
    c = b
    d = 1.0*data[0,1] - 2.0*data[1,1] + 1.0*data[2,1]    
    
    
    t = 1.0 # check the meaning of this parameter
    module = np.sqrt(a**2 + b*c +d**2)
    trace = a + d
    determinant = a*d - c*b
    first_eig = 0.5*(a + d) + np.sqrt(0.5*(4*b**2 + (a - d)**2))
    second_eig = 0.5*(a + d) - np.sqrt(0.5*(4*b**2 + (a - d)**2))
    orientation = 0.5*np.arccos(4*b**2 + (a - d)**2)
    square_diff = (t**4)*((a - d)**2)*(((a - d)**2) - 4*b**2)
    diff_square = (t**2)*(((a - d)**2) + 4*b**2)
    
    out = np.array([module,trace,determinant,first_eig,second_eig,orientation,square_diff,diff_square])
    # return the array of features to the convolution procedure ---> !(array of 8 numbers)!    
    return out
    

###--------------------------------------------------------###
###----------------------Testing part----------------------###
###--------------------------------------------------------###

def test_hessian():
    data = np.ones((3,3))
    z = hessian(data)  
    
    assert (type(z) == np.ndarray)
    assert (z.shape[0]==8)
    assert (z.ndim==1)



if __name__ == "__main__":
       
    test_hessian()