# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:48:54 2015

@author: good-cat
"""

import numpy as np

def sobel():
    """ 
    The kernel for Sobel filter. Arguments: data (3x3 2d-array).
    """
    So_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    So_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#    return So_x
    return [So_x,So_y]

#    return 0.5*(abs(data*So_x) + abs(data*So_y))
    
    if data.ndim !=2:
        raise ValueError("wrong dimension!")


def test_Sobel():
    data = np.ones((3,3))
    z = data*sobel()[0]  
    

    assert (type(z.sum()) == np.float64)



if __name__ == "__main__":
       
    test_Sobel()
