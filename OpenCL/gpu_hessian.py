# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:45:58 2015

@author: good-cat
"""

###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###

#import imgtools
from gputools import OCLProgram,  OCLArray

from numpy import *

def gpu_hessian(data):
    """Function to convolve an imgage with a bilateral filter on GPU."""
    # create numpy arrays
        
    data_g = OCLArray.from_array(data.astype(float32)) 
       
    res_g = OCLArray.empty((data.shape[0],data.shape[1],8),float32) 
    
    prog = OCLProgram("./OpenCL/gpu_kernels/gpu_hessian.cl")
    
    # start kernel on gput
    prog.run_kernel("hessian",   # the name of the kernel in the cl file
                   data_g.shape[::-1], # global size, the number of threads e.g. (128,128,) 
                    None,   # local size, just leave it to None
                    data_g.data,res_g.data) 
                    
    
#                    
    
    return res_g.get()
       
    
if __name__ == "__main__":
    

    data = imgtools.test_images.lena()
  
    
    out = gpu_hessian(data)
    
    #print out
