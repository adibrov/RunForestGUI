# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:17:59 2015

@author: alex
"""
#import imgtools
from gputools import OCLProgram,  OCLArray

from numpy import *

def gpu_mean(data, Nx=10,Ny=10):
    """Function to convolve an imgage with a mean filter on GPU."""
    # create numpy arrays
    
    
    data_g = OCLArray.from_array(data.astype(float32)) 
       
    res_g = OCLArray.empty(data.shape,float32) 
    
    prog = OCLProgram("./OpenCL/gpu_kernels/gpu_mean.cl")
    
    # start kernel on gput
    prog.run_kernel("mean",   # the name of the kernel in the cl file
                    data_g.shape[::-1], # global size, the number of threads e.g. (128,128,) 
                    None,   # local size, just leave it to None
                    data_g.data,res_g.data,
                    int32(Nx),int32(Ny)) 
                    
                    
    return res_g.get()
        
    
if __name__ == "__main__":
    

    data = imgtools.test_images.lena()
  
    
    out = gpu_mean(data,10)
    
    #print out
