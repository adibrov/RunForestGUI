# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:17:59 2015

@author: alex
"""
#import imgtools
from gputools import OCLProgram,  OCLArray

from numpy import *

def gpu_bilateral(data, Nx=10,Ny=10,sigma_int=10.0,sigma_dist=10.0):
    """Function to convolve an imgage with a bilateral filter on GPU."""
    # create numpy arrays
        
    data_g = OCLArray.from_array(data.astype(float32)) 
       
    res_g = OCLArray.empty(data.shape,float32) 
    
    prog = OCLProgram("./OpenCL/gpu_kernels/gpu_bilateral.cl")
    
    # start kernel on gput
    prog.run_kernel("bilateral",   # the name of the kernel in the cl file
                   data_g.shape[::-1], # global size, the number of threads e.g. (128,128,) 
                    None,   # local size, just leave it to None
                    data_g.data,res_g.data,
                    int32(Nx),int32(Ny),float32(sigma_int),float32(sigma_dist)) 
                    
    
#                    
    
    return res_g.get()
       
    
if __name__ == "__main__":
    
    import imgtools
    data = imgtools.test_images.lena()
  
    
    out = gpu_bilateral(data)
    
    #print out
