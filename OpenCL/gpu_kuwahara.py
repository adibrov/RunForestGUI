# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:17:59 2015

@author: alex
"""
#import imgtools
from gputools import OCLProgram,  OCLArray

from numpy import *

def gpu_kuwahara(data, N=5):
    """Function to convolve an imgage with the Kuwahara filter on GPU."""
    # create numpy arrays


    if (N%2==0):       
        raise ValueError("Data has to be a (2n+1)x(2n+1) array.")

    
    data_g = OCLArray.from_array(data.astype(float32)) 
       
    res_g = OCLArray.empty((data.shape[0],data.shape[1]),float32) 
    
    prog = OCLProgram("./OpenCL/gpu_kernels/gpu_kuwahara.cl")
    
    # start kernel on gput
    prog.run_kernel("kuwahara",   # the name of the kernel in the cl file
                   data_g.shape[::-1], # global size, the number of threads e.g. (128,128,) 
                    None,   # local size, just leave it to None
                    data_g.data,res_g.data,
                    int32(N)) 
                    
    
#                    
    
    return res_g.get()
       
    
if __name__ == "__main__":
    

    data = imgtools.test_images.lena()
  
    
    out = gpu_kuwahara(data,5)
    
    #print out
