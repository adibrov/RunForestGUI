# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:17:59 2015

@author: alex
"""

#import imgtools
from gputools import OCLProgram,  OCLArray

from numpy import *

def convolve(data,kernel):
    """convolves data with kernel"""
    NKy, NKx = kernel.shape    
    
    if not isinstance(data, OCLArray):
        data_g = OCLArray.from_array(data.astype(float32)) 
    else:
        # print "data already on the gpu!"
        data_g = data

    kernel_g = OCLArray.from_array(kernel.astype(float32)) 
    
    res_g = OCLArray.empty(data.shape,float32) 
    
    prog = OCLProgram("./OpenCL/gpu_kernels/convolve.cl")
    
    # start kernel on gput
    prog.run_kernel("convolve",   # the name of the kernel in the cl file
                    data_g.shape[::-1], # global size, the number of threads e.g. (128,128,) 
                    None,   # local size, just leave it to None
                    data_g.data,kernel_g.data,res_g.data,
                    int32(NKx),int32(NKy)) 
                    
                    
    return res_g.get()
    
if __name__ == "__main__":
    

    data = imgtools.test_images.lena()
    kernel = ones((10,10))
    kernel *= 1./sum(kernel)
    
    out = convolve(data,kernel)
    
#    print out
