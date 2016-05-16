# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:48:11 2015

@author: alex
"""

from volust.volgpu import OCLProgram,  OCLArray

from numpy import *

def add_1d():
    # create numpy arrays
    a = linspace(0,1,128).astype(float32)
    b = linspace(0,1,128).astype(float32)
    
    # create buffers on the gpu
    a_g = OCLArray.from_array(a) 
    b_g = OCLArray.from_array(b) 
    res_g = OCLArray.empty_like(a) 
    
    # compile program
    prog = OCLProgram("/home/alex/python/RunForest/OpenCL/kernels/add_kernel.cl")
    
    # start kernel on gput
    prog.run_kernel("add",   # the name of the kernel in the cl file
                    res_g.shape, # global size, the number of threads e.g. (128,128,) 
                    None,   # local size, just leave it to None
                    a_g.data,b_g.data,res_g.data)  # the pointers to the memory buffers
                    
                    
    res = res_g.get()
    return res
    
def add_2d():
    # create numpy arrays
    a = ones((128,128),float32)
    
    
    # create buffers on the gpu
    a_g = OCLArray.from_array(a) 
    
    res_g = OCLArray.empty_like(a) 
    
    # compile program
    prog = OCLProgram("/home/alex/python/RunForest/OpenCL/kernels/add_kernel.cl")
    
    # start kernel on gput
    prog.run_kernel("add2",   # the name of the kernel in the cl file
                    res_g.shape, # global size, the number of threads e.g. (128,128,) 
                    None,   # local size, just leave it to None
                    a_g.data,float32(10.),res_g.data)  # the pointers to the memory buffers
                    
                    
    res = res_g.get()
    
if __name__ == "__main__":
    

    add_1d()
    add_2d()
    