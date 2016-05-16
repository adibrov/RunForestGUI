# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:33:17 2015

@author: good-cat
"""
import numpy as np


def myfilter(data,sigma = 1.):
    """ this is my awesome filter 
    """
    if data.ndim !=2:
        raise ValueError("wrong dimension!")
    return data*sigma


def test_filter():
    data = np.ones((100,100))
    for sig in np.random.uniform(-1,1,10):
        print sig
        out = myfilter(data,sig)
        assert out[0,0] == data[0,0] * sig



if __name__ == "__main__":
   
    
    
    test_filter()