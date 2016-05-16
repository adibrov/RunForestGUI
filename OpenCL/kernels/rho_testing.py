# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:35:55 2015

@author: good-cat
"""

import deconv_data as deconv_data
import numpy as np

refr = np.loadtxt('/home/good-cat/Documents/SLIT/python/refr_ind.txt')

q = np.zeros(10)

for i in range(10):
    q[i] = ((refr-deconv_data.deconv(refr,40,40,i+ 1)[1])**2).sum()