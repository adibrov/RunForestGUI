# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:32:07 2015

@author: good-cat
"""

import numpy as np

av_im = np.loadtxt('/home/good-cat/Documents/SLIT/python/new_data/bg_imag.txt')
av_re = np.loadtxt('/home/good-cat/Documents/SLIT/python/new_data/bg_real.txt')
spec_im = np.loadtxt('/home/good-cat/Documents/SLIT/python/new_data/fg_imag.txt')
spec_re = np.loadtxt('/home/good-cat/Documents/SLIT/python/new_data/fg_real.txt')
refr = np.loadtxt('/home/good-cat/Documents/SLIT/python/refr_ind.txt')
x = np.linspace(0,599,600)
y = np.linspace(0,599,600)
X,Y = np.meshgrid(x,y)
g = np.exp(-0.5*((X-300)**2+(Y-300)**2)/(10**2))
G = np.fft.fft2(g)
R = np.fft.fft2(refr)

re = (1.0/g.sum())*np.fft.ifftshift(np.fft.ifft2(G*R))



av = av_re + 1.0j*av_im
spec = spec_re + 1.0j*spec_im

