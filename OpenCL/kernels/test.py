# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:15:47 2015

@author: good-cat
"""
import numpy as np

x = np.linspace(0,599,600)
y = np.linspace(0,599,600)
X,Y = np.meshgrid(x,y)

g = np.exp(-((X-300)**2+(Y-300)**2)/(2*(10**2)))
G = np.fft.fft2(g)

C = S*G
c = np.fft.ifft2(C)
c = np.fft.fftshift(c)
c = c/c.max()
r = np.real(c)