# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:32:07 2015

@author: good-cat
"""

import numpy as np

av_im = np.loadtxt('/home/good-cat/Downloads/files/bg_imag.txt')
av_re = np.loadtxt('/home/good-cat/Downloads/files/bg_real.txt')
spec_im = np.loadtxt('/home/good-cat/Downloads/files/fg_imag.txt')
spec_re = np.loadtxt('/home/good-cat/Downloads/files/fg_real.txt')

av = av_re + 1.0j*av_im
spec = spec_re + 1.0j*spec_im
ref = np.loadtxt('/home/good-cat/Documents/SLIT/python/refr_ind.txt')
ref=np.transpose(ref)
bla = spec - av

for i in range(600):
    for j in range(600):
        if (np.isnan(bla[i,j])):
            bla[i,j]=0 
            spec[i,j]=0
            av[i,j]=0
        
        
        if (bla[i,j]<=-np.pi):
            bla[i,j] = bla[i,j]+np.pi
        elif (bla[i,j]>np.pi):
            bla[i,j] = bla[i,j]-np.pi


        if (av[i,j]<=-np.pi):
            av[i,j] = av[i,j]+np.pi
        elif (av[i,j]>np.pi):
            av[i,j] = av[i,j]-np.pi

        if (spec[i,j]<-np.pi):
            spec[i,j] = spec[i,j]+np.pi
        elif (spec[i,j]>np.pi):
            spec[i,j] = spec[i,j]-np.pi