# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 00:05:50 2015

@author: good-cat
"""
import numpy as np

# --- Convolutional kernels --- #

# --- Mean --- #

def mean(N):
    return np.ones((N+1,N+1))

M = mean(2)

# --- Sobel filter --- #

So_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
So_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])


# --- Gabor filter --- #
N = 10
x = np.linspace(0,N-1,N)
y = np.linspace(0,N-1,N)

X,Y = np.meshgrid(x,y)

sigma_x = 2.0
sigma_y = 2.0

kx = 2.0*np.pi/7.0
ky = 2.0*np.pi/7.0


g = np.exp(-0.5*(((X-x[N/2])/sigma_x)**2 + ((Y-y[N/2])/sigma_y)**2))
s = np.exp(-1.0j*(kx*X + ky*Y))


Ga = g*s

# --- Gaussian --- #
# Coupled to the grid introduced in Gabor filter

sig_x = 4.0
sig_y = 4.0
G = np.exp(-0.5*(((X-x[N/2])/sig_x)**2 + ((Y-y[N/2])/sig_y)**2))

# --- Laplacian --- #
L = np.array([[0,1,0],[1,-4,1],[0,1,0]])

