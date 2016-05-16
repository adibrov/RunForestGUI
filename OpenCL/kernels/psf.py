# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:38:54 2015

@author: good-cat
"""

# Imported libraries for math and plots
#------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import signal

#------------------------------------------------#
# Constants
#------------------------------------------------#
Fs = 100 # sampling frequency
#------------------------------------------------#


# Meshgrid
#------------------------------------------------#
kx = np.linspace(0,99,Fs)
kz = np.linspace(0,99,Fs)
KX,KZ = np.meshgrid(kx,kz)
#------------------------------------------------#

# Average refracted index
#------------------------------------------------#
n_av = 1.3392237097317652

#------------------------------------------------#

w_min = 1.0
w_max = 20.0

#Main part of the PSF
#------------------------------------------------#
w = np.sqrt(KX**2 + KZ**2)
beta = n_av*w
q = np.sqrt(beta**2 - KX**2)
Q = q - beta  - .0001

main_part = (4*beta**2)/Q

for i in range(Fs):
   for j in range(Fs):
      if ((w[i,j]>=w_max) or (w[i,j]<=w_min)):
          main_part[i,j] = 0.0
          
main_part_nr = (4*beta**2)/(Q)
#------------------------------------------------#

# Spectrum
#------------------------------------------------#
ww = np.linspace(0,Fs-1,Fs)
w_min = 1.0
w_max = 20.0
w_0 = (w_max+w_min)/2.0
w_width = (w_max-w_min)/2.0
spectrum = np.exp(-((w-w_0)**2)/(2.0*w_width))

#------------------------------------------------#


# OTF
#------------------------------------------------#
OTF = (1.0/n_av**2)*main_part_nr*spectrum
PSF = np.fft.ifft2(OTF)

PSF_plot = np.fft.ifftshift(np.abs(PSF))

aux = np.fliplr(OTF[:,1:Fs/2+1])
OTF[:,Fs/2:] = aux
aux = OTF[1:Fs/2+1,:]
OTF[Fs/2:,:] = np.transpose(np.fliplr(np.transpose(aux)))

#------------------------------------------------#

# plotting procedure
#------------------------------------------------#
plt.subplot(211)
plt.imshow(OTF)
plt.colorbar()

plt.subplot(212)
plt.imshow(PSF_plot)
plt.colorbar()


pylab.show()
#------------------------------------------------#