# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 18:15:26 2015

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
T = 2.0
#------------------------------------------------#


# Meshgrid
#------------------------------------------------#
kx = (T/Fs)*np.linspace(0,Fs-1,Fs)
q = - (T/Fs)*np.linspace(0,Fs-1,Fs)
KX,Q = np.meshgrid(kx,q)


# Average refracted index
#------------------------------------------------#
n_av = 1.33
#------------------------------------------------#

w_min = 1.25/T
w_max = 2.5/T
w_0 = (w_max+w_min)/2.0
w_width = (w_max-w_min)/2.0

#Main part of the PSF
#------------------------------------------------#
beta = -(Q**2+KX**2)/(2*Q)

w = beta/n_av

#Q = q - beta  - .0001
NA = 1.0
KX_max = n_av*w_0*NA
Q_max = np.sqrt((n_av*w_0)**2-KX_max**2) - (n_av*w_0)**2


main_part = ((Q**2+KX**2)**2)/(Q**3)
main_part[0,0]=0.0
main_part_mod = main_part

for i in range(Fs):
   for j in range(Fs):
      if (main_part[i,j]==-np.inf):
          main_part_mod[i,j] = -10000.0



filter0 = np.ones((Fs,Fs))

for i in range(Fs):
   for j in range(Fs):
      if ((i>j)or(T*j/Fs > KX_max)or(T*i/Fs<Q_max)):
          filter0[i,j] = 0.0

gaussian = np.exp(-(KX**2+Q**2)/(5.0*T/Fs))
filt = np.fft.ifft2(np.fft.fft2(gaussian)*np.fft.fft2(filter0))
norm = 1.0/(filt.max())
filt = norm*filt



main_part_mod = main_part_mod*np.real(filt)
#------------------------------------------------#

# Spectrum
#------------------------------------------------#


spectrum = np.exp(-((beta/n_av-w_0)**2)/(2.0*w_width))

#------------------------------------------------#


# OTF
#------------------------------------------------#
res = (1.0/n_av**2)*main_part_mod*spectrum
res[0,0]=0.0


aux = np.fliplr(res)
#bottom = np.zeros((Fs,2*Fs))
OTF = np.concatenate((res,aux),axis=1)
#OTF = np.concatenate((interm,bottom))
#aux = OTF[1:Fs/2+1,:]
#OTF[Fs/2:,:] = np.transpose(np.fliplr(np.transpose(aux)))
PSF = np.fft.ifft2(OTF)



PSF_plot = np.fft.ifftshift(PSF)
phase = np.angle(PSF)
#------------------------------------------------#

# plotting procedure
#------------------------------------------------#
plt.subplot(211)
plt.imshow(OTF)
plt.colorbar()

plt.subplot(212)
plt.imshow(abs(PSF_plot))
plt.colorbar()


pylab.show()
#------------------------------------------------#