# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:12:46 2015

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
Fs = 600 # sampling frequency
T = 100.0
L = 18000.0
#------------------------------------------------#
# Average refracted index
#------------------------------------------------#
n_av = 1.3392237097317652

# Meshgrid
#------------------------------------------------#
kx = (2*np.pi/(L))*np.linspace(0,Fs-1,Fs)
q = - (2*np.pi/(L))*np.linspace(0,Fs-1,Fs)
KX,Q = np.meshgrid(kx,q)




#------------------------------------------------#

w_min = 2.0*np.pi/800.0
w_max = 2.0*np.pi/400.0
w_0 = (w_max+w_min)/2.0
w_width = (w_max-w_min)/2.0

#Main part of the PSF
#------------------------------------------------#
beta = -(Q**2+KX**2)/(2*Q)

w = beta/n_av

#Q = q - beta  - .0001
NA = 1.4
KX_max = n_av*w_0*NA
Q_max = np.sqrt((n_av*w_0)**2-KX_max**2) - (n_av*w_0)


main_part = ((Q**2+KX**2)**2)/(Q**3)
main_part[0,0]=0.0
main_part_mod = main_part

for i in range(Fs):
   for j in range(Fs):
      if (main_part[i,j]==-np.inf):
          main_part_mod[i,j] = -1.0



filter0 = np.ones((Fs,Fs))

for i in range(Fs):
   for j in range(Fs):
      if ((i>j)or(2*j*np.pi/L > KX_max)or(-2*i*np.pi/L<Q_max)):
          filter0[i,j] = 0.0

gaussian = np.exp(-((KX-kx[Fs/2])**2+(Q-q[Fs/2])**2)/(2.0*(10.0*KX_max/100.0)**2))
filt = np.fft.ifft2(np.fft.fft2(gaussian)*np.fft.fft2(filter0))
norm = 1.0/(filt.max())
filt = norm*np.fft.fftshift(filt)
#filt = filter0



main_part_mod = main_part_mod*np.real(filter0)
#------------------------------------------------#

# Spectrum
#------------------------------------------------#


spectrum = np.exp(-((beta/n_av-w_0)**2)/(2.0*(w_width)**2))

#------------------------------------------------#


# OTF
#------------------------------------------------#
res = (1.0/n_av**2)*main_part_mod*spectrum

res[0,0]=0.0
R = np.fft.fft2(res)
FF = np.fft.fft2(gaussian)
rr = np.fft.ifftshift(np.fft.ifft2(FF*R))
#OTF=res
aux = np.fliplr(rr)
#bottom = np.zeros((Fs,2*Fs))
OTF = np.concatenate((rr[:,:Fs/2+1],aux[:,Fs/2:-1]),axis=1)
#bottom = np.transpose(np.fliplr(np.transpose(top)))
#OTF = np.concatenate((top,bottom))
#OTF = np.concatenate((interm,bottom))
#aux = OTF[1:Fs/2+1,:]
#OTF[Fs/2:,:] = np.transpose(np.fliplr(np.transpose(aux)))
aux1 = np.fft.fftshift(OTF)
PSF = np.fft.ifft2(aux1)



PSF_plot = np.fft.ifftshift(PSF)
phase = np.angle(PSF)
#------------------------------------------------#

# plotting procedure
#------------------------------------------------#
plt.subplot(311)
plt.imshow(abs(OTF))
plt.colorbar()

plt.subplot(312)
plt.imshow(np.abs(PSF_plot))
plt.colorbar()

plt.subplot(313)
plt.imshow(spectrum)
plt.colorbar()

pylab.show()
#------------------------------------------------#