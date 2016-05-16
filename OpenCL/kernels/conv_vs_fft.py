# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 21:39:38 2015

@author: good-cat
"""

# Imported libraries for math and plots
#------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import signal
#------------------------------------------------#
# sampling rate and time characteristics
#------------------------------------------------#
Fs = 100;
T = 1.0;
t = (T/Fs)*np.arange(Fs);
#t_aux = np.arange(199)
#------------------------------------------------#
gauss = np.exp(-(t-.5)**2/0.001)
x0 = np.sin(4*np.pi*t)#  + np.sin(12*np.pi*t); # the main signal

x = np.convolve(x0,gauss)

G = np.fft.fft(gauss)
X0 = np.fft.fft(x0)
Y = G*X0
y = np.fft.ifft(Y)

plt.subplot(411)
plt.xlabel('Time, sec')
plt.ylabel('Amplitude')
plt.plot(x,'r')

plt.subplot(412)
plt.xlabel('Time, sec')
plt.ylabel('Amplitude')
plt.plot(y,'r')


