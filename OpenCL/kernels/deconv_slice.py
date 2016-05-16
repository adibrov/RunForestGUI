# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:48:38 2015

@author: good-cat
"""


# Imported libraries for math and plots
#------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import pylab

def deconv_slice(data,rho):


    #------------------------------------------------#
    # Constants
    #------------------------------------------------#
    Fs = data.shape[1] # sampling frequency
    Fs1 = data.shape[0] # slice width
   # delta = 10  
    T = 100.0
    L = 18000.0
    #------------------------------------------------#
    
    
    # Meshgrid
    #------------------------------------------------#
    kx = (2*np.pi/L)*np.linspace(0,Fs-1,Fs)
    q = - (2*np.pi/L)*np.linspace(0,Fs-1,Fs)
    KX,Q = np.meshgrid(kx,q)
    
    
    # Average refracted index
    #------------------------------------------------#
    n_av = 1.3392237097317652
    
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
    NA = 1.0
    KX_max = n_av*w_0*NA
    Q_max = np.sqrt((n_av*w_0)**2-KX_max**2) - (n_av*w_0)
    
    
    main_part = ((Q**2+KX**2)**2)/(Q**3)
    main_part[0,0]=0.0
    main_part_mod = main_part
    
    for i in range(Fs):
       for j in range(Fs):
          if (main_part[i,j]==-np.inf):
              main_part_mod[i,j] = -100000.0
    
    
    
    
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
    
  
    #--------------------------------------------------------------------------------#
    OTF_SL = OTF[0:Fs1/2,:]
    H = np.concatenate((OTF_SL,np.zeros((Fs1/2,600))),axis=0)
    #h = PSF
   # refr = np.loadtxt('/home/good-cat/Documents/SLIT/python/refr_ind.txt')
   # rrr = np.zeros((2400,600))
   # rrr_dist = np.zeros((2400,600))
   # x = np.linspace(0,599,600)
   # y = np.linspace(0,599,600)
   # X,Y = np.meshgrid(x,y)
   # g = np.exp(-0.5*((X-300)**2+(Y-300)**2)/(10**2))
   # G = np.fft.fft2(g)
   # R = np.fft.fft2(refr)
    
    #re = (1.0/g.sum())*np.fft.ifftshift(np.fft.ifft2(G*R))
    # definition of signals, forward and inverse transforms
    #------------------------------------------------#
    noise = 0.1*np.random.rand(Fs1,Fs) - 0.05
    
    #s = np.sin(2*np.pi*X/Tx)*np.sin(2*np.pi*Y/Ty)
    #s = misc.lena()[200:300,200:300]
    s = data
    S = np.fft.fft2(s)
    S[Fs1/2:,:] = np.zeros((Fs1/2,data.shape[1]))
    
    
    #y = np.convolve(q,x);
    #y = y0[50:150]+ noise
    
    F = S*H;
    
    f0 = np.fft.ifft2(F) 
    f = f0 + noise
    #rrr_dist[40*i:40*i+40,:] = f
    F = np.fft.fft2(f)
    
    #PP_noise = np.fft.fft2(noise)
    
    #P_noise = (abs(PP_noise)*abs(PP_noise))#/10000.0
    
    #y = signal.fftconvolve(h,x,mode="valid") ;
    
    # regularization funcitonal
    Dx1 = np.zeros((Fs1,Fs)) + 1j*np.zeros((Fs1,Fs))
    Dy1 = np.zeros((Fs1,Fs)) + 1j*np.zeros((Fs1,Fs))
    Dx2 = np.zeros((Fs1,Fs)) + 1j*np.zeros((Fs1,Fs))
    Dy2 = np.zeros((Fs1,Fs)) + 1j*np.zeros((Fs1,Fs))
    Dcross1 = np.zeros((Fs1,Fs)) + 1j*np.zeros((Fs1,Fs))
    Dcross2 = np.zeros((Fs1,Fs)) + 1j*np.zeros((Fs1,Fs))
    
    for k in range(Fs1):
        for m in range(Fs):
            Dx1[k,m] = np.exp(-2.0j*np.pi*k/L) - 1.0
            Dy1[k,m] = np.exp(-2.0j*np.pi*m/L) - 1.0
            Dx2[k,m] = 2.0*(1.0 - np.cos(-2.0j*np.pi*k/L))
            Dy2[k,m] = 2.0*(1.0 - np.cos(-2.0j*np.pi*m/L))
            Dcross1[k,m] = (np.exp(-2.0j*np.pi*k/L) - np.exp(-2.0j*np.pi*m/L))
            Dcross2[k,m] = np.exp(-2.0j*np.pi*m/L)*np.exp(-2.0j*np.pi*k/L) - 1.0
    
    
    Ax1 = np.ones((Fs1,Fs))
    Ay1 = np.ones((Fs1,Fs))
    Ax2 = np.ones((Fs1,Fs))
    Ay2 = np.ones((Fs1,Fs))
    Across1 = np.ones((Fs1,Fs))
    Across2 = np.ones((Fs1,Fs))
    
    # calculate the power spectrum
    #------------------------------------------------#
    #f, P = signal.periodogram(y, Fs, return_onesided=False)
    #f, P_noise = signal.periodogram(noise, Fs, return_onesided=False)
    #------------------------------------------------#
    
    
    
    diff = 1
    buff = np.zeros((Fs1,Fs))
    count = 0;
    
    while (diff>=.0001):
        up = np.conj(H)
        down = .00000001 + abs(H)*abs(H) + rho*(np.conj(Dx1)*Ax1*Dx1 + np.conj(Dy1)*Ay1*Dy1 + np.conj(Dx2)*Ax2*Dx2 + np.conj(Dy2)*Ay2*Dy2 +  np.conj(Dcross1)*Across1*Dcross1 + np.conj(Dcross2)*Across2*Dcross2)
            
        G = np.divide(up,down)
            #------------------------------------------------#
            
            # deconvolution
            #------------------------------------------------#
    #    g = np.fft.ifft2(G)
            #y = np.fft.ifft(Y[0:len(H)])
            #X = np.multiply(G,Y);
        X_rec = G*F#F
        x_rec = np.fft.ifft2(X_rec);
        diff = (x_rec - buff).max()
        Ax1 = 1.0/(abs(Dx1*X_rec)**2 + 10**(-9))
        Ay1 = 1.0/(abs(Dy1*X_rec)**2 + 10**(-9))
        Ax2 = 1.0/(abs(Dx2*X_rec)**2 + 10**(-9))
        Ay2 = 1.0/(abs(Dy2*X_rec)**2 + 10**(-9))
        Across1 = 1.0/(abs(Dcross1*X_rec)**2 + 10**(-9))
        Across2 = 1.0/(abs(Dcross2*X_rec)**2 + 10**(-9))
        
        buff = x_rec
        count = count+1

    return np.real(f), np.real(x_rec)

