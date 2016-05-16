# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:01:01 2015

@author: good-cat
"""
###--------------------------------------------------------###
###-------------------Initializations----------------------###
###--------------------------------------------------------###

import numpy as np

def proj(size=19,angle=6.0):
    
###--------------------------------------------------------###
###-------------------Possible errors----------------------###
###--------------------------------------------------------### 
    
 
 
   
    if size%2==0:
        raise ValueError("Wrong patch size! Has to be 2n+1 x 2n+1")
    
###--------------------------------------------------------###
###----------------------Main part-------------------------###
###--------------------------------------------------------###    
    
#    ret = np.zeros(30)
    
    
#    for k in range(30):
#        angle = k*6.0
        
        # angle in radians
    a = angle*np.pi/180.0
    aa = abs(np.round(np.tan(a),5))
  
    N=size/2 + 1
    M = np.zeros((N,N))
  
    for i in range(N):
        for j in range(N):
              
              a1 = (1.0*(i+.5))/(1.0*(j+.5))
              a2 = (1.0*(i-.5))/(1.0*(j+.5))
              a3 = (1.0*(i+.5))/(1.0*(j-.5))
              a4 = (1.0*(i-.5))/(1.0*(j-.5))
              
              if (i == 0):
                  a2 = 0
                  a4 = 0
              elif (j == 0):
                  a3 = np.inf
                  a4 = np.inf
              if ((aa>= min(a1,a2,a3,a4)) and (aa< max(a1,a2,a3,a4))):
                  M[i,j] = 1
      
    M[0,0] = 1
    m = np.fliplr(np.flipud(M))
  
    out = np.zeros((2*N-1,2*N-1))
    out[N-1:,N-1:] = M
    out[:N,:N] = m
  
    if (angle>=90)and(angle<180):
           out = np.flipud(out)
            
        
        
    
    # return the array of features to the convolution procedure ---> !(array of 6 numbers)!
    return out
    
###--------------------------------------------------------###
###----------------------Testing part----------------------###
###--------------------------------------------------------###
    
def test_proj():

    z = proj(21)  
    assert (type(z) == np.ndarray)
    assert (z.ndim == 2)
    assert (z.shape[0] == 21)

if __name__ == "__main__":
       
    test_proj()
