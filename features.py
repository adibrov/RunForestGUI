# ----------------- Extracting features from image patches ---------------#

# ---- --- --- Changes --- --- --- #
# - additional features from applying 4 filters to blurred images (Laplacian, Sobel, Structure, Hessian)

# --- path for the modules --- #
import sys
sys.path.append('./OpenCL/kernels')
sys.path.append('./OpenCL/')
print sys.path

# --- GPU-based modules --- #

import gputools

from convolve import convolve
from gpu_bilateral import gpu_bilateral
from gpu_entropy import gpu_entropy
from gpu_hessian import gpu_hessian
from gpu_kuwahara import gpu_kuwahara
from gpu_maximum import gpu_maximum
from gpu_median import gpu_median
from gpu_minimum import gpu_minimum
from gpu_structure import gpu_structure
from gpu_variance import gpu_variance
# --- python kernels ---#
from Gaussian import gauss
from mean import Mean
from membrane_projections import proj
from Laplacian import laplacian
from Gabor import gabor
from sobel import sobel
#--- numpy ---#
import numpy as np
import time


def feat_dummy(data):
    
    """ Returns an array of features for a given patch. Standard dimensions: patch_size_0 x patch_size_1 x # of kernels used. """


    names = 284*["fiii"]

    feat = np.stack([gpu_kuwahara(data, N=5) for i in range(284)])

    return feat, names 

# --- function  for feature extraction---#
def feat(data):
    
    """ Returns an array of features for a given patch. Standard dimensions: patch_size_0 x patch_size_1 x # of kernels used. """


    fi = []
    names = []

    t0 = time.time()
  #  print 'entered feat'
    # gaussian 57 - 68

    fi.append(data)
    names.append('initial picture: intesity values')
    g05 = convolve(data,gauss(sigma_x = 0.5, sigma_y = 0.5))
    g1 = convolve(data,gauss(sigma_x = 1.0, sigma_y = 1.0))
    g2 = convolve(data,gauss(sigma_x = 2.0, sigma_y = 2.0))
    g3 = convolve(data,gauss(sigma_x = 3.0, sigma_y = 3.0))
    g4 = convolve(data,gauss(sigma_x = 4.0, sigma_y = 4.0))
    g5 = convolve(data,gauss(sigma_x = 5.0, sigma_y = 5.0))
    g6 = convolve(data,gauss(sigma_x = 6.0, sigma_y = 6.0))
    g7 = convolve(data,gauss(sigma_x = 7.0, sigma_y = 7.0))
    g10 = convolve(data,gauss(sigma_x = 10.0, sigma_y = 10.0))
    g12 = convolve(data,gauss(sigma_x = 12.0, sigma_y = 12.0))
    g14 = convolve(data,gauss(sigma_x = 14.0, sigma_y = 14.0))
    g16 = convolve(data,gauss(sigma_x = 16.0, sigma_y = 16.0))

    
    fi.append(g05)
    names.append('gauss(sigma_x = 0.5, sigma_y = 0.5))')
    fi.append(g1)
    names.append('gauss(sigma_x = 1., sigma_y = 1.))')
    fi.append(g2)
    names.append('gauss(sigma_x = 2., sigma_y = 2.))')
    fi.append(g3)
    names.append('gauss(sigma_x = 3., sigma_y = 3.)')
    fi.append(g4)
    names.append('gauss(sigma_x = 4., sigma_y = 4.)')
    fi.append(g5)
    names.append('gauss(sigma_x = 5., sigma_y = 5.)')
    fi.append(g6)
    names.append('gauss(sigma_x = 6., sigma_y = 6.)')
    fi.append(g7)
    names.append('gauss(sigma_x = 7., sigma_y = 7.)')
    fi.append(g10)
    names.append('gauss(sigma_x = 10., sigma_y = 10.)')
    fi.append(g12)
    names.append('gauss(sigma_x = 12., sigma_y = 12.)')
    fi.append(g14)
    names.append('gauss(sigma_x = 14., sigma_y = 14.)')
    fi.append(g16)
    names.append('gauss(sigma_x = 16., sigma_y = 16.)')

    tg = time.time()
   # print 'performed gaussian blurs - '+str(tg-t0)

    # bilateral 0 - 15

    
    fi.append(gpu_bilateral(data, Nx=10, Ny=10, sigma_int=1.0, sigma_dist=1.0))
    names.append('gpu_bilateral(data, Nx=10, Ny=10, sigma_int=1.0, sigma_dist=1.0)')   
    fi.append(gpu_bilateral(data, Nx=10, Ny=10, sigma_int=5.0, sigma_dist=1.0))
    names.append('gpu_bilateral(data, Nx=10, Ny=10, sigma_int=5.0, sigma_dist=1.0)')
    fi.append(gpu_bilateral(data, Nx=10, Ny=10, sigma_int=10.0, sigma_dist=1.0))
    names.append('gpu_bilateral(data, Nx=10, Ny=10, sigma_int=10.0, sigma_dist=1.0)')
    # fi.append(gpu_bilateral(data, Nx=20, Ny=20, sigma_int=2.0, sigma_dist=5.0))
    # names.append('gpu_bilateral(data, Nx=20, Ny=20, sigma_int=2.0, sigma_dist=5.0)')
    # fi.append(gpu_bilateral(data, Nx=20, Ny=20, sigma_int=5.0, sigma_dist=5.0))
    # names.append('gpu_bilateral(data, Nx=20, Ny=20, sigma_int=5.0, sigma_dist=5.0)')
    # fi.append(gpu_bilateral(data, Nx=20, Ny=20, sigma_int=7.0, sigma_dist=5.0))
    # names.append('gpu_bilateral(data, Nx=20, Ny=20, sigma_int=7.0, sigma_dist=5.0)')
    # fi.append(gpu_bilateral(data, Nx=20, Ny=20, sigma_int=10.0, sigma_dist=5.0))
    # names.append('gpu_bilateral(data, Nx=20, Ny=20, sigma_int=10.0, sigma_dist=5.0)')
    # fi.append(gpu_bilateral(data, Nx=20, Ny=20, sigma_int=2.0, sigma_dist=7.0))
    # names.append('gpu_bilateral(data, Nx=20, Ny=20, sigma_int=2.0, sigma_dist=7.0)')
    # fi.append(gpu_bilateral(data, Nx=20, Ny=20, sigma_int=5.0, sigma_dist=7.0))
    # names.append('gpu_bilateral(data, Nx=20, Ny=20, sigma_int=5.0, sigma_dist=7.0)')
    # fi.append(gpu_bilateral(data, Nx=20, Ny=20, sigma_int=7.0, sigma_dist=7.0))
    # names.append('gpu_bilateral(data, Nx=20, Ny=20, sigma_int=7.0, sigma_dist=7.0)')
    # fi.append(gpu_bilateral(data, Nx=20, Ny=20, sigma_int=10.0, sigma_dist=7.0))
    # names.append('gpu_bilateral(data, Nx=20, Ny=20, sigma_int=10.0, sigma_dist=7.0)')
    # fi.append(gpu_bilateral(data, Nx=50, Ny=50, sigma_int=10.0, sigma_dist=10.0))
    # names.append('gpu_bilateral(data, Nx=50, Ny=50, sigma_int=10.0, sigma_dist=10.0)')
    # fi.append(gpu_bilateral(data, Nx=50, Ny=50, sigma_int=10.0, sigma_dist=20.0))
    # names.append('gpu_bilateral(data, Nx=50, Ny=50, sigma_int=10.0, sigma_dist=20.0)')
    # fi.append(gpu_bilateral(data, Nx=50, Ny=50, sigma_int=10.0, sigma_dist=30.0))
    # names.append('gpu_bilateral(data, Nx=50, Ny=50, sigma_int=10.0, sigma_dist=30.0)')
    # fi.append(gpu_bilateral(data, Nx=50, Ny=50, sigma_int=20.0, sigma_dist=10.0))
    # names.append('gpu_bilateral(data, Nx=50, Ny=50, sigma_int=20.0, sigma_dist=10.0)')
    # fi.append(gpu_bilateral(data, Nx=50, Ny=50, sigma_int=30.0, sigma_dist=10.0))
    # names.append('gpu_bilateral(data, Nx=50, Ny=50, sigma_int=30.0, sigma_dist=10.0)')

# entropy    16 - 20
    fi.append(gpu_entropy(data, Nx=2,Ny=2))
    names.append('gpu_entropy(data, Nx=2,Ny=2)')
    fi.append(gpu_entropy(data, Nx=3,Ny=3))
    names.append('gpu_entropy(data, Nx=2,Ny=2)')
    fi.append(gpu_entropy(data, Nx=5,Ny=4))
    names.append('gpu_entropy(data, Nx=2,Ny=2)')
    fi.append(gpu_entropy(data, Nx=5,Ny=5))
    names.append('gpu_entropy(data, Nx=2,Ny=2)')
    fi.append(gpu_entropy(data, Nx=7,Ny=7))
    names.append('gpu_entropy(data, Nx=2,Ny=2)')

    # hessian  21 - 28
    H = gpu_hessian(data)
    fi.append(H[:,:,0])
    names.append('H[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H[:,:,7]')

    H = gpu_hessian(g05)
    fi.append(H[:,:,0])
    names.append('H05[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H05[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H05[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H05[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H05[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H05[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H05[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H05[:,:,7]')

    H = gpu_hessian(g1)
    fi.append(H[:,:,0])
    names.append('H1[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H1[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H1[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H1[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H1[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H1[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H1[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H1[:,:,7]')

    H = gpu_hessian(g2)
    fi.append(H[:,:,0])
    names.append('H2[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H2[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H2[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H2[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H2[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H2[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H2[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H2[:,:,7]')

    H = gpu_hessian(g3)
    fi.append(H[:,:,0])
    names.append('H3[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H3[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H3[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H3[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H3[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H3[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H3[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H3[:,:,7]')

    H = gpu_hessian(g4)
    fi.append(H[:,:,0])
    names.append('H4[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H4[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H4[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H4[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H4[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H4[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H4[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H4[:,:,7]')

    H = gpu_hessian(g5)
    fi.append(H[:,:,0])
    names.append('H5[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H5[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H5[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H5[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H5[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H5[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H5[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H5[:,:,7]')

    H = gpu_hessian(g6)
    fi.append(H[:,:,0])
    names.append('H6[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H6[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H6[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H6[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H6[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H6[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H6[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H6[:,:,7]')

    H = gpu_hessian(g7)
    fi.append(H[:,:,0])
    names.append('H7[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H7[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H7[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H7[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H7[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H7[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H7[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H7[:,:,7]')

    H = gpu_hessian(g10)
    fi.append(H[:,:,0])
    names.append('H10[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H10[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H10[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H10[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H10[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H10[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H10[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H10[:,:,7]')

    H = gpu_hessian(g12)
    fi.append(H[:,:,0])
    names.append('H12[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H12[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H12[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H12[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H12[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H12[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H12[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H12[:,:,7]')

    H = gpu_hessian(g14)
    fi.append(H[:,:,0])
    names.append('H14[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H14[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H14[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H14[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H14[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H14[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H14[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H14[:,:,7]')

    H = gpu_hessian(g16)
    fi.append(H[:,:,0])
    names.append('H16[:,:,0]')
    fi.append(H[:,:,1])
    names.append('H16[:,:,1]')
    fi.append(H[:,:,2])
    names.append('H16[:,:,2]')
    fi.append(H[:,:,3])
    names.append('H16[:,:,3]')
    fi.append(H[:,:,4])
    names.append('H16[:,:,4]')
    fi.append(H[:,:,5])
    names.append('H16[:,:,5]')
    fi.append(H[:,:,6])
    names.append('H16[:,:,6]')
    fi.append(H[:,:,7])
    names.append('H16[:,:,7]')

# kuwahara 29 - 33
    fi.append(gpu_kuwahara(data, N=3))
    names.append('gpu_kuwahara(data, N=3)')
    fi.append(gpu_kuwahara(data, N=5))
    names.append('gpu_kuwahara(data, N=5)')
    fi.append(gpu_kuwahara(data, N=7))
    names.append('gpu_kuwahara(data, N=7)')
    fi.append(gpu_kuwahara(data, N=9))
    names.append('gpu_kuwahara(data, N=9)')
    fi.append(gpu_kuwahara(data, N=11))
    names.append('gpu_kuwahara(data, N=11)')

# maximum    34 - 38
    fi.append(gpu_maximum(data, Nx=2, Ny=2))
    names.append('gpu_maximum(data, Nx=2, Ny=2)')
    fi.append(gpu_maximum(data, Nx=3, Ny=3))
    names.append('gpu_maximum(data, Nx=3, Ny=3)')
    fi.append(gpu_maximum(data, Nx=4, Ny=4))
    names.append('gpu_maximum(data, Nx=4, Ny=4)')
    fi.append(gpu_maximum(data, Nx=5, Ny=5))
    names.append('gpu_maximum(data, Nx=5, Ny=5)')
    fi.append(gpu_maximum(data, Nx=7, Ny=7))
    names.append('gpu_maximum(data, Nx=7, Ny=7)')

# median    39 - 43
    
    fi.append(gpu_median(data, Nx=2, Ny=2))
    names.append('gpu_median(data, Nx=2, Ny=2)')
    fi.append(gpu_median(data, Nx=3, Ny=3))
    names.append('gpu_median(data, Nx=3, Ny=3)')
    fi.append(gpu_median(data, Nx=4, Ny=4))
    names.append('gpu_median(data, Nx=4, Ny=4)')
    fi.append(gpu_median(data, Nx=5, Ny=5))
    names.append('gpu_median(data, Nx=5, Ny=5)')
    fi.append(gpu_median(data, Nx=7, Ny=7))
    names.append('gpu_median(data, Nx=7, Ny=7)')

    # minimum 44 - 48

    fi.append(gpu_minimum(data, Nx=2, Ny=2))
    names.append('gpu_minimum(data, Nx=2, Ny=2)')
    fi.append(gpu_minimum(data, Nx=3, Ny=3))
    names.append('(gpu_minimum(data, Nx=3, Ny=3)')
    fi.append(gpu_minimum(data, Nx=4, Ny=4))
    names.append('gpu_minimum(data, Nx=4, Ny=4)')
    fi.append(gpu_minimum(data, Nx=5, Ny=5))
    names.append('gpu_minimum(data, Nx=5, Ny=5)')
    fi.append(gpu_minimum(data, Nx=7, Ny=7))
    names.append('gpu_minimum(data, Nx=7, Ny=7)')

    # structure 49 - 50
    S = gpu_structure(data)
    fi.append(S[:,:,0])
    names.append('S[:,:,0]')
    fi.append(S[:,:,1])
    names.append('S[:,:,1]')
    
    S_g05 = gpu_structure(g05)
    fi.append(S_g05[:,:,0])
    names.append('S_g05[:,:,0]')
    fi.append(S_g05[:,:,1])
    names.append('S_g05[:,:,1]')

    S_g1 = gpu_structure(g1)
    fi.append(S_g1[:,:,0])
    names.append('S_g1[:,:,0]')
    fi.append(S_g1[:,:,1])
    names.append('S_g1[:,:,1]')

    S_g2 = gpu_structure(g2)
    fi.append(S_g2[:,:,0])
    names.append('S_g2[:,:,0]')
    fi.append(S_g2[:,:,1])
    names.append('S_g2[:,:,1]')

    S_g3 = gpu_structure(g3)
    fi.append(S_g3[:,:,0])
    names.append('S_g3[:,:,0]')
    fi.append(S_g3[:,:,1])
    names.append('S_g3[:,:,1]')

    S_g4 = gpu_structure(g4)
    fi.append(S_g4[:,:,0])
    names.append('S_g4[:,:,0]')
    fi.append(S_g4[:,:,1])
    names.append('S_g4[:,:,1]')

    S_g5 = gpu_structure(g5)
    fi.append(S_g5[:,:,0])
    names.append('S_g5[:,:,0]')
    fi.append(S_g5[:,:,1])
    names.append('S_g5[:,:,1]')

    S_g6 = gpu_structure(g6)
    fi.append(S_g6[:,:,0])
    names.append('S_g6[:,:,0]')
    fi.append(S_g6[:,:,1])
    names.append('S_g6[:,:,1]')

    S_g7 = gpu_structure(g7)
    fi.append(S_g7[:,:,0])
    names.append('S_g7[:,:,0]')
    fi.append(S_g7[:,:,1])
    names.append('S_g7[:,:,1]')

    S_g10 = gpu_structure(g10)
    fi.append(S_g10[:,:,0])
    names.append('S_g10[:,:,0]')
    fi.append(S_g10[:,:,1])
    names.append('S_g10[:,:,1]')

    S_g12 = gpu_structure(g12)
    fi.append(S_g12[:,:,0])
    names.append('S_g12[:,:,0]')
    fi.append(S_g12[:,:,1])
    names.append('S_g12[:,:,1]')

    S_g14 = gpu_structure(g14)
    fi.append(S_g14[:,:,0])
    names.append('S_g14[:,:,0]')
    fi.append(S_g14[:,:,1])
    names.append('S_g14[:,:,1]')

    S_g16 = gpu_structure(g16)
    fi.append(S_g16[:,:,0])
    names.append('S_g16[:,:,0]')
    fi.append(S_g16[:,:,1])
    names.append('S_g16[:,:,1]')

    
    # variance 51 - 56
    fi.append(gpu_variance(data, Nx=2, Ny=2))
    names.append('gpu_variance(data, Nx=2, Ny=2)')
    fi.append(gpu_variance(data, Nx=3, Ny=3))
    names.append('gpu_variance(data, Nx=3, Ny=3)')
    fi.append(gpu_variance(data, Nx=4, Ny=4))
    names.append('gpu_variance(data, Nx=4, Ny=4)')
    fi.append(gpu_variance(data, Nx=5, Ny=5))
    names.append('gpu_variance(data, Nx=5, Ny=5)')
    fi.append(gpu_variance(data, Nx=6, Ny=6))
    names.append('gpu_variance(data, Nx=6, Ny=6)')
    fi.append(gpu_variance(data, Nx=7, Ny=7))
    names.append('gpu_variance(data, Nx=7, Ny=7)')
    


    # difference of gaussians 69 - 83

    fi.append(convolve(data,gauss(sigma_x = 0.5, sigma_y = 0.5) - gauss()))
    names.append('gauss(sigma_x = 0.5, sigma_y = 0.5) - gauss()')
    fi.append(convolve(data,gauss(sigma_x = 1., sigma_y = 1.) - gauss()))
    names.append('gauss(sigma_x = 1., sigma_y = 1.) - gauss()')
    fi.append(convolve(data,gauss(sigma_x = 2., sigma_y = 2.) - gauss()))
    names.append('gauss(sigma_x = 2., sigma_y = 2.) - gauss()')
    fi.append(convolve(data,gauss(sigma_x = 4., sigma_y = 4.) - gauss(sigma_x = 2., sigma_y = 2.)))
    names.append('gauss(sigma_x = 4., sigma_y = 4.) - gauss(sigma_x = 2., sigma_y = 2.)')
    fi.append(convolve(data,gauss(sigma_x = 6., sigma_y = 6.) - gauss(sigma_x = 2., sigma_y = 2.)))
    names.append('gauss(sigma_x = 6., sigma_y = 6.) - gauss(sigma_x = 2., sigma_y = 2.)')
    fi.append(convolve(data,gauss(sigma_x = 8., sigma_y = 8.) - gauss(sigma_x = 2., sigma_y = 2.)))
    names.append('gauss(sigma_x = 8., sigma_y = 8.) - gauss(sigma_x = 2., sigma_y = 2.)')
    fi.append(convolve(data,gauss(sigma_x = 10., sigma_y = 10.) - gauss(sigma_x = 2., sigma_y = 2.)))
    names.append('gauss(sigma_x = 10., sigma_y = 10.) - gauss(sigma_x = 2., sigma_y = 2.)')
    fi.append(convolve(data,gauss(sigma_x = 6., sigma_y = 6.) - gauss(sigma_x = 4., sigma_y = 4.)))
    names.append('gauss(sigma_x = 6., sigma_y = 6.) - gauss(sigma_x = 4., sigma_y = 4.)')
    fi.append(convolve(data,gauss(sigma_x = 8., sigma_y = 8.) - gauss(sigma_x = 4., sigma_y = 4.)))
    names.append('gauss(sigma_x = 8., sigma_y = 8.) - gauss(sigma_x = 4., sigma_y = 4.)')
    fi.append(convolve(data,gauss(sigma_x = 10., sigma_y = 10.) - gauss(sigma_x = 4., sigma_y = 4.)))
    names.append('gauss(sigma_x = 10., sigma_y = 10.) - gauss(sigma_x = 4., sigma_y = 4.)')
    fi.append(convolve(data,gauss(sigma_x = 12., sigma_y = 12.) - gauss(sigma_x = 4., sigma_y = 4.)))
    names.append('gauss(sigma_x = 12., sigma_y = 12.) - gauss(sigma_x = 4., sigma_y = 4.)')
    fi.append(convolve(data,gauss(sigma_x = 14., sigma_y = 14.) - gauss(sigma_x = 4., sigma_y = 4.)))
    names.append('gauss(sigma_x = 14., sigma_y = 14.) - gauss(sigma_x = 4., sigma_y = 4.)')
    fi.append(convolve(data,gauss(sigma_x = 10., sigma_y = 10.) - gauss(sigma_x = 6., sigma_y = 6.)))
    names.append('gauss(sigma_x = 10., sigma_y = 10.) - gauss(sigma_x = 6., sigma_y = 6.)')
    fi.append(convolve(data,gauss(sigma_x = 12., sigma_y = 12.) - gauss(sigma_x = 6., sigma_y = 6.)))
    names.append('gauss(sigma_x = 12., sigma_y = 12.) - gauss(sigma_x = 6., sigma_y = 6.)')
    fi.append(convolve(data,gauss(sigma_x = 14., sigma_y = 14.) - gauss(sigma_x = 6., sigma_y = 6.)))
    names.append('gauss(sigma_x = 14., sigma_y = 14.) - gauss(sigma_x = 6., sigma_y = 6.)')
 
    # mean 84 - 89
    
    fi.append(convolve(data,Mean(Nx=2, Ny=2)))
    names.append('Mean(Nx=2, Ny=2)')
    fi.append(convolve(data,Mean(Nx=3, Ny=3)))
    names.append('Mean(Nx=3, Ny=3)')
    fi.append(convolve(data,Mean(Nx=4, Ny=4)))
    names.append('Mean(Nx=4, Ny=4)')
    fi.append(convolve(data,Mean(Nx=5, Ny=5)))
    names.append('Mean(Nx=5, Ny=5)')
    fi.append(convolve(data,Mean(Nx=6, Ny=6)))
    names.append('Mean(Nx=6, Ny=6)')
    fi.append(convolve(data,Mean(Nx=7, Ny=7)))
    names.append('Mean(Nx=7, Ny=7)')

    # laplacian 90
    
    fi.append(convolve(data,laplacian())) 
    names.append('laplacian')

    fi.append(convolve(g05,laplacian())) 
    names.append('laplacian_g05')

    fi.append(convolve(g1,laplacian())) 
    names.append('laplacian_g1')

    fi.append(convolve(g2,laplacian())) 
    names.append('laplacian_g2')

    fi.append(convolve(g3,laplacian())) 
    names.append('laplacian_g3')

    fi.append(convolve(g4,laplacian())) 
    names.append('laplacian_g4')

    fi.append(convolve(g5,laplacian())) 
    names.append('laplacian_g5')

    fi.append(convolve(g6,laplacian())) 
    names.append('laplacian_g6')

    fi.append(convolve(g7,laplacian())) 
    names.append('laplacian_g7')

    fi.append(convolve(g10,laplacian())) 
    names.append('laplacian_g10')

    fi.append(convolve(g12,laplacian())) 
    names.append('laplacian_g12')

    fi.append(convolve(g14,laplacian())) 
    names.append('laplacian_g14')

    fi.append(convolve(g16,laplacian())) 
    names.append('laplacian_g16')
    # gabor 91 - 198
    
    t = np.linspace(0,2.0*np.pi,18)
    KX =np. cos(t)
    KY = np.sin(t)

    for i in range(KX.size):
       fi.append(convolve(data,gabor(kx=KX[i], ky=KY[i], sigma_x=2., sigma_y=2.)))
       names.append('gabor(kx=KX'+str(i)+'], ky=KY['+str(i)+'], sigma_x=2., sigma_y=2.)')
    for i in range(KX.size):
       fi.append(convolve(data,gabor(kx=KX[i], ky=KY[i], sigma_x=2., sigma_y=2., P=0.1)))
       names.append('gabor(kx=KX'+str(i)+'], ky=KY['+str(i)+'], sigma_x=2., sigma_y=2., P=0.1)')
    for i in range(KX.size):
       fi.append(convolve(data,gabor(kx=KX[i], ky=KY[i], sigma_x=2., sigma_y=2., P=0.5)))
       names.append('gabor(kx=KX'+str(i)+'], ky=KY['+str(i)+'], sigma_x=2., sigma_y=2., P=0.5)')
   
   
    
    # sobel 199
    
    fi.append(np.sqrt((convolve(data,sobel()[0]))**2 + (convolve(data,sobel()[1]))**2))
    names.append('sobel')

    fi.append(np.sqrt((convolve(g05,sobel()[0]))**2 + (convolve(g05,sobel()[1]))**2))
    names.append('sobel_g05')

    fi.append(np.sqrt((convolve(g1,sobel()[0]))**2 + (convolve(g1,sobel()[1]))**2))
    names.append('sobel_g1')

    fi.append(np.sqrt((convolve(g2,sobel()[0]))**2 + (convolve(g2,sobel()[1]))**2))
    names.append('sobel_g2')

    fi.append(np.sqrt((convolve(g3,sobel()[0]))**2 + (convolve(g3,sobel()[1]))**2))
    names.append('sobel_g3')

    fi.append(np.sqrt((convolve(g4,sobel()[0]))**2 + (convolve(g4,sobel()[1]))**2))
    names.append('sobel_g4')

    fi.append(np.sqrt((convolve(g5,sobel()[0]))**2 + (convolve(g5,sobel()[1]))**2))
    names.append('sobel_g5')

    fi.append(np.sqrt((convolve(g6,sobel()[0]))**2 + (convolve(g6,sobel()[1]))**2))
    names.append('sobel_g6')

    fi.append(np.sqrt((convolve(g7,sobel()[0]))**2 + (convolve(g7,sobel()[1]))**2))
    names.append('sobel_g7')

    fi.append(np.sqrt((convolve(g10,sobel()[0]))**2 + (convolve(g10,sobel()[1]))**2))
    names.append('sobel_g10')

    fi.append(np.sqrt((convolve(g12,sobel()[0]))**2 + (convolve(g12,sobel()[1]))**2))
    names.append('sobel_g12')

    fi.append(np.sqrt((convolve(g14,sobel()[0]))**2 + (convolve(g14,sobel()[1]))**2))
    names.append('sobel_g14')

    fi.append(np.sqrt((convolve(g16,sobel()[0]))**2 + (convolve(g16,sobel()[1]))**2))
    names.append('sobel_g16')

# --- Membrane Projections --- # 200 - 205
    mp = np.zeros((data.shape[0],data.shape[1],30))

    for i in range(30):
        mp[:,:,i] = convolve(data, proj(angle=i*6.0))

        
    fi.append(np.sum(mp, axis=2))
    names.append('mj_sum')
    fi.append(np.mean(mp, axis=2))
    names.append('mj_mean')
    fi.append(np.std(mp, axis=2))
    names.append('mj_std')
    fi.append(np.median(mp, axis=2))
    names.append('mj_median')
    fi.append(np.max(mp, axis=2))
    names.append('mj_max')
    fi.append(np.min(mp, axis=2))
    names.append('mj_min')


    
    # convert list fi to array fi
    fi = np.asarray(fi)

    return fi, names



if __name__ == "__main__":
    N = 100
    out = feat(np.zeros((N,N)))
        
