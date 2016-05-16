# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 00:07:16 2015

@author: good-cat
"""

def gs_orthonormalization(V) :
    
    #V is a matrix where each column contains 
    #the vectors spanning the space of which we want to compute the orthonormal base
    #Will return a matrix where each column contains an ortho-normal vector of the base of the space
    
    numberLines = V.shape[0]
    numberColumns = V.shape[1]
    
    #U is a matrix containing the orthogonal vectors (non normalized)
    from numpy.linalg import norm
    import numpy as np
    U = np.zeros((numberLines,numberColumns))
    R = np.zeros((numberLines,numberColumns))
    
    for indexColumn in range(0,numberColumns) :
        U[:,indexColumn] = V[:,indexColumn]
        
        for index in range(0,indexColumn):
            R[index,indexColumn] = np.dot(U[:,index],V[:,indexColumn])
            U[:,indexColumn] = U[:,indexColumn] - R[index,indexColumn]*U[:,index]
            
        R[indexColumn,indexColumn] = norm(U[:,indexColumn])
        U[:,indexColumn] = U[:,indexColumn] / float(R[indexColumn, indexColumn])
    
    return U