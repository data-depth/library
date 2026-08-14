import numpy as np
from ctypes import *
from math import ceil
import sys, os, glob
import platform
from .import_CDLL import libExact

def IsInConvexes(X,z,distributions,seed):
    """
    Check if points are inside the convex hull 
    """
    try:
        n, d = X.shape
    except ValueError:
        n, d = X.shape[0], 1
    n_z = z.shape[0]
    
    
    distr_uniques, counts= np.unique(distributions, return_counts=True)
    numClasses=int(counts.shape[0])

    cumsum_arr= np.zeros(numClasses, dtype=np.int32)
    cumsum_arr[1:]=counts.cumsum()[:-1]
    
    c_points=(c_double*X.size)(*X.flatten().astype(np.float64))
    c_objects=(c_double*z.size)(*z.flatten().astype(np.float64))
    c_distrSeq=(c_int*len(distributions))(*distributions.flatten().astype(np.int32))
    c_cardin=(c_int*numClasses)(*counts.astype(np.int32))
    c_cumsum=(c_int*numClasses)(*cumsum_arr)

    output_size= numClasses*n_z
    c_belongs= (c_int*output_size)(*([0]*output_size))

    c_dim=c_int(d)
    c_numClasses=c_int(numClasses)
    c_numObjects=c_int(n_z)
    c_seed=c_int(int(seed))
    
    libExact.IsInConvexes.restype=None
    libExact.IsInConvexes.argtypes=[
        POINTER(c_double), # points
        POINTER(c_int), # dimension
        POINTER(c_int), # cardinalities 
        POINTER(c_int), # number distributions 
        POINTER(c_double), # objects 
        POINTER(c_int), # number objects
        POINTER(c_int), # seed 
        POINTER(c_int), # output 
        POINTER(c_int), # cumulative sum 
        POINTER(c_int), # distribution seq
    ]

    libExact.IsInConvexes(
        c_points,
        byref(c_dim),
        c_cardin,
        byref(c_numClasses),
        c_objects,
        byref(c_numObjects),
        byref(c_seed),
        c_belongs,
        c_cumsum,
        c_distrSeq,
    )


    res=np.zeros((n_z,numClasses), dtype=np.int32)

    for i in range(n_z):
        for j in range(numClasses):
            res[i,j]=c_belongs[numClasses*i+j]



    return res



    