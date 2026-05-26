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
        n = X.shape[0]
        d = 1
    n_z = z.shape[0]
    points_list=X.flatten()
    points=(c_double*len(points_list))(*points_list)
    objects_list=z.flatten()
    objects=(c_double*len(objects_list))(*objects_list)
    distrSeq_list=distributions.flatten()
    distrSeq=(c_int*len(distrSeq_list))(*distrSeq_list)
    points=pointer(points)
    objects=pointer(objects)
    distrSeq=pointer(distrSeq)

    distr=np.unique(distributions,return_counts=True)[1]
    distribution_list=distr.flatten()
    distribution=(c_int*len(distribution_list))(*distribution_list)
    distribution=pointer(distribution)

    CSum=np.zeros(distr.shape,dtype=int)
    CSum[1:]=distr.cumsum(dtype=int)[:-1]
    cumSum_list=CSum.flatten()
    cumSum=(c_int*len(cumSum_list))(*cumSum_list)
    cumSum=pointer(cumSum)
    numPoints=pointer(c_int(n))
    numObjects=pointer(c_int(n_z))
    dimension=pointer(c_int(d))
    seed=pointer((c_int(seed)))
    numClasses=pointer(c_int(distr.shape[0]))
    belongs=pointer((c_int*len(z))(*np.zeros(distr.shape[0],dtype=int)))
    libExact.IsInConvexes(points,dimension,distribution,numClasses,
                      objects,numObjects,seed,belongs,cumSum, distrSeq,)
    res=np.zeros((distr.shape[0],len(z)))
    for i in range(distr.shape[0]):
        for j in range(len(z)):
            res[i][j]=belongs[i][j]


    return res


    