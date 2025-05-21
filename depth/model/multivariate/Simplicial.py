import numpy as np
from ctypes import *
from multiprocessing import *
import scipy.special as scspecial
import sys, os, glob
import platform
from .import_CDLL import libExact

def longtoint(k):
  limit = 2000000000
  k1 = int(k/limit)
  k2 = int(k - k1*limit)
  return np.array([k1,k2])

def simplicial(x, data, exact=True, k=0.05, seed=0):
    points_list=data.flatten()
    objects_list=x.flatten()
    points=(c_double*len(points_list))(*points_list)
    objects=(c_double*len(objects_list))(*objects_list)
    points=pointer(points)
    objects=pointer(objects)
    
    numPoints=pointer(c_int(len(data)))
    numObjects=pointer(c_int(len(x)))
    dimension=pointer(c_int(len(data[0])))
    seed=pointer((c_int(seed)))
    exact=pointer((c_int(exact)))
    if k<=0:
        print("k must be positive")
        print("k=1")
        k=scspecial.comb(len(data),len(data[0]),exact=True)*k
        k=pointer((c_int*2)(*longtoint(k)))
    elif k<=1:
        k=scspecial.comb(len(data),len(data[0]),exact=True)*k
        k=pointer((c_int*2)(*longtoint(k)))
    else:
        k=pointer((c_int*2)(*longtoint(k)))

    depths=pointer((c_double*len(x))(*np.zeros(len(x))))

    libExact.SimplicialDepth(points,objects, numPoints,numObjects,dimension,seed,exact,k,depths)

    res=np.zeros(len(x))
    for i in range(len(x)):
        res[i]=depths[0][i]
    return res

simplicial.__doc__ = """

Description
    Calculates the simplicial depth of points w.r.t. a multivariate data set.

Arguments
    x 			
            Matrix of objects (numerical vector as one object) whose depth is to be calculated;
            each row contains a d-variate point. Should have the same dimension as data.

    data 			
            Matrix of data where each row contains a d-variate point, w.r.t. which the depth is to be calculated.

    exact 			
            ``exact=True`` (by default) implies the exact algorithm, ``exact=False`` implies the approximative algorithm, considering k simplices.

    k 			
            |	Number (``k > 1``) or portion (if ``0 < k < 1``) of simplices that are considered if ``exact=False``.
            |	If ``k > 1``, then the algorithmic complexity is polynomial in d but is independent of the number of observations in data, given k. 
            |	If ``0 < k < 1``,then the algorithmic complexity is exponential in the number of observations in data, but the calculation precision stays approximately the same.

    seed 			
            The random seed. The default value ``seed=0`` makes no change.

References
    * Liu , R. Y. (1990). On a notion of data depth based on random simplices. *The Annals of Statistics*, 18, 405–414.

Examples
            >>> import numpy as np
            >>> from depth.multivariate import *
            >>> mat1=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
            >>> mat2=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0], mat1, 25)
            >>> simplicial(x, data,)
            [0.04458498 0.         0.         0.         0.         0.
             0.         0.         0.         0.        ]
"""

    

