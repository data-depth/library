import numpy as np
from ctypes import *
from multiprocessing import *
import scipy.spatial as scsp
import sys, os, glob
import platform

if sys.platform=='linux':
    
    for i in sys.path :
        if i.split('/')[-1]=='site-packages':
            ddalpha_exact=glob.glob(i+'/*ddalpha*.so')
            ddalpha_approx=glob.glob(i+'/*depth_wrapper*.so')
    


    libr=CDLL(ddalpha_exact[0])
    libRom=CDLL(ddalpha_approx[0])
    
if sys.platform=='darwin':
    for i in sys.path :
        if i.split('/')[-1]=='site-packages':
            ddalpha_exact=glob.glob(i+'/*ddalpha*.so')
            ddalpha_approx=glob.glob(i+'/*depth_wrapper*.so')
  
    libr=CDLL(ddalpha_exact[0])
    libRom=CDLL(ddalpha_approx[0])

if sys.platform=='win32' and platform.architecture()[0] == "64bit":
    site_packages = next(p for p in sys.path if 'site-packages' in p)
    
    os.add_dll_directory(site_packages)
    ddalpha_exact=glob.glob(site_packages+'/depth/src/*ddalpha*.dll')
    ddalpha_approx=glob.glob(site_packages+'/depth/src/*depth_wrapper*.dll')
    libr=CDLL(r""+ddalpha_exact[0])
    libRom=CDLL(r""+ddalpha_approx[0])
    
if sys.platform=='win32' and platform.architecture()[0] == "32bit":
    site_packages = next(p for p in sys.path if 'site-packages' in p)
    
    os.add_dll_directory(site_packages)
    ddalpha_exact=glob.glob(site_packages+'/depth/src/*ddalpha*.dll')
    ddalpha_approx=glob.glob(site_packages+'/depth/src/*depth_wrapper*.dll')
    libr=CDLL(r""+ddalpha_exact[0])
    libRom=CDLL(r""+ddalpha_approx[0])


def count_convexes(objects,points,cardinalities, seed = 0):
    tmp_x=points.flatten()
    tmp_x=pointer((c_double*len(tmp_x))(*tmp_x))
    dimension=pointer(c_int(len(points[0])))
    numClasses=pointer(c_int(1))
    tmp_objects=objects.flatten()
    tmp_objects=pointer((c_double*len(tmp_objects))(*tmp_objects))
    PY_numObjects=len(objects)
    numObjects=pointer(c_int(PY_numObjects))
    tmp_cardinalities=pointer(c_int(cardinalities))
    seed=pointer(c_int(seed))
    length=PY_numObjects*1
    init_zeros=np.zeros(length,dtype=int)
    isInConv=pointer((c_int*length)(*init_zeros))
    libr.IsInConvexes(tmp_x,dimension,tmp_cardinalities,numClasses,tmp_objects,numObjects,seed,isInConv)
    res=np.zeros(length)
    for i in range(length):
        res[i]=isInConv[0][i]
    res.reshape(PY_numObjects,1)
    return res

def is_in_convex(x, data, cardinalities, seed = 0):
    res=count_convexes(x, data, cardinalities, seed)
    return res 

def qhpeeling(x, data):
    points_list=data.flatten()
    objects_list=x.flatten()
    nrow_data=len(data)
    depths=np.zeros(len(x))
    tmpData=data
    for i in range(nrow_data):
        if (len(tmpData)<(len(data[0])*(len(data[0])+1)+0.5)):
            break
        tmp=is_in_convex(x,tmpData,len(tmpData))
        depths+=tmp
        tmp_conv=scsp.ConvexHull(tmpData)
        tmpData=np.delete(tmpData,np.unique(np.array(tmp_conv.simplices)),0)
    depths=depths/nrow_data
    return depths



qhpeeling.__doc__= """

Description
    Calculates the convex hull peeling depth of points w.r.t. a multivariate data set.

Usage
    depth.qhpeeling(x, data)

Arguments
    x
        Matrix of objects (numerical vector as one object) whose depth is to be calculated; each row contains a d-variate point. Should have the same dimension as data.

    data            
        Matrix of data where each row contains a d-variate point, w.r.t. which the depth is to be calculated.
            
References
    * Barnett, V. (1976). The ordering of multivariate data. *Journal of the Royal Statistical Society*, *Series A*, 139, 318–355.
    
    * Eddy, W. F. (1981). Graphics for the multivariate two-sample problem: Comment. *Journal of the American Statistical Association*, 76, 287–289.
            
Examples
            >>> from depth.multivariate import *
            >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
            >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 100)
            >>> qhpeeling(x, data)
            [0.   0.   0.   0.   0.   0.   0.01 0.   0.   0.01]

"""
