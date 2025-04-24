import numpy as np
from ctypes import *
from multiprocessing import *
import sys, os, glob
import platform
import sklearn.covariance as sk
    
def MCD_fun(data,alpha,NeedLoc=False):
    cov = sk.MinCovDet(support_fraction=alpha).fit(data)
    if NeedLoc:return([cov.covariance_,cov.location_])
    else:return(cov.covariance_)

def spatial(x, data,mah_estimate='moment',mah_parMcd=0.75):
    depths_tab=[]

    if mah_estimate=='none':
        lambda1=np.eye(len(data))
        cov=np.empty((data.shape[1], data.shape[1]))
        cov[:]=np.nan
    elif mah_estimate=='moment':
        cov=np.cov(np.transpose(data))
    elif mah_estimate=='MCD':
        cov=MCD_fun(data,mah_parMcd)
    if np.sum(np.isnan(cov))==0:
        w,v=np.linalg.eig(cov)
        lambda1=np.linalg.inv(np.matmul(v,np.diag(np.sqrt(w))))
    else:
        lambda1=np.eye(data.shape[1])

    depths=np.repeat(-1,len(x),axis=0)
    for i in range(len(x)):
        interm=[]
        tmp1_ter=np.transpose(x[i]-data)
        tmp1=np.transpose(np.matmul(lambda1,tmp1_ter))
        tmp1_bis=np.sum(tmp1,axis=1)
        for elements in tmp1_bis:
            if elements==0:
                interm.append(False)
            if elements!=0:
                interm.append(True)
        
        interm=np.array(interm)
        tmp1=tmp1[interm]
        tmp2=1/np.sqrt(np.sum(np.power(tmp1,2),axis=1))
        tmp3=np.zeros([len(tmp1),len(tmp1[0])])
        tmp1=np.transpose(tmp1)
        for jj in range(len(tmp1)):
            tmp3[:,jj]=tmp2*(tmp1[:][jj])
        tmp4=np.sum(tmp3,axis=0)/len(data)
        tmp5=np.power((tmp4),2)
        tmp6=np.sum(tmp5)
        depths_tab.append(1-np.sqrt(tmp6))
    return np.array(depths_tab)

spatial.__doc__=""" 

Description
	Calculates the spatial depth of points w.r.t. a multivariate data set.

Arguments
    x
        Matrix of objects (numerical array) whose depth is to be calculated; each row contains a d-variate point. Should have the same dimension as data.
        
    data
		Matrix of data where each row contains a d-variate point, w.r.t. which the depth is to be calculated.
  
    mah_estimate
		String character specifying which estimates to use when calculating sample covariance matrix. It can be ``'none'``, ``'moment'`` or ``'MCD'``, determining whether traditional moment or Minimum Covariance Determinant (MCD) estimates for mean and covariance are used. By default ``'moment'`` is used. With ``'none'`` the non-affine invariant version of spatial depth is calculated.
    
    mah_parMcd
		Argument alpha for the function covMcd is used when ``mah.estimate='MCD'``.

References
    * Serfling, R. (2002). A depth function and a scale curve based on spatial quantiles. In Dodge, Y. (Ed.), *Statistical Data Analysis Based on the L1-Norm and Related Methods*, *Statisctics in Industry and Technology*, Birkhäuser, Basel, 25–38.

Examples
        >>> import numpy as np
        >>> from depth.multivariate import *
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
        >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
        >>> spatial(x, data)
        [0.22548919451212823, 0.14038895785356165, 0.2745517635029123, 0.35450156620496354,
        0.42373722245348566, 0.34562025044812095, 0.3585616673301636, 0.16916309940691643,
        0.573349631625784, 0.32017213635679687]

"""
