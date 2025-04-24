import numpy as np
from ctypes import *
from multiprocessing import *
import sklearn.covariance as sk
import sys, os, glob
import platform

def MCD_fun(data,alpha,NeedLoc=False):
    cov = sk.MinCovDet(support_fraction=alpha).fit(data)
    if NeedLoc:return([cov.covariance_,cov.location_])
    else:return(cov.covariance_)

def L2(x, data,mah_estimate='moment',mah_parMcd=0.75):
	points_list=data.flatten()
	objects_list=x.flatten()
	
	if mah_estimate=='none':
		sigma=np.eye(len(data[0]))
	else:
		if mah_estimate=='moment':
			cov=np.cov(np.transpose(data))
		elif mah_estimate=='MCD':
			cov=MCD_fun(data, mah_parMcd)
		else :
			print("Wrong argument \"mah.estimate\", should be one of \"moment\", \"MCD\", \"none\"")
			print("moment is used")
			cov=np.cov(np.transpose(data))
			
		if np.sum(np.isnan(cov))==0:
			sigma=np.linalg.inv(cov)
		else:
			print("Covariance estimate not found, no affine-invariance-adjustment")
			sigma=np.eye(len(data))
	
	depths=(-1)*np.ones(len(x))
	for i in range(len(x)):
		tmp1=(x[i]-data)
		tmp2=np.matmul(tmp1,sigma)
		tmp3=np.sum(tmp2 * tmp1,axis=1)
		depths[i]=1/(1 + np.mean(np.sqrt(tmp3)))
	return depths

L2.__doc__=""" 

Description
			Calculates the L2-depth of points w.r.t. a multivariate data set.
			
Arguments
	x 			
			Matrix of objects (numerical vector as one object) whose depth is to be calculated; 
			each row contains a d-variate point. Should have the same dimension as data.

	data
 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

	mah_estimate
			Is a character string specifying which estimates to use when calculating sample
			covariance matrix; can be ``'none'``, ``'moment'`` or ``'MCD'``, determining whether
			traditional moment or Minimum Covariance Determinant (MCD) estimates for mean and covariance are used. By default ``'moment'`` is used. With
			``'none'`` the non-affine invariant version of the L2-depth is calculated.

	mah_parMcd
			is the value of the argument alpha for the function covMcd; is used when
			``mah.estimate='MCD'``.
			
References
    * Zuo, Y. and Serfling, R. (2000). General notions of statistical depth function. *The Annals of Statistics*, 28, 461–482.
    
    * Mosler, K. and Mozharovskyi, P. (2022). Choosing among notions of multivariate depth statistics. *Statistical Science*, 37(3), 348-368.
   
Examples
			>>> import numpy as np
			>>> from depth.multivariate import *
			>>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
			>>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
			>>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
			>>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
			>>> L2(x, data)
			[0.2867197  0.19718391 0.18896649 0.24623271 0.20979579 0.22055673
 			0.20396566 0.20779032 0.24901829 0.26734192]

"""
