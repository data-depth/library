import numpy as np
from ctypes import *
from multiprocessing import *
import math
import sklearn.covariance as sk
import sys, os, glob
import platform
from .import_CDLL import libExact

def MCD_fun(data,alpha,NeedLoc=False):
    cov = sk.MinCovDet(support_fraction=alpha).fit(data)
    if NeedLoc:return([cov.covariance_,cov.location_])
    else:return(cov.covariance_)

def betaSkeleton(x, data, beta = 2, distance = "Lp", Lp_p = 2, mah_estimate = "moment", mah_parMcd = 0.75):
	points_list=data.flatten()
	objects_list=x.flatten()
	if (distance == "Mahalanobis"):
		code = 5
		if (mah_estimate == "none"):
			sigma = np.eye(len(data[0]))
		else:
			if(mah_estimate == "moment"):
				tmpCov = np.cov(np.transpose(data))
			elif (mah_estimate == "MCD"):
				tmpCov = MCD_fun(data, mah_parMcd)
			else:
				print("Wrong argument \"mah_estimate\", should be one of \"moment\", \"MCD\", \"none\"")
			
			if (np.sum(np.isnan(tmpCov)) == 0):
				sigma = np.linalg.inv(tmpCov)
			else:
				sigma = np.eye(len(data[0]))
				print("Covariance estimate not found, no affine-invariance-adjustment")
	else:
		sigma = np.zeros(1)
		if (distance== "Lp"):
			code=4
			if (Lp_p == 1):
				code=1
			if (Lp_p == 2):
				code = 2
			if (Lp_p==math.inf and Lp_p > 0):
				code = 3
		else:print("Argument \"distance\" should be either \"Lp\" or \"Mahalanobis\"")

	points=pointer((c_double*len(points_list))(*points_list))
	objects=pointer((c_double*len(objects_list))(*objects_list))
	numPoints=pointer(c_int(len(data)))
	numObjects=pointer(c_int(len(x)))
	dimension=pointer(c_int(len(data[0])))
	beta=[beta]
	
	beta=pointer((c_double*1)(*beta))
	code=pointer(c_int(code))
	Lp_p=[Lp_p]
	Lp_p=pointer((c_double*1)(*Lp_p))
	sigma=pointer((c_double*len(sigma.flatten()))(*sigma.flatten()))
	depth=pointer((c_double*len(x))(*np.zeros(len(x))))

	libExact.BetaSkeletonDepth(points, objects, numPoints, numObjects, dimension, beta, code, Lp_p, sigma, depth)
    	
	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depth[0][i]
	return res

	

betaSkeleton.__doc__= """ 

Description
	Calculates the beta-skeleton depth of points w.r.t. a multivariate data set.

Arguments
	x		
			Matrix of objects (numerical vector as one object) whose depth is to be calculated. 
			Each row contains a d-variate point and should have the same dimension as data.

	data 		
			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

	beta
			The parameter defining the positionning of the balls’ centers, see `Yang and Modarres (2017)`_ for details.
			By default (together with other arguments) equals
			``2``, which corresponds to the lens depth, see Liu and Modarres (2011).

	distance	
			A character string defining the distance to be used for determining inclusion
			of a point into the lens (influence region), see Yang and Modarres (2017) for
			details. Possibilities are ``'Lp'`` for the Lp-metric (default) or ``'Mahalanobis'`` for
			the Mahalanobis distance adjustment.

	Lp_p
			A non-negative number defining the distance’s power equal ``2`` by default (Euclidean distance)
			is used only when ``distance='Lp'``.

	mah_estimate
			A character string specifying which estimates to use when calculating sample
			covariance matrix; can be ``'none'``, ``'moment'`` or ``'MCD'``, determining whether
			traditional moment or Minimum Covariance Determinant (MCD)
			estimates for mean and covariance are used. By default ``'moment'`` is used. Is
			used only when ``distance='Mahalanobis'``.

	mah_parMcd	
			The value of the argument alpha for Minimum Covariance Determinant (MCD); is used when ``distance='Mahalanobis'`` and ``mah.estimate='MCD'``.

References
    * Elmore, R. T., Hettmansperger, T. P. and Xuan, F. (2006). Spherical data depth and a multivariate median. In R. Y. Lui, R. Serfling, and D. L. Souvaine, (Eds.), *Data Depth: Robust Multivariate Analysis, Computational Geometry and Applications*, *DIMACS Series Discrete Mathematics and Theoretical Computer Science*, 72, American Mathematical Society, Providence, RI, 87–101.
    
    * Liu, Z. and Modarres, R. (2011). Lens data depth and median. *Journal of Nonparametric Statistics*, 23, 1063–1074.
    
    * Kleindessner, M. and Von Luxburg, U. (2017). Lens depth function and k-relative neighborhood graph: Versatile tools for ordinal data analysis. *Journal of Machine Learning Research*, 18, 58, 52.
    
    * Yang, M. and Modarres, R. (2018). :math:`{\\beta}`-skeleton depth functions and medians. *Communications in Statistics - Theory and Methods*, 47, 5127–5143.

Examples
			>>> import numpy as np
			>>> from depth.multivariate import *
			>>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
			>>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
			>>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
			>>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
			>>> BetaSkeleton(x, data)
			[0.16467668 0.336002   0.43702102 0.25827828 0.4204044  0.46894895
 			0.27825225 0.11572372 0.4663003  0.18778579]

"""
