import numpy as np
from ctypes import *
from multiprocessing import *
import math
import sklearn.covariance as sk
import sys, os, glob
import platform

if sys.platform=='linux':
	for i in sys.path :
		if i.split('/')[-1]=='site-packages':
			ddalpha_exact=glob.glob(i+'/depth/UNIX/'+'ddalpha.so')
			ddalpha_approx=glob.glob(i+'/depth/UNIX/'+'depth_wrapper.so')
	

	libr=CDLL(ddalpha_exact[0])
	libRom=CDLL(ddalpha_approx[0])
	
if sys.platform=='darwin':
	for i in sys.path :
		if i.split('/')[-1]=='site-packages':
			ddalpha_exact=glob.glob(i+'/depth/MACOS/'+'ddalpha.so')
			ddalpha_approx=glob.glob(i+'/depth/MACOS/'+'depth_wrapper.so')
	

	libr=CDLL(ddalpha_exact[0])
	libRom=CDLL(ddalpha_approx[0])

if sys.platform=='win32' and platform.architecture()[0] == "64bit":
	site_packages = next(p for p in sys.path if 'site-packages' in p)
	print(site_packages)
	os.add_dll_directory(site_packages+"\depth\Win64")
	libr=CDLL(r""+site_packages+"\depth\Win64\ddalpha.dll")
	libRom=CDLL(r""+site_packages+"\depth\Win64\depth_wrapper.dll")
	
if sys.platform=='win32' and platform.architecture()[0] == "32bit":
	site_packages = next(p for p in sys.path if 'site-packages' in p)
	print(site_packages)
	os.add_dll_directory(site_packages+"\depth\Win32")
	libr=CDLL(r""+site_packages+"\depth\Win32\ddalpha.dll")
	libRom=CDLL(r""+site_packages+"\depth\Win32\depth_wrapper.dll")


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
				cov = np.cov(np.transpose(data))
			elif (mah_estimate == "MCD"):
				cov = MCD_fun(data, mah_parMcd)
			else:
				print("Wrong argument \"mah_estimate\", should be one of \"moment\", \"MCD\", \"none\"")
			
			if (np.sum(np.isnan(cov)) == 0):
				sigma = np.linalg.inv(cov)
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

	
	
	libr.BetaSkeletonDepth(points, objects, numPoints, numObjects, dimension, beta, code, Lp_p, sigma, depth)
    	
	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depth[0][i]
	return res

	
	
	
betaSkeleton.__doc__= """ 

Description
	Calculates the beta-skeleton depth of points w.r.t. a multivariate data set.

Usage
	depth.betaSkeleton(x, data, beta = 2, distance = "Lp", Lp.p = 2, mah.estimate = "moment", mah.parMcd = 0.75)

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
			details. Possibilities are ``Lp`` for the Lp-metric (default) or ``Mahalanobis`` for
			the Mahalanobis distance adjustment.

	Lp.p 		
			A non-negative number defining the distance’s power equal ``2`` by default (Euclidean distance)
			is used only when distance = ``Lp``.

	mah.estimate 	
			A character string specifying which estimates to use when calculating sample
			covariance matrix; can be ``none``, ``moment`` or ``MCD``, determining whether
			traditional moment or Minimum Covariance Determinant (MCD) (see covMcd)
			estimates for mean and covariance are used. By default ``moment`` is used. Is
			used only when distance = ``Mahalanobis``.

	mah.parMcd	
			The value of the argument alpha for the function covMcd; is used when distance
			= ``Mahalanobis`` and mah.estimate = ``MCD``.

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



