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

## the moment trabnsform requires MCD func
def potential(x, data, pretransform = "1Mom", kernel="EDKernel" ,mah_parMcd=0.75, kernel_bandwidth=0):

	if(kernel=="GKernel" or kernel==2):
		kernel=2
	elif(kernel=="EKernel" or kernel==3):
		kernel=3
	elif(kernel=="TriangleKernel" or kernel ==4):
		kernel=4
	else:
		kernel = 1

	if (pretransform == "1Mom" or pretransform == "NMom"):
		[mu,B_inv,cov]=Maha_moment(data)
	elif (pretransform == "1MCD" or pretransform == "NMCD"):
		[mu,B_inv,cov]=Maha_mcd(data, mah_parMcd)
	data=Maha_transform(data,mu,B_inv)
	x =Maha_transform(x,mu,B_inv)

	points_list=data.flatten()
	objects_list=x.flatten()
	points=(c_double*len(points_list))(*points_list)
	objects=(c_double*len(objects_list))(*objects_list)
	points=pointer(points)
	points2=pointer(objects)
	
	numPoints=pointer(c_int(len(data)))
	numpoints2=pointer(c_int(len(x)))
	dimension=pointer(c_int(len(data[0])))
	
	KernelType=pointer(c_int(kernel))
	ignoreself=pointer(c_int(0))
	classes=pointer((c_int(1)))
	
	if kernel_bandwidth==0:
		kernel_bandwidth=pointer(c_double(math.pow(len(data),-2/(len(data[0])+4))))
	else:
		kernel_bandwidth=pointer(c_double(kernel_bandwidth))
		
	depth=pointer((c_double*len(x))(*np.zeros(len(x))))

	libExact.PotentialDepthsCount(points,numPoints,dimension,classes,numPoints,points2,numpoints2,KernelType,kernel_bandwidth,ignoreself,depth)
	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depth[0][i]
	return res

def Maha_moment (x):
	x=np.transpose(x)
	mu =np.mean(x,axis=1)
	cov=np.cov(x)
	w,v=np.linalg.eig(cov)
	B_inv=np.linalg.inv(np.matmul(v,np.diag(np.sqrt(w))))
	return ([mu,B_inv,cov])

def Maha_mcd(x, alpha =0.5):
	[cov,mu] = MCD_fun(x,alpha,1)
	w,v=np.linalg.eig(cov)
	B_inv=np.linalg.inv(np.matmul(v,np.diag(np.sqrt(w))))
	return ([mu,B_inv,cov])


def Maha_transform (x, mu, B_inv): 
	return(np.transpose(np.matmul(B_inv,np.transpose(x-mu))))

potential.__doc__="""

Description
	Calculate the potential of the points w.r.t. a multivariate data set. The potential is the kernel-estimated density multiplied by the prior probability of a class. Different from the data depths, a density estimate measures at a given point how much mass is located around it.

Arguments
	x 			
			Matrix of objects (numerical vector as one object) whose depth is to be calculated;
			each row contains a d-variate point. Should have the same dimension as data.

	data 			
			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

	pretransform 		
			|	The method of data scaling.
			|	``'1Mom'`` or ``'NMom'`` for scaling using data moments.
			|	``'1MCD'`` or ``'NMCD'`` for scaling using robust data moments (Minimum Covariance Determinant (MCD).

	kernel			
			|	``'EDKernel'`` for the kernel of type 1/(1+kernel.bandwidth*EuclidianDistance2(x,y)),
			|	``'GKernel'`` [default and recommended] for the simple Gaussian kernel,
			|	``'EKernel'`` exponential kernel: exp(-kernel.bandwidth*EuclidianDistance(x, y)),
			|	``'VarGKernel'`` variable Gaussian kernel, where kernel.bandwidth is proportional to the depth.zonoid of a point.

	kernel.bandwidth	
			the single bandwidth parameter of the kernel. If ``0`` - the Scott’s rule of thumb is used.

	mah.parMcd		
			is the value of the argument alpha for the function covMcd is used when ``pretransform='MCD'``.

References
    * Pokotylo, O. and Mosler, K. (2019). Classification with the pot–pot plot. *Statistical Papers*, 60, 903-931.
			
Examples
			>>> import numpy as np
			>>> from depth.multivariate import *
			>>> mat1=[[1, 0, 0],[0, 2, 0],[0, 0, 1]]
			>>> mat2=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
			>>> x = np.random.multivariate_normal([1,1,1], mat2, 10)
			>>> data = np.random.multivariate_normal([0,0,0], mat1, 20)
			>>> potential(x, data)
			[7.51492797 8.34322926 5.42761506 6.25418171 4.25774485 8.09733146
 			6.65788017 5.11324521 5.74407939 9.26030661]
			>>> potential(x, data, kernel_bandwidth=0.1)
			[13.56510469 13.95553893 11.23251702 12.42491604 10.17527509 13.70947682
 			12.67352469 11.2080649  11.73402562 14.93067103]
			>>> potential(x, data, pretransform = "NMCD", mah_parMcd=0.6, kernel_bandwidth=0.1)
			[11.0603282  11.49509828  8.99303793  8.63168006  7.86456928 11.03588551
 			10.45468945  8.84989798  9.56799496 12.29832608]

"""
