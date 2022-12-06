import numpy as np
from ctypes import *
from multiprocessing import *
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


def spatial(x, data,mah_estimate='moment',mah_parMcd=0.75):
        depths_tab=[]

        if mah_estimate=='none':
                print('none')
                lambda1=np.eye(len(data))
        elif mah_estimate=='moment':
                print('moment')
                cov=np.cov(np.transpose(data))
        elif mah_estimate=='MCD':
                print('mcd')
        if np.sum(np.isnan(cov))==0:
                w,v=np.linalg.eig(cov)
                lambda1=np.linalg.inv(np.matmul(v,np.diag(np.sqrt(w))))#invàconfirmer
        else:
                lambda1=np.eye(len(data))

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
        return depths_tab








spatial.__doc__=""" 

Description
	Calculates the spatial depth of points w.r.t. a multivariate data set.

Usage
	depth.spatial(x, data, mah.estimate = "moment", mah.parMcd = 0.75)

Arguments
        x 			
		Matrix of objects (numerical vector as one object) whose depth is to be calculated;
		each row contains a d-variate point. Should have the same dimension as data.

	data			
		Matrix of data where each row contains a d-variate point, w.r.t. which the depth
		is to be calculated.

	mah.estimate 		
		String character specifying which estimates to use when calculating sample
		covariance matrix. It can be ``none``, ``moment`` or ``MCD``, determining whether
		traditional moment or Minimum Covariance Determinant (MCD) (see covMcd)
		estimates for mean and covariance are used. By default ``moment`` is used. With
		``none`` the non-affine invariant version of Spatial depth is calculated

	mah.parMcd 		
		Argument alpha for the function covMcd is used when mah.estimate = ``MCD``.

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