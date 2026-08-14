import numpy as np
from ctypes import *
from scipy.stats import chi2
from .import_CDLL import libExact

def MCD(data, h, seed=1, mfull = 10, nstep = 7, hiRegimeCompleteLastComp = True):

    try:
        n, d = data.shape
    except ValueError:
        n, d = data.shape[0], 1

    hParam = pointer(c_int(h))
    numPoints = pointer(c_int(n))
    dimension = pointer(c_int(d))

    points_list=np.ascontiguousarray(data, dtype=np.float64).flatten()
    points=(c_double*len(points_list))(*points_list)
    c_points=cast(points, POINTER(c_double))
    # points=pointer(points)

    c_seed=(c_int(seed))

    cov_size = d*d
    # print("cov_size",cov_size)
    mat_MCD=(c_double*cov_size)(*([0.0]*cov_size))
    c_mat_MCD=cast(mat_MCD, POINTER(c_double))

    chisqr05 =  chi2(d).isf(0.5)
    chisqr0975 = chi2(d).isf(0.025)
    # print("chisqr05",chisqr05)
    # print("chisqr0975",chisqr0975)
    c_chisqr05 = c_double(chisqr05)
    c_chisqr0975 = c_double(chisqr0975)
    c_mfull = c_int(mfull)
    c_nstep = c_int(nstep)
    c_hiRegimeCompleteLastComp = c_bool(hiRegimeCompleteLastComp)

    libExact.MinimumCovarianceDeterminantEstim.restype=None
    libExact.MinimumCovarianceDeterminantEstim.argtypes = [
                        POINTER(c_double), # points (flattened row-major matrix)
                        POINTER(c_int), # numPoints
                        POINTER(c_int), # dimension
                        POINTER(c_int), # hParam
                        POINTER(c_int), # seed
                        POINTER(c_double), # mat_MCD (output, d*d)
                        c_double, # chisqr05 
                        c_double, # chisqr0975
                        c_int, # mfull
                        c_int, # nstep
                        c_bool, # hiRegimeCompleteLastComp
                        ]
    libExact.MinimumCovarianceDeterminantEstim(
                c_points, # POINTER(c_double)
                byref(c_int(n)), # POINTER(c_int)
                byref(c_int(d)), # POINTER(c_int)
                byref(c_int(h)), # POINTER(c_int)
                byref(c_seed), # POINTER(c_int)
                c_mat_MCD, # POINTER(c_double) 
                c_chisqr05, # plain c_double
                c_chisqr0975,# plain c_double
                c_mfull, # plain c_int
                c_nstep, # plain c_int
                c_hiRegimeCompleteLastComp, 
                )


    res = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            res[i,j]=c_mat_MCD[i*d+j]    

    return res

# def MCD(data, h, seed=2801, mfull = 10, nstep = 7, hiRegimeCompleteLastComp = True):

#     try:
#         n, d = data.shape
#     except ValueError:
#         n,d = data.shape[0],1

#     c_hParam = c_int(h)
#     c_numPoints = c_int(n)
#     c_dimension = c_int(d)

#     points=(c_double*data.size)(*data.flatten().astype(np.float64))
#     # points=pointer(points)

#     c_seed=c_int(int(seed))

#     cov_size = d*d
#     c_mat_MCD=(c_double*(cov_size))(*([0]*cov_size))
#     chisqr05 =  chi2(d).isf(0.5)
#     chisqr0975 = chi2(d).isf(0.025)
#     c_chisqr05 = c_double(chisqr05)
#     c_chisqr0975 = c_double(chisqr0975)
#     c_mfull = c_int(mfull)
#     c_nstep = c_int(nstep)
#     c_hiRegimeCompleteLastComp = c_bool(hiRegimeCompleteLastComp)

#     #MinimumCovarianceDeterminantEstim(double *points, int *numPoints, int *dimension, int *hParam, int *seed, double *mat_MCD, double chisqr05, double chisqr0975, int mfull, 
# 	# int nstep, bool hiRegimeCompleteLastComp)

#     libExact.MinimumCovarianceDeterminantEstim(
#         points, 
#         byref(c_numPoints), 
#         byref(c_dimension), 
#         byref(c_hParam), 
#         byref(c_seed), 
#         c_mat_MCD,
#         byref(c_chisqr05),
#         byref(c_chisqr0975),
#         byref(c_mfull),
#         byref(c_nstep),
#         byref(c_hiRegimeCompleteLastComp),
#         )

#     res = np.zeros((d,d))
#     print(c_mat_MCD[8])
#     # for i in range(d):
#     #     for j in range(d):
#     #         print(c_mat_MCD[i])        
#     # res[i,j]=c_mat_MCD[0][i*d+j]    

#     return res

MCD.__doc__= """

Description
    Calculates the Minimum Covariance Determinant covariance matrix

Arguments
    data 		
        Matrix of data where each row contains a d-variate point.

    h
        Size of the data subset to use during estimation.

    mfull 
        In the high regime n>600, number of best results we keep before computing on the full dataset (cf paper by Rousseuw and van Driessen).

    hiRegimeCompleteLastComp
        "True" if in the high n regime case in the last computation we carry computation until convergence of the solutions, 
        false if we use a fix amount of nstep number of steps.

    nstep
        In high n regime, finite number of steps to carry last computations for final solutions if we do not want to compute until convergence 
        (hiRegimeCompleteLastComp is set to false).



References
    * Peter J. Rousseeuw & Katrien Van Driessen (1999) A Fast Algorithm for the Minimum Covariance Determinant Estimator, Technometrics, 41:3, 212-223

Examples
    To write

"""
