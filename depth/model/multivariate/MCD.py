import numpy as np
from ctypes import *
from scipy.stats import chi2
from .import_CDLL import libExact

def MCD(data, h, seed=None, mfull = 10, nstep = 7, hiRegimeCompleteLastComp = True):

    try:
        n, d = data.shape
    except ValueError:
        n = data.shape[0]
        d = 1

    hParam = pointer(c_int(h))
    numPoints = pointer(c_int(n))
    dimension = pointer(c_int(d))

    points_list=data.flatten()
    points=(c_double*len(points_list))(*points_list)
    points=pointer(points)

    if seed==None:
        seeded = False
        seed=0
    else:
        seeded = True

    seed=pointer((c_int(seed)))
    c_seeded = c_bool(seeded)

    cov_size = d*d
    # print("cov_size",cov_size)
    mat_MCD=pointer((c_double*(cov_size))(*np.zeros((cov_size))))

    chisqr05 =  chi2(d).isf(0.5)
    chisqr0975 = chi2(d).isf(0.025)
    # print("chisqr05",chisqr05)
    # print("chisqr0975",chisqr0975)
    c_chisqr05 = c_double(chisqr05)
    c_chisqr0975 = c_double(chisqr0975)
    c_mfull = c_int(mfull)
    c_nstep = c_int(nstep)
    c_hiRegimeCompleteLastComp = c_bool(hiRegimeCompleteLastComp)

    libExact.MinimumCovarianceDeterminantEstim(points, numPoints, dimension, hParam, seed, mat_MCD,c_chisqr05,c_chisqr0975,c_mfull,c_nstep,c_hiRegimeCompleteLastComp,c_seeded)

    res = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            res[i,j]=mat_MCD[0][i*d+j]    

    return res

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
