import numpy as np
from ctypes import *
import sklearn.covariance as sk
from .Depth_approximation import depth_approximation
import sys, os, glob
import platform
from .import_CDLL import libExact,libApprox

def MCD_fun(data,alpha,NeedLoc=False):
    cov = sk.MinCovDet(support_fraction=alpha).fit(data)
    if NeedLoc:return([cov.covariance_,cov.location_])
    else:return(cov.covariance_)

def mahalanobis(x, data, exact=True, mah_estimate="moment", mah_parMcd = 0.75,
                solver = "neldermead",
                NRandom = 1000,
                option = 1,
                n_refinements = 10,
                sphcap_shrink = 0.5,
                alpha_Dirichlet = 1.25,
                cooling_factor = 0.95,
                cap_size = 1,
                start = "mean",
                space = "sphere",
                line_solver = "goldensection",
                bound_gc = True):
                        
    if exact:
        points_list=data.flatten()
        objects_list=x.flatten()
        
        points=(c_double*len(points_list))(*points_list)
        objects=(c_double*len(objects_list))(*objects_list)

        points=pointer(points)
        objects=pointer(objects)
        numPoints=pointer(c_int(len(data)))
        numObjects=pointer(c_int(len(x)))
        dimension=pointer(c_int(len(data[0])))
        if mah_estimate=='moment': # compute cov based on user choice
            PY_MatMCD=np.cov(np.transpose(data))
        else: # compute cov based on user choice
            PY_MatMCD=MCD_fun(data,mah_parMcd)
        PY_MatMCD=PY_MatMCD.flatten(order='C')
        mat_MCD=pointer((c_double*len(PY_MatMCD))(*PY_MatMCD))

        depths=pointer((c_double*len(x))(*np.zeros(len(x))))

        libExact.MahalanobisDepth(points,objects,numPoints,numObjects,dimension,mat_MCD,depths)
        res=np.zeros(len(x))
        for i in range(len(x)):
            res[i]=depths[0][i]
        return res
    else:
        return depth_approximation(x, data, "mahalanobis", solver, NRandom, option, n_refinements,
        sphcap_shrink, alpha_Dirichlet, cooling_factor, cap_size, start, space, line_solver, bound_gc)

mahalanobis.__doc__= """

Description
    Calculates the Mahalanobis depth of points w.r.t. a multivariate data set.

Arguments
    x 		
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as
        data.

    data 		
        Matrix of data where each row contains a d-variate point, w.r.t. which the depth
        is to be calculated.

    exact
        The type of the used method. The default is ``exact=False``, which leads to approx-
        imate computation of the Mahalanobis depth using the method defined by the argument ``solver``.
        If ``exact=True``, the Mahalanobis depth is computed exactly, using the closed-form expression.

    mah_estimate
        A character string specifying which estimates to use when calculating the Mahalanobis depth; can be "'moment'" or ``'MCD'``,
        determining whether traditional moment or Minimum Covariance Determinant (MCD) 
        estimates for mean and covariance are used. By default ``'moment'`` is used.

    mah_parMcd
        is the value of the argument alpha for the function covMcd; is used when
        mah.estimate = ``'MCD'``.
    
    solver
        The type of solver used to approximate the depth.
        {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}

    NRandom
        The total number of iterations to compute the depth. Some solvers are converging
        faster so they are run several time to achieve ``NRandom`` iterations.
                   
    option
        |        If ``option=1``, only approximated depths are returned.
        |        If ``option=2``, best directions to approximate depths are also returned.
        |        If ``option=3``, depths calculated at every iteration are also returned.
        |        If ``option=4``, random directions used to project depths are also returned with indices of converging for the solver selected.

        n_refinements
        Set the maximum of iteration for computing the depth of one point.
        For ``solver='refinedrandom'`` or ``'refinedgrid'``.
                      
    sphcap_shrink
        It's the shrinking of the spherical cap. For ``solver='refinedrandom'`` or ``'refinedgrid'``.

    alpha_Dirichlet
        It's the parameter of the Dirichlet distribution. For ``solver='randomsimplices'``.

    cooling_factor
        It's the cooling factor. For ``solver='simulatedannealing'``.

    cap_size
        It's the size of the spherical cap. For ``solver='simulatedannealing'`` or ``'neldermead'``.

    start
        {``'mean'``, ``'random'``}.
        For ``solver='simulatedannealing'`` or ``'neldermead'``, it's the method used to compute the first depth.
                      
    space
        {``'sphere'``, ``'euclidean'``}.
        For ``solver='coordinatedescent'`` or ``'neldermead'``, it's the type of spacecin which the solver is running.
                      
    line_solver
        {``'uniform'``, ``'goldensection'``}.
        For ``solver='coordinatedescent'``, it's the line searh strategy used by this solver.
                      
    bound_gc
        For ``solver='neldermead'``, it's ``True`` if the search is limited to the closed hemisphere.

References
    * Mahalanobis, P. C. (1936). On the generalized distance in statistics. *Proceedings of the National Institute of Sciences of India*, 12, 49–55.
    
    * Mosler, K. and Mozharovskyi, P. (2022). Choosing among notions of multivariate depth statistics. *Statistical Science*, 37(3), 348-368.

Examples
        >>> import numpy as np
        >>> from depth.multivariate import *
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
        >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
        >>> mahalanobis(x, data)
        [0.17849871 0.10412453 0.1331417  0.13578021 0.3154836  0.29103769
            0.13398989 0.13913017 0.59339051 0.10556139]
        >>> mahalanobis(x, data, exact="True", mah_estimate="MCD", mah_parMcd = 0.75)
        [0.17758703 0.10367974 0.131705   0.13575221 0.31847867 0.29034948
            0.13291613 0.13792774 0.59094958 0.10491694]

"""
