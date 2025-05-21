import numpy as np
from ctypes import *
from .Depth_approximation import depth_approximation
import sys, os, glob
import platform
from .import_CDLL import libExact,libApprox

def zonoid(x, data, seed=0, exact=True, solver="neldermead",
                        NRandom=1000,
                        option=1,
                        n_refinements=10,
                        sphcap_shrink=0.5,
                        alpha_Dirichlet=1.25,
                        cooling_factor=0.95,
                        cap_size=1,
                        start="mean",
                        space="sphere",
                        line_solver="goldensection",
                        bound_gc=True):
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
        seed=pointer((c_int(seed)))
        depths=pointer((c_double*len(x))(*np.zeros(len(x))))

        libExact.ZDepth(points,objects, numPoints,numObjects,dimension,seed,depths)

        res=np.zeros(len(x))
        for i in range(len(x)):
            res[i]=depths[0][i]
        return res
    else:
        return depth_approximation(x, data, "zonoid", solver, NRandom, option, n_refinements,
        sphcap_shrink, alpha_Dirichlet, cooling_factor, cap_size, start, space, line_solver, bound_gc)

zonoid.__doc__= """

Description
    Calculates the zonoid depth of points w.r.t. a multivariate data set.

Arguments
    x 		
        Matrix of objects (numerical vector as one object) whose depth is to be calculated;
        each row contains a d-variate point. Should have the same dimension as data.

    data 		
        Matrix of data where each row contains a d-variate point, w.r.t. which the depth is to be calculated.

    exact
        The type of the used method. The default is ``exact=True``, which leads to exact computation of the zonoid depth using the method described by Dyckerhoff et al. (1996). If ``exact=False``, approximate computation of the zonoid depth is performed using the method defined by the argument ``solver``.
            
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
        {'mean', 'random'}.
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
    * Dyckerhoff, R., Koshevoy, G. and Mosler, K. (1996). Zonoid data depth: Theory and computation. In A. Pratt, (Ed.), COMPSTAT 1996, *Proceedings in Computational Statistics*, Physica-Verlag, Heidelberg, 235–240.
    
    * Koshevoy, G. and Mosler, K. (1997). Zonoid trimming for multivariate distributions. *The Annals of Statistics*, 25, 1998–2017.
    
    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
        >>> import numpy as np
        >>> from depth.multivariate import *
        >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
        >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
        >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
        >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 1000)
        >>> zonoid(x, data)
        [0.         0.00769552 0.03087017 0.         0.30945453 0.0142515
            0.         0.01970896 0.02169483 0.        ]

"""
