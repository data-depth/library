import numpy as np
from ctypes import *
from .Depth_approximation import depth_approximation
import sys, os, glob
import platform
from .CUDA_approximation import cudaApprox
from .import_CDLL import libExact,libApprox

def halfspace(x, data, exact=True, method="recursive",
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
                bound_gc = True,
                CUDA=False):
    if exact:
        if (method =="recursive" or method==1):
            method=1
        elif (method =="plane" or method==2):
            method=2
        elif (method =="line" or method==3):
            method=3
        else:
            print("Wrong argument, method=str(recursive) or str(plane) or str(line)")
            print("recursive by default")
            method=3

        points_list=data.flatten()
        objects_list=x.flatten()
        points=(c_double*len(points_list))(*points_list)
        objects=(c_double*len(objects_list))(*objects_list)
        # k=numDirections

        points=pointer(points)

        objects=pointer(objects)
        numPoints=pointer(c_int(len(data)))
        numObjects=pointer(c_int(len(x)))
        dimension=pointer(c_int(len(data[0])))
        algNo=pointer((c_int(method)))
        depths=pointer((c_double*len(x))(*np.zeros(len(x))))
    
        libExact.HDepthEx(points,objects, numPoints,numObjects,dimension,algNo,depths)
    
        res=np.zeros(len(x))
        for i in range(len(x)):
            res[i]=depths[0][i]
        return res
    else:	
        if CUDA==False:return depth_approximation(x, data, "halfspace", solver, NRandom ,option, n_refinements,
        sphcap_shrink, alpha_Dirichlet, cooling_factor, cap_size, start, space, line_solver, bound_gc)
        if CUDA==True:
            return cudaApprox(data,x, "halfspace", solver, option,NRandom, n_refinements,
        sphcap_shrink,)

halfspace.__doc__="""

Description
    Calculates the exact and approximated Tukey (=halfspace, location) depth (Tukey, 1975) of points w.r.t. a multivariate data set.

Arguments
    x 			
        Matrix of objects (numerical vector as one object) whose depth is to be calculated; 
        each row contains a d-variate point. Should have the same dimension as data.

    data 			
        Matrix of data where each row contains a d-variate point, w.r.t. which the depth
        is to be calculated.

    exact
        The type of the used method. The default is ``exact=False``, which leads to approx-
        imate computation of the Tukey depth.
        If ``exact=True``, the Tukey depth is computed exactly, with ``method='recursive'`` by default.

    method			
        For ``exact=True``, the Tukey depth is calculated as the minimum over all combinations of k points from data (see Details below).
        In this case parameter method specifies k, with possible values 1 for ``method='recursive'`` (by default), d−2
        for ``method='plane'``, d−1 for ``'method=line'``.
        The name of the method may be given as well as just parameter exact, in which
        case the default method will be used.
                   
    solver 	       
        The type of solver used to approximate the depth.
        {``'simplegrid'``, ``'refinedgrid'``, ``'simplerandom'``, ``'refinedrandom'``, ``'coordinatedescent'``, ``'randomsimplices'``, ``'neldermead'``, ``'simulatedannealing'``}

    NRandom 	       
        The total number of iterations to compute the depth. Some solvers are converging 
        faster so they are run several time to achieve ``NRandom`` iterations.
                   
    option                
        |		If ``option=1``, only approximated depths are returned.
        |		If ``option=2``, best directions to approximate depths are also returned.
        |		If ``option=3``, depths calculated at every iteration are also returned.
        |		If ``option=4``, random directions used to project depths are also returned with indices of converging for the solver selected.

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
    * Tukey, J. W. (1975). Mathematics and the picturing of data. In R. James (Ed.), *Proceedings of the International Congress of Mathematicians*, Volume 2, Canadian Mathematical Congress, 523–531.
    
    * Donoho, D. L. and M. Gasko (1992). Breakdown properties of location estimates based on halfspace depth and projected outlyingness. *The Annals of Statistics*, 20(4), 1803–1827.
    
    * Dyckerhoff, R. and Mozharovskyi, P. (2016): Exact computation of the halfspace depth. *Computational Statistics and Data Analysis*, 98, 19–30.

    * Dyckerhoff, R., Mozharovskyi, P., and Nagy, S. (2021). Approximate computation of projection depths. *Computational Statistics and Data Analysis*, 157, 107166.

Examples
        >>> import numpy as np
        >>> from depth.multivariate import *
        >>> mat1=[[1, 0, 0],[0, 2, 0],[0, 0, 1]]
        >>> mat2=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
        >>> x = np.random.multivariate_normal([1,1,1], mat2, 10)
        >>> data = np.random.multivariate_normal([0,0,0], mat1, 200)
        >>> halfspace(x, data)
        [0.    0.005 0.005 0.    0.04  0.01  0.    0.    0.04  0.01 ]
        >>> halfspace(x, data, exact=True)
        [0.    0.005 0.005 0.    0.04  0.01  0.    0.    0.04  0.01 ]

"""
    

    

