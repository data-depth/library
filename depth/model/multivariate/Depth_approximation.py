import numpy as np
from ctypes import *
from math import ceil
import sys, os, glob
import platform
from .import_CDLL import libApprox
    
def depth_approximation(z,
                        X,
                        notion = "halfspace",
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

    depth_indice = check_depth(notion)
    check_space(space)
    solver_indice = check_solver(solver, space)
    start_indice = check_start(start)
    line_solver_indice = check_line_solver(line_solver)
    check_bound(bound_gc)

    try:
        n, d = X.shape
    except ValueError:
        n = X.shape[0]
        d = 1
    n_z = z.shape[0]

    if(d == 1):
        option = 1

    try:
        n_z, d_z = z.shape
    except ValueError:
        if(d == 1):
            try:
                n_z = z.shape[0]
            except IndexError:
                n_z = 1
        else:
            n_z = 1

    if(option == 1):
        depths = np.empty(n_z, dtype=np.double)
        best_directions = np.empty(((1, 1)), dtype=np.double)
        depths_iter = np.empty((1, 1), dtype=np.double)
        directions = np.full(((1, 1, 1)), -1, dtype=np.double)
        directions_card = np.full(((1, 1)), -1, dtype=np.int32)
    elif(option == 2):
        depths = np.empty(n_z, dtype=np.double)
        best_directions = np.empty(((n_z, d)), dtype=np.double)
        depths_iter = np.empty((1, 1), dtype=np.double)
        directions = np.full(((1, 1, 1)), -1, dtype=np.double)
        directions_card = np.full(((1, 1)), -1, dtype=np.int32)
    elif(option == 3):
        depths = np.empty(n_z, dtype=np.double)
        best_directions = np.empty(((n_z, d)), dtype=np.double)
        depths_iter = np.empty((n_z, NRandom), dtype=np.double)
        directions = np.full(((1, 1, 1)), -1, dtype=np.double)
        directions_card = np.full(((1, 1)), -1, dtype=np.int32)
    elif(option == 4):
        depths = np.empty(n_z, dtype=np.double)
        best_directions = np.empty(((n_z, d)), dtype=np.double)
        depths_iter = np.empty((n_z, NRandom), dtype=np.double)
        directions = np.full(((n_z, NRandom, d)), -1, dtype=np.double)
        directions_card = np.full(((n_z, NRandom)), -1, dtype=np.int32)

    points_list=X.flatten()
    objects_list=z.flatten()
    points=(c_double*len(points_list))(*points_list)
    objects=(c_double*len(objects_list))(*objects_list)
    points=pointer(points)
    objects=pointer(objects)


    libApprox.depth_approximation(
        objects,
        points,
        c_int(depth_indice),
        c_int(solver_indice),
        c_int(NRandom),
        c_int(option),
        c_int(n_refinements),
        c_double(sphcap_shrink),
        c_double(alpha_Dirichlet),
        c_double(cooling_factor),
        c_double(cap_size),
        c_int(start_indice),
        c_int(line_solver_indice),
        c_int(bound_gc),
        c_int(n),
        c_int(d),
        c_int(n_z),
        c_void_p(depths.ctypes.data),
        c_void_p(depths_iter.ctypes.data),
        c_void_p(directions.ctypes.data),
        c_void_p(directions_card.ctypes.data),
        c_void_p(best_directions.ctypes.data)
        )
    
    if(option == 2 or option == 3 or option == 4):
        for i in range(n_z):
            if(np.sum(z[i]*best_directions[i]) < np.sum(z[i]*(-best_directions[i]))):
                    best_directions[i] = -best_directions[i]


    if(option == 1):
        return depths
    elif(option == 2):
        return depths, best_directions
    elif(option == 3):
        return depths, best_directions, depths_iter
    elif(option == 4):
        # Resize and clear array of every directions unused
        directions = directions.tolist()
        for i in range(n_z):
            for j in range(NRandom):
                if(directions[i][j].count(-1) != 0):
                    directions[i] = directions[i][:j] # Clear -1 values
                    break
        
        # Fill indices for every start of convergence
        ind_convergence = []
        for i in range(n_z):
            if(solver == "refinedgrid" or solver == "refinedrandom"): # Return indices of refinements step
                ind_convergence = np.arange(0, NRandom, NRandom//n_refinements)[:ceil(len(directions[0])/(NRandom/n_refinements))].tolist()
            else:
                ind_bin = directions_card[i, ~(directions_card[i] == -1)] # Clear every -1 value 
                ind_bin_cumsum = np.cumsum(ind_bin)
                ind_convergence.append((ind_bin_cumsum - ind_bin).tolist())
        return depths, best_directions, depths_iter, directions, ind_convergence

def check_depth(depth):
    all_depths = ["mahalanobis", "halfspace", "zonoid", "projection", "aprojection", "cexpchullstar", "cexpchull", "geometrical"]
    if (depth not in all_depths):
        raise ValueError("Depths approximation is available only for depths in %s, got %s."%(all_depths, depth))
    else:
        return all_depths.index(depth)

def check_solver(solver, space):
    all_solvers = ["simplegrid", "refinedgrid", "simplerandom", "refinedrandom",
                "coordinatedescent", "randomsimplices", "neldermead", "simulatedannealing"]
    if solver not in all_solvers:
        raise ValueError("Depths approximation supports only solvers in %s, got %s."%(all_solvers, solver))
    else:
        if(solver == "coordinatedescent" and space == "sphere"):return 8 # Indice of the solver in ProjectionDepths
        elif(solver == "coordinatedescent" and space == "euclidean"):return all_solvers.index("coordinatedescent")
        elif(solver == "neldermead" and space == "sphere"):return 9 # Indice of the solver in ProjectionDepths
        elif(solver == "neldermead" and space == "euclidean"):return all_solvers.index("neldermead")
        else:
            return all_solvers.index(solver)

def check_start(start):
    all_start = ["mean", "random"]
    if (start not in all_start):
        raise ValueError("Only start available are in %s, got %s."%(all_start, start))
    else:
        return all_start.index(start)

def check_space(space):
    all_space = ["sphere", "euclidean"]
    if (space not in all_space):
        raise ValueError("Only space available are in %s, got %s."%(all_space, space))

def check_line_solver(line_solver):
    all_line_solver = ["uniform", "goldensection"]
    if (line_solver not in all_line_solver):
        raise ValueError("Only line_solver available are in %s, got %s."%(all_line_solver, line_solver))
    else:
        return all_line_solver.index(line_solver)

def check_bound(bound):
    all_bound = [True, False]
    if (bound not in all_bound):
        raise ValueError("Only bound option available are in %r, got %r."%(all_bound, bound))
    
depth_approximation.__doc__="""

Description
     Compute data depth approximation based on the weak projection property.
     
Usage
    depth_approximation(z, X, notion = "halfspace", solver = "neldermead", NRandom = 100, option = 1, n_refinements = 10, sphcap_shrink = 0.5, alpha_Dirichlet = 1.25, cooling_factor = 0.95, cap_size = 1, start = "mean", space = "sphere", line_solver = "goldensection", bound_gc = True)

Arguments
    z 
           Points whose depth is to be calculated, each row contains a d-variate point.
           Should have the same dimension as `X`.
        
    X 
           Data where each row contains a d-variate point, w.r.t. which the depth is to be calculated.
           
    notion 
           {'halfspace', 'mahalanobis', 'zonoid', 'projection', 'aprojection', 'cexpchull'}, **optional**
           Which depth will be computed.
           
    solver 
           {'simplegrid', 'refinedgrid', 'simplerandom', 'refinedrandom', 'coordinatedescent', 'randomsimplices', 'neldermead', 'simulatedannealing'}, **optional**
           The type of solver used to approximate the depth.
           
    NRandom 
           The total number of iterations to compute the depth. Some solvers are converging
           faster so they are run several time to achieve ``NRandom`` iterations.
           
    option
       |		If ``option`` = ``1``, only approximated depths are returned.
       |		If ``option`` = ``2``, depths calculated at every iteration are also returned.
       |		If ``option`` = ``3``, best directions to approximate depths are also returned  
       |		If ``option`` = ``4``, random directions used to project depths are also returned with indices of converging for the solver selected.
        
    n_refinements  
        For ``solver`` = ``refinedrandom`` or ``refinedgrid``, set the maximum of iteration for 
        computing the depth of one point. **Optional**
        
    sphcap_shrink  
        For ``solver`` = ``refinedrandom`` or `refinedgrid`, it's the shrinking of the spherical cap. **Optional**
        
    alpha_Dirichlet  
        For ``solver`` = ``randomsimplices``. it's the parameter of the Dirichlet distribution. **Optional**
        
    cooling_factor  
        For ``solver`` = ``randomsimplices``, it's the cooling factor. **Optional**
        
    cap_size 
        For ``solver`` = ``simulatedannealing`` or ``neldermead``, it's the size of the spherical cap. **Optional**
        
    start 
        {'mean', 'random'}, **optional**
        For ``solver`` = ``simulatedannealing`` or ``neldermead``, it's the method used to compute the first depth.
        
    space  
        {'sphere', 'euclidean'}, **optional**
        For ``solver`` = ``coordinatedescent`` or ``neldermead``, it's the type of spacecin which
        the solver is running.
        
    line_solver 
        {'uniform', 'goldensection'}, **optional**
        For ``solver`` = ``coordinatedescent``, it's the line searh strategy used by this solver.
        
    bound_gc 
        For ``solver`` = ``neldermead``, it's ``True`` if the search is limited to the closed hemisphere.

Examples
            >>> import numpy as np
            >>> from depth.multivariate import *
            >>> np.random.seed(1)
            >>> n = 100
            >>> d = 3
            >>> mean = np.zeros(d)
            >>> cov = np.eye(d)
            >>> X = np.random.multivariate_normal(mean, cov, n)
            >>> z = np.random.multivariate_normal(mean, cov, 20)
            >>> depth_approximation(z, X, notion = "halfspace", solver = "neldermead", NRandom = 100, option = 1, cap_size = 1, start = "mean", space = "sphere", bound_gc = True)		
            [0.   0.02 0.15 0.08 0.   0.1  0.09 0.07 0.03 0.04 0.02 0.03 0.   0.
             0.25 0.28 0.03 0.11 0.13 0.1 ]


"""
