import numpy as np
from math import ceil
import os
from scipy.linalg import null_space
from scipy import stats
import time
import ctypes as ct
import multiprocessing as mp 
import sys, os, glob
from .import_CDLL import libACA

def ACA(X, dim = 1, sample_size = None, sample = None, notion = "projection", # Can't use halfspace with NelderMead
        solver = "neldermead", NRandom = 100, n_refinements = 10, sphcap_shrink = 0.5,
        alpha_Dirichlet = 1.25, cooling_factor = 0.95, cap_size = 1, start = "mean",
        space = "sphere", line_solver = "goldensection", bound_gc = True):
    
    z=X.copy()

    if(sample_size != None and sample == None): # Run method on a (specified) sample
        ind = np.random.default_rng().choice(X.shape[0], size=sample_size, replace=False)
        X = X[ind]
    elif(sample_size == None and sample is not None):
        ind = sample
        X = X[sample]
    elif(sample_size != None and sample is not None):
        print("Can't give size of uniform sampling and your own index for sampling")
        return(None)

    # Check arguments
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
    basis = np.eye(d, dtype=np.double)
    d_aca = d
    iter_aca = dim
    n_z = z.shape[0]
    for compt in range(iter_aca):
        depths = np.empty(n_z, dtype=np.double)
        best_directions = np.empty(((n_z, d)), dtype=np.double)
        points_list=X.flatten()
        objects_list=z.flatten()
        points=(ct.c_double*len(points_list))(*points_list)
        objects=(ct.c_double*len(objects_list))(*objects_list)
        points=ct.pointer(points)
        objects=ct.pointer(objects)

        libACA.ACA(
            objects,
            points,
            ct.c_int(depth_indice),
            ct.c_int(solver_indice),
            ct.c_int(NRandom),
            ct.c_int(n_refinements),
            ct.c_double(sphcap_shrink),
            ct.c_double(alpha_Dirichlet),
            ct.c_double(cooling_factor),
            ct.c_double(cap_size),
            ct.c_int(start_indice),
            ct.c_int(line_solver_indice),
            ct.c_int(bound_gc),
            ct.c_int(n),
            ct.c_int(d),
            ct.c_int(n_z),
            ct.c_void_p(depths.ctypes.data),
            ct.c_void_p(best_directions.ctypes.data),
            ct.c_void_p(basis.ctypes.data),
            ct.c_int(d_aca),
            ct.c_int(2),
            )
        
        best_directions = np.array(best_directions, dtype=np.double)
        min_score = np.argmin(depths) # Point with highest anomaly score
        u1 = np.array(best_directions[min_score], dtype=np.double).reshape(1,-1) # Direction corresponding to highest anomaly score
        if(np.sum(z[min_score]*u1) < np.sum(z[min_score]*(-u1))): # Get a direction "pointing" to most abnormal point
            u1 = -u1
        if(compt == 0):
            ACA_tab = u1
        else:
            ACA_tab = np.concatenate((ACA_tab,u1), dtype=np.double)
        basis = null_space(ACA_tab) # Find orthogonale basis of u1 and precedent ui
        if(d_aca == 1):
            return ACA_tab.T
        d_aca -= 1

    return ACA_tab.T

######################################
# Functions for arguments verification
######################################

def check_depth(depth):
    all_depths = ["mahalanobis", "halfspace", "zonoid", "projection", "aprojection"]
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
