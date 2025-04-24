import numpy as np
from ctypes import *
from .Depth_approximation import depth_approximation
from .Halfspace import halfspace
from .Projection import projection
from .Aprojection import aprojection
from .SimplicialVolume import simplicialVolume
from .Spatial import spatial
from .Simplicial import simplicial
from .Zonoid import zonoid
from .BetaSkeleton import betaSkeleton
from .L2 import L2
from .Qhpeeling import qhpeeling
from .Mahalanobis import mahalanobis
from .Cexpchull import cexpchull
from .Cexpchullstar import cexpchullstar
from .Geometrical import geometrical
import sys, os, glob
import platform
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from .import_CDLL import libExact,libApprox

def depth_mesh(data, notion = "halfspace",
                freq = [100, 100],
                xlim = None,
                ylim = None,
                mah_estimate = "moment",
                mah_parMCD = 0.75,
                beta = 2,
                distance = "Lp",
                Lp_p = 2,
                exact = True,
                method = "recursive",
                k = 0.05,
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
    
    # Prepare the depth-calculating arguments
    xs, ys = np.meshgrid(np.linspace(xlim[0], xlim[1], freq[0]), np.linspace(ylim[0], ylim[1], freq[1]))
    objects = np.c_[xs.ravel(), ys.ravel()]
    # Fork for calling a depth notion
    if notion == "halfspace":
        zDepth = halfspace(objects, data, NRandom, exact, method, solver,
            NRandom, option, n_refinements, sphcap_shrink, alpha_Dirichlet,
            cooling_factor, cap_size, start, space, line_solver, bound_gc)
    elif notion == "projection":
        zDepth = projection(objects, data, solver, NRandom, option,
            n_refinements, sphcap_shrink, alpha_Dirichlet, cooling_factor,
            cap_size, start, space, line_solver, bound_gc)
    elif notion == "aprojection":
        zDepth = aprojection(objects, data, solver, NRandom, option,
            n_refinements, sphcap_shrink, alpha_Dirichlet, cooling_factor,
            cap_size, start, space, line_solver, bound_gc)
    elif notion == "simplicialVolume":
        zDepth = simplicialVolume(objects, data, exact, k, mah_estimate,
            mah_parMCD)
    elif notion == "spatial":
        zDepth = spatial(objects, data, mah_estimate, mah_parMCD)
    elif notion == "simplicial":
        zDepth = simplicial(objects, data, exact, k)
    elif notion == "zonoid":
        zDepth = zonoid(objects, data, 0, exact, solver, NRandom, option,
            n_refinements, sphcap_shrink, alpha_Dirichlet, cooling_factor,
            cap_size, start, space, line_solver, bound_gc)
    elif notion == "betaSkeleton":
        zDepth = betaSkeleton(objects, data, beta, distance, Lp_p,
            mah_estimate, mah_parMCD)
    elif notion == "L2":
        zDepth = L2(objects, data, mah_estimate, mah_parMCD)
    elif notion == "qhpeeling":
        zDepth = qhpeeling(objects, data)
    elif notion == "mahalanobis":
        zDepth = mahalanobis(objects, data, mah_estimate, mah_parMCD)
    elif notion == "cexpchullstar":
        zDepth = cexpchullstar(objects, data, solver, NRandom, option,
            n_refinements, sphcap_shrink, alpha_Dirichlet, cooling_factor,
            cap_size, start, space, line_solver, bound_gc)
    elif notion == "cexpchull":
        zDepth = cexpchull(objects, data, solver, NRandom, option,
            n_refinements, sphcap_shrink, alpha_Dirichlet, cooling_factor,
            cap_size, start, space, line_solver, bound_gc)
    elif notion == "geometrical":
        zDepth = geometrical(objects, data, solver, NRandom, option,
            n_refinements, sphcap_shrink, alpha_Dirichlet, cooling_factor,
            cap_size, start, space, line_solver, bound_gc)
    # Shape the grid
    depth_grid = zDepth.reshape(xs.shape)
    
    return xs, ys, depth_grid

def depth_plot2d(data, notion = "halfspace",
                freq = [100, 100],
                xlim = None,
                ylim = None,
                cmap = "YlOrRd",
                ret_depth_mesh = False,
                xs = None,
                ys = None,
                val_mesh = None,
                mah_estimate = "moment",
                mah_parMCD = 0.75,
                beta = 2,
                distance = "Lp",
                Lp_p = 2,
                exact = True,
                method = "recursive",
                k = 0.05,
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

    if xs is None or ys is None or val_mesh is None:
        # Verify the plot's limits
        if xlim is None:
            x_span = max(data[:,0]) - min(data[:,0])
            cur_xlim = [min(data[:,0]) - x_span * 0.1, max(data[:,0]) + x_span * 0.1]
        else:
            cur_xlim = xlim
        if ylim is None:
            y_span = max(data[:,1]) - min(data[:,1])
            cur_ylim = [min(data[:,1]) - y_span * 0.1, max(data[:,1] + y_span * 0.1)]
        else:
            cur_ylim = ylim

        # Caclulate the depth mesh
        cur_xs, cur_ys, zDepth = depth_mesh(data, notion, freq, cur_xlim,
            cur_ylim, mah_estimate, mah_parMCD, beta, distance, Lp_p, exact,
            method, k, solver, NRandom, option, n_refinements, sphcap_shrink,
            alpha_Dirichlet, cooling_factor, cap_size, start, space,
            line_solver, bound_gc)
        # Shape the grid
        depth_grid = zDepth.reshape(cur_xs.shape)
    else:
        cur_xs, cur_ys, depth_grid = xs, ys, val_mesh

    # Introduce colors
    levels = MaxNLocator(nbins = 100).tick_values(0, 1)
    if isinstance(cmap, str):
        col_map = plt.get_cmap(cmap)
    else:
        col_map = cmap
    norm = BoundaryNorm(levels, ncolors = col_map.N, clip = True)
    # Plot the color mesh
    depth_mesh_cut = np.copy(depth_grid)
    depth_mesh_cut[depth_mesh_cut == 0] = float('nan') # white color for zero depth
    depth_mesh_cut = depth_mesh_cut[:-1,:-1]
    fig, ax = plt.subplots()
    im = ax.pcolormesh(cur_xs, cur_ys, depth_mesh_cut, cmap = col_map, norm = norm)
    
    # Return selected values depending on arguments
    if ret_depth_mesh:
        return fig, ax, im, cur_xs, cur_ys, depth_grid
    else:
        return fig, ax, im

depth_mesh.__doc__="""

Description
    Computes data depth values for a bi-variate mesh of points w.r.t. a multivariate data set.

Arguments
    data 			
        Matrix of data where each row contains a d-variate point, w.r.t. which the depth
        is to be calculated.

    notion
         - ``'aprojection'`` for asymmetric projection depth;
         - ``'betaSkeleton'`` for beta-skeleton depth (including lens and spherical depth);
         - ``'cexpchull'`` for continuous expected convex hull depth;
         - ``'cexpchullstar'`` for continuous modified expected convex hull depth;
         - ``'geometrical'`` for geometrical depth;
         - ``'halfspace'`` for the halfspace depth;
         - ``'L2'`` for L2-depth;
         - ``'mahalanobis'`` for Mahalanobis depth;
         - ``'projection'`` for projection depth;
         - ``'qhpeeling'`` for convex hull peeling (or onion) depth;
         - ``'simplicial'`` for simplicial depth;
         - ``'simplicialVolume'`` for simplicial volume (or Oja) depth;
         - ``'spatial'`` for spatial depth;
         - ``'zonoid'`` for zonoid depth.

    freq
        Frequency in abscisse and ordinate.
        
    xlim
        Range of values for abscisse.
        
    ylim
        Range of values for ordinate.
        
    mah_estimate, ...
        Depth-dependent parameters (see the corresponding depth functions), which shall be forwarded to the depth-computing routines.

"""

depth_plot2d.__doc__="""

Description
    Plots the surface of data depth values for a bi-variate mesh of points w.r.t. a multivariate data set.

Arguments
    data             
        Matrix of data where each row contains a d-variate point, w.r.t. which the depth
        is to be calculated.

    notion
         - ``'aprojection'`` for asymmetric projection depth;
         - ``'betaSkeleton'`` for beta-skeleton depth (including lens and spherical depth);
         - ``'cexpchull'`` for continuous expected convex hull depth;
         - ``'cexpchullstar'`` for continuous modified expected convex hull depth;
         - ``'geometrical'`` for geometrical depth;
         - ``'halfspace'`` for the halfspace depth;
         - ``'L2'`` for L2-depth;
         - ``'mahalanobis'`` for Mahalanobis depth;
         - ``'projection'`` for projection depth;
         - ``'qhpeeling'`` for convex hull peeling (or onion) depth;
         - ``'simplicial'`` for simplicial depth;
         - ``'simplicialVolume'`` for simplicial volume (or Oja) depth;
         - ``'spatial'`` for spatial depth;
         - ``'zonoid'`` for zonoid depth.

    freq
        Frequency in abscisse and ordinate.
        
    xlim
        Range of values for abscisse.
        
    ylim
        Range of values for ordinate.
        
    cmap
        Color map.
    
    ret_depth_mesh
        Should the depth mesh be returned?

    mah_estimate, ...
        Depth-dependent parameters (see the corresponding depth functions), which shall be forwarded to the depth-computing routines.

"""
