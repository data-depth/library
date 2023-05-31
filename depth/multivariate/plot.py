import numpy as np
from depth.multivariate.Depth_approximation import depth_approximation
from depth.multivariate.Halfspace import halfspace
from depth.multivariate.Projection import projection
from depth.multivariate.SimplicialVolume import simplicialVolume
from depth.multivariate.Spatial import spatial
from depth.multivariate.Simplicial import simplicial
from depth.multivariate.Zonoid import zonoid
from depth.multivariate.BetaSkeleton import betaSkeleton
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

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
        cur_xs, cur_ys, depth_grid = xs, ys, depth_mesh

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
