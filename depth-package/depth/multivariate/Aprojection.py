from ctypes import *
from depth.multivariate.Depth_approximation import depth_approximation
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

def aprojection(x, data,
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

    return depth_approximation(x, data, "aprojection", solver, NRandom, option, n_refinements,
    sphcap_shrink, alpha_Dirichlet, cooling_factor, cap_size, start, space, line_solver, bound_gc)


aprojection.__doc__="""

Description
    Calculates the asymmetric projection depth of points w.r.t. a multivariate data set.
    
Usage
    aprojection(x, data, solver = "neldermead", NRandom = 100, option = 1, n_refinements = 10, sphcap_shrink = 0.5, alpha_Dirichlet = 1.25, cooling_factor = 0.95, cap_size = 1,start = "mean", space = "sphere", line_solver = "goldensection", bound_gc = True)

Arguments
    x 			
            Matrix of objects (numerical vector as one object) whose depth is to be calculated;
            each row contains a d-variate point. Should have the same dimension as data.

    data 			
            Matrix of data where each row contains a d-variate point, w.r.t. which the depth
            is to be calculated.

    solver 	       
            The type of solver used to approximate the depth.
            {`simplegrid`, 'refinedgrid', 'simplerandom', 'refinedrandom', 'coordinatedescent', 'randomsimplices', 'neldermead','simulatedannealing'}, **optional**

    NRandom 	       
            The total number of iterations to compute the depth. Some solvers are converging
            faster so they are run several time to achieve ``NRandom`` iterations.	       

    option                
        |       If ``option`` = ``1``, only approximated depths are returned.
        |		If ``option`` = ``2``, best directions to approximate depths are also returned.
        |		If ``option`` = ``3``, depths calculated at every iteration are also returned.
        |		If ``option`` = ``4``, random directions used to project depths are also returned with indices of converging for the solver selected.
            
        
    n_refinements         
            Set the maximum of iteration for computing the depth of one point.
            For ``solver`` = ``refinedrandom`` or ``refinedgrid``.
            
    sphcap_shrink         
            It's the shrinking of the spherical cap. For ``solver`` = ``refinedrandom`` or ``refinedgrid``.

    alpha_Dirichlet       
            It's the parameter of the Dirichlet distribution. For ``solver`` = ``randomsimplices``. 

    cooling_factor        
            It's the cooling factor. For ``solver`` = ``randomsimplices``.

    cap_size              
            It's the size of the spherical cap. For ``solver`` = ``simulatedannealing`` or ``neldermead``. 

    start                
            {'mean', 'random'}, **optional**
            For ``solver`` = ``simulatedannealing`` or ``neldermead``, it's the method used to compute the first depth.
                        
    space         
            {'sphere', 'euclidean'}, **optional**
            For ``solver`` = ``coordinatedescent`` or ``neldermead``, it's the type of spacecin which the solver is running.
                        
    line_solver           
            {'uniform', 'goldensection'}, **optional**
            For ``solver`` = ``coordinatedescent``, it's the line searh strategy used by this solver.
                        
    bound_gc             
            For ``solver`` = ``neldermead``, it's ``True`` if the search is limited to the closed hemisphere.

Examples
            >>> import numpy as np
            >>> from depth.multivariate import *
            >>> np.random.seed(0)
            >>> mat1=[[1, 0, 0, 0, 0],[0, 2, 0, 0, 0],[0, 0, 3, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]]
            >>> mat2=[[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]]
            >>> x = np.random.multivariate_normal([1,1,1,1,1], mat2, 10)
            >>> data = np.random.multivariate_normal([0,0,0,0,0], mat1, 100)
            >>> aprojection(x, data, NRandom=1000)
            [0.090223   0.19577999 0.15769263 0.20123535 0.10375507 0.14635662
             0.20611053 0.17846703 0.19801984 0.23230606]             
"""