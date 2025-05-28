import numpy as np
import matplotlib.pyplot as plt
from ..model.DepthEucl import DepthEucl
from ..model import multivariate as mtv
from typing import List


#### Plot ####
def depth_mesh(model:DepthEucl,notion:str = "halfspace",freq:List[int] = [100, 100],xlim:List[int]|None = None,ylim:List[int]|None = None,
                mah_estimate:str = "moment",mah_parMCD:float = 0.75,beta:int = 2,distance:str = "Lp",Lp_p:int = 2,exact:bool = True,
                method:str = "recursive",k:float = 0.05,solver:str = "neldermead",NRandom:int = 1000,n_refinements:int = 10,
                sphcap_shrink:float = 0.5,alpha_Dirichlet:float = 1.25,cooling_factor:float = 0.95, cap_size:float|int = 1,
                start:str = "mean", space:str = "sphere", line_solver:str = "goldensection", bound_gc:bool = True
                )->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Computes the depth mesh
            
    Parameters
        notion: str, default="halfspace"
            Chosen notion for depth computation. The mesh will be computed using this notion to map the 2D space

        freq: List[int], defaul=[100,100]
            Amount of points to map depth in both dimensions. 

        xlim: List[int], default=None
            Limits for x value computation. 
            If None, value is determined based on dataset values. 
        
        ylim: List[int], default=None
            Limits for y value computation. 
            If None, value is determined based on dataset values.

    Results
        xs: np.ndarray
            x coordinate for plotting

        ys: np.ndarray
            y coordinate for plotting

        depth_grid: np.ndarray
            depth values for the grid
    """
    if type(xlim)==type(None):xlim=[model.data[:,0].min(),model.data[:,0].max()]
    if type(ylim)==type(None):ylim=[model.data[:,1].min(),model.data[:,1].max()]
    model._check_variables(mah_estimate=mah_estimate,mah_parMCD=mah_parMCD,NRandom=NRandom,n_refinements=n_refinements,
                            sphcap_shrink=sphcap_shrink,alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,)
    xs, ys, depth_grid=mtv.depth_mesh(data=model.data,notion=notion,freq=freq,xlim=xlim,ylim=ylim,mah_estimate=mah_estimate,mah_parMCD=mah_parMCD,beta=beta,option=1,
                    distance=distance,Lp_p=Lp_p,exact=exact,method=method,k=k,solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                    alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,line_solver=line_solver,bound_gc=bound_gc,)
    return xs, ys, depth_grid

depth_mesh.__doc__==    """
    Computes the depth mesh
            
    Parameters
        model: Euclidean Depth model
            Model with loaded dataset

        notion: str, default="halfspace"
            Chosen notion for depth computation. The mesh will be computed using this notion to map the 2D space

        freq: List[int], defaul=[100,100]
            Amount of points to map depth in both dimensions. 

        xlim: List[int], default=None
            Limits for x value computation. 
            If None, value is determined based on dataset values. 
        
        ylim: List[int], default=None
            Limits for y value computation. 
            If None, value is determined based on dataset values.

        exact : bool, delfaut=True
            Whether the depth computation is exact.
    
        mah_estimate : str, {"moment", "mcd"}, default="moment"
            Specifying which estimates to use when calculating the depth
        
        mah_parMcd : float, default=0.75
            Value of the argument alpha for the function covMcd
        
        solver : str, default="neldermead"
            The type of solver used to approximate the depth.
        
        NRandom : int, default=1000
            Total number of directions used for approximate depth

        n_refinements : int, default = 10
            Number of iterations used to approximate the depth
            For ``solver='refinedrandom'`` or ``'refinedgrid'`` 
        
        sphcap_shrink : float, default = 0.5
            For ``solver`` = ``refinedrandom`` or `refinedgrid`, it's the shrinking of the spherical cap.
        
        alpha_Dirichlet : float, default = 1.25
            For ``solver`` = ``randomsimplices``. it's the parameter of the Dirichlet distribution. 
        
        cooling_factor : float, default = 0.95
            For ``solver`` = ``randomsimplices``, it's the cooling factor.
        
        cap_size : int | float, default = 1
            For ``solver`` = ``simulatedannealing`` or ``neldermead``, it's the size of the spherical cap.
        
        start : str {'mean', 'random'}, default = mean 
            For ``solver`` = ``simulatedannealing`` or ``neldermead``, it's the method used to compute the first depth.
        
        space : str {'sphere', 'euclidean'}, default = sphere 
            For ``solver`` = ``coordinatedescent`` or ``neldermead``, it's the type of spacecin which
        
        line_solver : str {'uniform', 'goldensection'}, default = goldensection
            For ``solver`` = ``coordinatedescent``, it's the line searh strategy used by this solver.
        
        bound_gc : bool, default = True
            For ``solver`` = ``neldermead``, it's ``True`` if the search is limited to the closed hemispher
                pretransform: str, default="1Mom"
    		The method of data scaling.
			``'1Mom'`` or ``'NMom'`` for scaling using data moments.
			``'1MCD'`` or ``'NMCD'`` for scaling using robust data moments (Minimum Covariance Determinant (MCD).
        
        kernel: str, default="EDKernel"
			``'EDKernel'`` for the kernel of type 1/(1+kernel.bandwidth*EuclidianDistance2(x,y)),
			``'GKernel'`` [default and recommended] for the simple Gaussian kernel,
			``'EKernel'`` exponential kernel: exp(-kernel.bandwidth*EuclidianDistance(x, y)),
			``'VarGKernel'`` variable Gaussian kernel, where kernel.bandwidth is proportional to the depth.zonoid of a point.
        
        kernel_bandwidth: int, default=0
			the single bandwidth parameter of the kernel. If ``0`` - the Scott`s rule of thumb is used.
        
        k: float, default=0.05
            Number (``k > 1``) or portion (if ``0 < k < 1``) of simplices that are considered if ``exact=False``.
            If ``k > 1``, then the algorithmic complexity is polynomial in d but is independent of the number of observations in data, given k. 
            If ``0 < k < 1``,then the algorithmic complexity is exponential in the number of observations in data, 
                but the calculation precision stays approximately the same.


    Results
        xs: np.ndarray
            x coordinate for plotting

        ys: np.ndarray
            y coordinate for plotting

        depth_grid: np.ndarray
            depth values for the grid
    """