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
    ----------
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
    ----------
    xs: np.ndarray
        x coordinate for plotting

    ys: np.ndarray
        y coordinate for plotting

    depth_grid: np.ndarray
        depth values for the grid
    """
    model._check_variables(mah_estimate=mah_estimate,mah_parMCD=mah_parMCD,NRandom=NRandom,n_refinements=n_refinements,
                            sphcap_shrink=sphcap_shrink,alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,)
    xs, ys, depth_grid=mtv.depth_mesh(data=model.data,notion=notion,freq=freq,xlim=xlim,ylim=ylim,mah_estimate=mah_estimate,mah_parMCD=mah_parMCD,beta=beta,option=1,
                    distance=distance,Lp_p=Lp_p,exact=exact,method=method,k=k,solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                    alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,line_solver=line_solver,bound_gc=bound_gc,)
    return xs, ys, depth_grid

def depth_plot2d(model:DepthEucl, notion:str = "halfspace",freq:list = [100, 100], xlim:List[int]|List[float]=None, ylim:List[int]|List[float]=None, cmap:str = "YlOrRd", 
                    ret_depth_mesh:bool= False,xs = None, ys = None,
                    val_mesh = None,mah_estimate = "moment",mah_parMCD = 0.75,beta = 2,distance = "Lp",Lp_p = 2,exact = True,method = "recursive",k = 0.05,
                    solver = "neldermead",NRandom = 1000,n_refinements = 10,sphcap_shrink = 0.5,alpha_Dirichlet = 1.25,cooling_factor = 0.95,
                    cap_size = 1, start = "mean", space = "sphere", line_solver = "goldensection", bound_gc = True):
    """
    Plots the 2D view of the depth
    """
    model._check_variables(mah_estimate=mah_estimate, mah_parMCD=mah_parMCD, beta=beta, distance=distance, NRandom=NRandom, n_refinements=n_refinements, 
                            sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, cap_size=cap_size,)
    fig, ax, im =mtv.depth_plot2d(data=model.data,
                        notion=notion, freq=freq, xlim=xlim, ylim=ylim, cmap=cmap, ret_depth_mesh=ret_depth_mesh, xs=xs, ys=ys, val_mesh=val_mesh, 
                        mah_estimate=mah_estimate, mah_parMCD=mah_parMCD, beta=beta, distance=distance, Lp_p=Lp_p, exact=exact, method=method, k=k, 
                        solver=solver, NRandom=NRandom, option=1, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, 
                        cooling_factor=cooling_factor, cap_size=cap_size, start=start, space=space, line_solver=line_solver, bound_gc=bound_gc, )
    return fig, ax, im