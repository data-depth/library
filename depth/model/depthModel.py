import numpy as np
from . import multivariate as mtv
from typing import Literal

class depthModel():
    """
    Statistical data depth.

    Return the depth of each sample w.r.t. a dataset, D(z,X), using a chosen depth notion.
    
    Data depth computes the centrality (similarity, belongness) of a sample 'z' given a dataset 'X'.

    """
    def __init__(self,):
        """
        Initialize depthModel instance for statistical depth computation.

        Parameters
        ----------
        
        """  
        self.data=None
        self.dataDpt=None
    

    def load_dataset(self,data:np.ndarray=None,*,y:np.ndarray|None=None)->None:
        """
        Load the dataset X
        """
        if type(data)==None:
            raise Exception("You must load a dataset")
        assert(type(data)==np.ndarray), "The dataset must be a numpy array"
        self.data=data
        return self

    def mahalanobis(self, x: np.ndarray = None, exact: bool = True, mah_estimate: Literal["moment", "mcd"] = "moment",
                    mah_parMcd: float = 0.75,**kwargs)->np.ndarray:
        """Mahalanobis depth
        
        

        """
        # check if values are correct 
        assert(type(x)==np.ndarray),f"x must be a numpy array, got {type(x)}"
        if type(exact)!=bool or exact not in [0,1]: 
            raise ValueError(f"exact must be a boolean or [0,1], got {exact}.")
        assert(type(mah_estimate)==str), "mah_estimate must be a string" 
        if mah_estimate.lower() not in {"moment", "mcd"}: 
            raise ValueError(f"Only mah_estimate possibilities are {{'moment', 'mcd'}}, got {mah_estimate}.")
        assert(type(mah_parMcd)==float),f"mah_parMcd must be a float, got {type(mah_parMcd)}"
        
        self._check_dataset() #check if dataset is loaded

        self.mahDepth=mtv.mahalanobis(
            x,self.data,exact,mah_estimate,mah_parMcd,
            solver= "neldermead", NRandom= 1000, 
            option= 1, n_refinements= 10, sphcap_shrink=0.5, # non necessary features
            alpha_Dirichlet= 1.25, cooling_factor=0.95, 
            cap_size=1, start="mean", space= "sphere", 
            line_solver="goldensection", bound_gc= True
                        )

        return self.mahDepth

    def aprojection(self,):
        """
        A
        """
        self._check_dataset()
        pass    
    def betaSkeleton(self,):
        pass
    def cexpchull(self,):
        pass
    def cexpchullstar(self,):
        pass
    def geometrical(self,):
        pass
    def halfspace(self,):
        pass
    def L2(self,):
        pass
    def potential(self,):
        pass
    def projection(self,):
        pass
    def qhpeeling(self,):
        pass
    def simplicial(self,):
        pass
    def simplicialVolume(self,):
        pass
    def spatial(self,):
        pass
    def zonoid(self,):
        pass
    def depth_mesh(self,):
        pass
    def depth_plot2d(self,):
        pass
    def calcDet(self,):
        pass
    def MCD(self,):
        pass


    def _check_dataset(self,)->None:
        """Check if the dataset is loaded"""
        if type(self.data)==None:
            raise Exception("A dataset must be loaded before depth computation")
    

    aprojection.__doc__=mtv.aprojection.__doc__
    betaSkeleton.__doc__=mtv.betaSkeleton.__doc__
    cexpchull.__doc__=mtv.cexpchull.__doc__
    cexpchullstar.__doc__=mtv.cexpchullstar.__doc__
    geometrical.__doc__=mtv.geometrical.__doc__
    halfspace.__doc__=mtv.halfspace.__doc__
    L2.__doc__=mtv.L2.__doc__
    mahalanobis.__doc__=mtv.mahalanobis.__doc__
    potential.__doc__=mtv.potential.__doc__
    projection.__doc__=mtv.projection.__doc__
    qhpeeling.__doc__=mtv.qhpeeling.__doc__
    simplicial.__doc__=mtv.simplicial.__doc__
    simplicialVolume.__doc__=mtv.simplicialVolume.__doc__
    spatial.__doc__=mtv.spatial.__doc__
    zonoid.__doc__=mtv.zonoid.__doc__
    depth_mesh.__doc__=mtv.depth_mesh.__doc__
    depth_plot2d.__doc__=mtv.depth_plot2d.__doc__
    calcDet.__doc__=mtv.calcDet.__doc__
    MCD.__doc__=mtv.MCD.__doc__
    
