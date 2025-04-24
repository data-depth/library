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
        self.approxOption=["lowest_depth","final_direction","all_depth","all_depth_directions"]
        self.set_seed()
        if True:
            # main direction
            self.betaSkeletonDir=None
            self.cexpchullDir=None
            self.cexpchullstarDir=None
            self.geometricalDir=None
            self.halfspaceDir=None
            self.L2Dir=None
            self.mahalanobisDir=None
            self.potentialDir=None
            self.projectionDir=None
            self.aprojectionDir=None
            self.qhpeelingDir=None
            self.simplicialDir=None
            self.simplicialVolumeDir=None
            self.zonoidDir=None
            # depth values
            self.betaSkeletonDepth=None
            self.cexpchullDepth=None
            self.cexpchullstarDepth=None
            self.geometricalDepth=None
            self.halfspaceDepth=None
            self.L2Depth=None
            self.mahalanobisDepth=None
            self.potentialDepth=None
            self.projectionDepth=None
            self.aprojectionDepth=None
            self.qhpeelingDepth=None
            self.simplicialDepth=None
            self.simplicialVolumeDepth=None
            self.spatialDepth=None
            self.zonoidDepth=None
            # MCD
            self.MCD=None
            # approximate depth and direction
            self.allDepth,self.allDirections,self.dirIndiex=None,None,None
    

    def load_dataset(self,data:np.ndarray=None,y:np.ndarray|None=None)->None:
        """
        Load the dataset X
        """
        if type(data)==None:
            raise Exception("You must load a dataset")
        assert(type(data)==np.ndarray), "The dataset must be a numpy array"
        self.data=data
        return self

    def mahalanobis(self, x: np.ndarray = None, exact: bool = True, mah_estimate: Literal["moment", "mcd"] = "moment",
                    mah_parMcd: float = 0.75,solver= "neldermead", NRandom= 1000, 
                    n_refinements= 10, sphcap_shrink=0.5, 
                    alpha_Dirichlet= 1.25, cooling_factor=0.95, 
                    cap_size=1, start="mean", space= "sphere", 
                    line_solver="goldensection", bound_gc= True, 
                    output_option:Literal["lowest_depth","final_direction","all_depth","all_depth_directions"]="final_direction")->np.ndarray:
        """Mahalanobis depth
        Parameters
        ----------
        x
        exact 
        mah_estimate
        mah_parMcd
        """
        
        self._check_dataset() #check if dataset is loaded
        self._check_variables(x=x,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
            NRandom=NRandom, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
            alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
            cap_size=cap_size, output_option=output_option, )

        option=self.approxOption.index(output_option)+1 # define option for function return 
        memorySize=x.size*x.itemsize*NRandom//1048576 # compute an estimate of the memory amount used for option 4
        if memorySize>2 and option==4:
            print("output_option demands too much memory, output_option automatically set to 'final_direction'")
            option=2
        DM=mtv.mahalanobis(
            x,self.data,exact,mah_estimate.lower(),mah_parMcd,
            solver=solver, NRandom=NRandom, 
            n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
            alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
            cap_size=cap_size, start=start, space=space, 
            line_solver=line_solver, bound_gc=bound_gc,option=option, 
                        ) #compute depth value
        if exact or option==1:self.mahalanobisDepth=DM # assign value - exact or option 1
        elif option==2:self.mahalanobisDepth,self.mahalanobisDir=DM # assign value option 2
        elif option==3:self.mahalanobisDepth,self.mahalanobisDir,self.allDepth=DM # assign value option 3
        elif option==4:self.mahalanobisDepth,self.mahalanobisDir,self.allDepth,self.allDirections,self.dirIndiex=DM # assign value option 4
        
        return self.mahalanobisDepth

    def aprojection(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass    
    def betaSkeleton(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def cexpchull(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def cexpchullstar(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def geometrical(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def halfspace(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def L2(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def potential(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def projection(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def qhpeeling(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def simplicial(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def simplicialVolume(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def spatial(self,x:np.ndarray,mah_estimate:str='moment',mah_parMcd:float=0.75):
        """
        TO DO
        """
        self._check_dataset() # check if dataset is loaded
        self._check_variables(x=x,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd) #check if passed values are viable
        self.spatialDepth=mtv.spatial(x,self.data,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd)

        return self.spatialDepth
        
    def zonoid(self,x:np.ndarray, exact:bool=True,
               solver="neldermead",NRandom=1000,n_refinements=10,
               sphcap_shrink=0.5,alpha_Dirichlet=1.25,cooling_factor=0.95,cap_size=1,
               start="mean",space="sphere",line_solver="goldensection",bound_gc=True,
               output_option:Literal["lowest_depth","final_direction","all_depth","all_depth_directions"]="final_direction")->np.ndarray:
        """
        Compute zonoide depth
        """
        self._check_dataset() # check if dataset is loaded
        self._check_variables(x=x,exact=exact, 
            NRandom=NRandom,n_refinements=n_refinements,
            sphcap_shrink=sphcap_shrink,alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,
            cap_size=cap_size,output_option=output_option) #check if passed values are viable
        
        # seedZ=seed if seed!=self.seed else self.seed #set seed value to default if seed is not passed
        option=["lowest_depth","final_direction","all_depth","all_depth_directions"].index(output_option)+1 # define option for function return
        memorySize=x.size*x.itemsize*NRandom//1048576 # compute an estimate of the memory amount used for option 4
        if memorySize>2 and option==4:
            print("output_option demands too much memory, output_option automatically set to 'final_direction'")
            option=2
        DZ=mtv.zonoid(
            x,self.data,seed=self.seed,exact=exact, 
            solver=solver,NRandom=NRandom,n_refinements=n_refinements,
            sphcap_shrink=sphcap_shrink,alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,
            cap_size=cap_size,start=start,space=space,line_solver=line_solver,
            bound_gc=bound_gc,option=option) # compute zonoid depth
        
        if exact or option==1:self.zonoidDepth=DZ # assign value
        elif option==2:self.zonoidDepth,self.zonoidDir=DZ # assign value
        elif option==3:self.zonoidDepth,self.zonoidDir,self.allDepth=DZ # assign value
        elif option==4:self.zonoidDepth,self.zonoidDir,self.allDepth,self.allDirections,self.dirIndiex=DZ # assign value

        return self.zonoidDepth
    
    def depth_mesh(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def depth_plot2d(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    def calcDet(self,):
        """
        TO DO
        """
        self._check_dataset()
        pass
    
    def _MCD(self, h, seed=None, mfull: int = 10, nstep: int = 7, hiRegimeCompleteLastComp: bool = True):
        """
        TO DO
        """
        self._check_dataset()
        
        self.MCD=mtv.MCD(self.data,h,seed,mfull, nstep, hiRegimeCompleteLastComp)
        return 
    
    #### auxiliar functions #### 
    def set_seed(self,seed:int=2801):
        """Set seed for computation"""
        self.seed=seed

    def _check_dataset(self,)->None:
        """Check if the dataset is loaded"""
        if type(self.data)==None:
            raise Exception("A dataset must be loaded before depth computation")
    
    def _check_variables(self,**kwargs)->None:
        """Check if passed variable has valid value"""
        for key, value in kwargs.items():
            if key=="x":
                assert(type(value)==np.ndarray),f"x must be a numpy array, got {type(value)}"
            if key=="exact":
                if type(value)!=bool or value not in [0,1]: 
                    raise ValueError(f"exact must be a boolean or [0,1], got {value}.")
            if key=="mah_estimate":
                assert(type(value)==str), f"mah_estimate must be a string, got {type(value)}" 
                if value.lower() not in {"moment", "mcd"}: 
                    raise ValueError(f"Only mah_estimate possibilities are {{'moment', 'mcd'}}, got {value}.")
            if key=="mah_parMcd": 
                assert type(value)==float, f"mah_parMcd must be a float, got {type(value)}"
            if key=="NRandom":
                assert type(value)==int, f"NRandom must be an integer, got {type(value)}"
            if key=="n_refinements":
                assert type(value)==int, f"n_refinements must be an integer, got {type(value)}"
            if key=="sphcap_shrink":
                assert type(value)==float, f"sphcap_shrink must be a float, got {type(value)}"
            if key=="alpha_Dirichlet": 
                assert type(value)==float, f"alpha_Dirichlet must be a float, got {type(value)}"
            if key=="cooling_factor": 
                assert type(value)==float, f"cooling_factor must be a float, got {type(value)}"
            if key=="cap_size":
                assert type(value)==float or type(value)==int, f"cap_size must be a float or integer, got {type(value)}"
            if key=="output_option":
                assert type(value)==str, f"output_option must be a float, got {type(value)}"
                if value not in self.approxOption: 
                    raise ValueError(f"Only output_option possibilities are {self.approxOption}, got {value}.")
            

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
    _MCD.__doc__=mtv.MCD.__doc__
    
