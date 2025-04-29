import numpy as np
from . import multivariate as mtv
from typing import Literal, List
from numba import cuda, float32
import torch
import sys, os
try:os.environ['CUDA_HOME']=os.environ.get('CUDA_PATH').split(";")[0] # Force add cuda path
except:pass

class depthModel():
    """
    Statistical data depth.

    Return the depth of each sample w.r.t. a dataset, D(x,data), using a chosen depth notion.
    
    Data depth computes the centrality (similarity, belongness) of a sample 'x' given a dataset 'data.

    Parameters
    ----------
    data : {array-like} of shape (n,d).
        Reference dataset to compute the depth of a sample x

    x : {array-like} of shape (n_samples,d).
        Samples matrix to compute depth

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

    n_refinements : int, default=10
        Number of iterations used to approximate the depth
        For ``solver='refinedrandom'`` or ``'refinedgrid'`` 
    
    sphcap_shrink : float, default =
    
    alpha_Dirichlet : float, default =
    
    cooling_factor : float, default =
    
    cap_size : int | float, default =
    
    start : str, default =
    
    space : str, default =
    
    line_solver : str, default =
    
    bound_gc : bool, default =
    
    output_option : str, default =

    """
    def __init__(self,):
        """
        Initialize depthModel instance for statistical depth computation.        
        """  
        self.data=None
        self.approxOption=["lowest_depth","final_depht_dir","all_depth","all_depth_directions"]
        self.set_seed() # set initial seed  
        self._create_selfRef() # create self. referecnces for storing depth and directions

    def load_dataset(self,data:np.ndarray=None,y:np.ndarray|None=None,cluster:np.ndarray|None=None, CUDA:bool=False)->None:
        """
        Load the dataset X for reference calculations. Depth is computed with respect to this dataset.

        Parameters
        ----------
        data : {array-like} of shape (n,d).
            Dataset that will be used for depth computation
        
        y : Ignored, default=None
            Not used, present for API consistency by convention.
        
        cluster : Ignored, default=None
            Not used, present for API consistency by convention.

        CUDA : bool, default=False
            Determine with device CUDA will be used


        Returns
        ----------
        loaded dataset
        """
        if type(data)==None:
            raise Exception("You must load a dataset")
        assert(type(data)==np.ndarray), "The dataset must be a numpy array"
        self._nSamples=data.shape[0] # define dataset size - n
        self._spaceDim=data.shape[1] # define space dimension - d
        if CUDA==False:
            self.data=data
        else: 
            if cuda.is_available():
                self.dataCuda=torch.tensor(data.T,device="cuda:0",dtype=torch.float32) 
                # Tensor is transposed to facilitate projection and depth  computation
            else:
                self.data=data
                print("CUDA is set to True, but cuda is not available, CUDA is automatically set to False")
        return self

    def mahalanobis(self, x: np.ndarray = None, exact: bool = True, mah_estimate: Literal["moment", "mcd"] = "moment",
                    mah_parMcd: float = 0.75,solver= "neldermead", NRandom= 1000, 
                    n_refinements= 10, sphcap_shrink=0.5, 
                    alpha_Dirichlet= 1.25, cooling_factor=0.95, 
                    cap_size=1, start="mean", space= "sphere", 
                    line_solver="goldensection", bound_gc= True, 
                    output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir")->np.ndarray:
        """
        Mahalanobis depth

        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth
        
        Returns
        ----------
        Mahalanobis depth : {array-like}
        """
        
        self._check_variables(x=x,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
            NRandom=NRandom, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
            alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
            cap_size=cap_size, output_option=output_option, ) # check if parameters are valid

        option=self._determine_option(x,NRandom,output_option) # determine option number 
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

    def aprojection(self,x:np.ndarray,solver: str = "neldermead", NRandom: int = 1000,
                    n_refinements: int = 10, sphcap_shrink: float = 0.5, alpha_Dirichlet: float = 1.25, 
                    cooling_factor: float = 0.95,cap_size: int = 1, start: str = "mean", space: str = "sphere", 
                    line_solver: str = "goldensection", bound_gc: bool = True,
                    output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir")->np.ndarray:
        """
        Compute asymmetric projection depth

        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth
        
        Returns
        ----------
        Asymmetrical projection depth : {array like}
        """
        self._check_variables(x=x, solver=solver, NRandom=NRandom,n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
                              alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor,cap_size=cap_size,
                              ) # check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number

        DAP=mtv.aprojection(x=x,data=self.data,solver=solver,NRandom=NRandom,option=option,
                            n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
                            cap_size=cap_size,start=start,space=space,line_solver=line_solver,bound_gc=bound_gc) #compute depth value
        
        if option==1:self.aprojectionDepth=DAP # assign val option 1
        elif option==2:self.aprojectionDepth,self.aprojectionDir=DAP # assign value option 2
        elif option==3:self.aprojectionDepth,self.aprojectionDir,self.allDepth=DAP # assign value option 3
        elif option==4:self.aprojectionDepth,self.aprojectionDir,self.allDepth,self.allDirections,self.dirIndiex=DAP # assign value option 4
        
        return self.aprojectionDir
    
    def betaSkeleton(self,x:np.ndarray, beta:int=2,distance: str = "Lp", 
                     Lp_p: int = 2, mah_estimate: str = "moment", mah_parMcd: float = 0.75)->np.ndarray:
        """
        Calculates the beta-skeleton depth.

        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Beta-skeleton depth : {array like}
        """
        self._check_variables(x=x,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd) #check validity
        self.betaSkeletonDepth=mtv.betaSkeleton(x=x,data=self.data,beta=beta,distance=distance,
                                                mah_estimate=mah_estimate,mah_parMcd=mah_parMcd) # compute depth
        return self.betaSkeletonDepth


    def cexpchull(self,x: np.ndarray,solver:str= "neldermead",NRandom:int = 1000,
                  n_refinements:int = 10, sphcap_shrink:float = 0.5,
                  alpha_Dirichlet:float = 1.25,cooling_factor:float = 0.95,
                  cap_size:int|float = 1,start:str = "mean",space:str = "sphere",
                  line_solver:str = "goldensection",bound_gc:bool = True,
                  output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir")->np.ndarray:
        """
        Compute approximately the continuous explected convex hull depth of all samples w.r.t. the dataset.

        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Continuous explected convex hull depth : {array like}
        """

        self._check_variables(
            x=x,NRandom =NRandom,output_option =output_option,n_refinements =n_refinements,
            sphcap_shrink=sphcap_shrink,alpha_Dirichlet =alpha_Dirichlet,
            cooling_factor=cooling_factor,cap_size =cap_size,
        ) # check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number 
        DC=mtv.cexpchull(
            x=x, data=self.data,solver=solver,NRandom=NRandom,option=option,n_refinements=n_refinements,
            sphcap_shrink=sphcap_shrink,alpha_Dirichlet =alpha_Dirichlet,cooling_factor=cooling_factor,
            cap_size =cap_size,start =start,space =space,line_solver =line_solver,bound_gc =bound_gc,
            ) # compute depth 
        if option==1:self.cexpchullDepth=DC # assign value
        elif option==2:self.cexpchullDepth,self.cexpchullDir=DC # assign value
        elif option==3:self.cexpchullDepth,self.cexpchullDir,self.allDepth=DC # assign value
        elif option==4:self.cexpchullDepth,self.cexpchullDir,self.allDepth,self.allDirections,self.dirIndiex=DC # assign value

        return self.cexpchullDepth
        
    def cexpchullstar(self,x: np.ndarray, solver: str = "neldermead", NRandom: int = 1000, 
        option: int = 1, n_refinements: int = 10, sphcap_shrink: float = 0.5, 
        alpha_Dirichlet: float = 1.25, cooling_factor: float = 0.95, cap_size: int = 1,
        start: str = "mean", space: str = "sphere", line_solver: str = "goldensection", bound_gc: bool = True,
        output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir")->np.ndarray:
        """
        Calculates approximately the continuous modified explected convex hull depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Continuous modified explected convex hull depth : {array like}
        """
        
        self._check_variables(x=x,NRandom=NRandom, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink,
                            alpha_Dirichlet=alpha_Dirichlet, cooling_factor= cooling_factor, cap_size=cap_size,
                              ) # check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number 
        
        DC=mtv.cexpchullstar(x=x,data=self.data, solver=solver, NRandom=NRandom, option=option, n_refinements=n_refinements, 
                          sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
                          cap_size=cap_size,start=start, space=space, line_solver=line_solver, bound_gc=bound_gc)
        if option==1:self.cexpchullstarDepth=DC # assign value
        elif option==2:self.cexpchullstarDepth,self.cexpchullstarDir=DC # assign value
        elif option==3:self.cexpchullstarDepth,self.cexpchullstarDir,self.allDepth=DC # assign value
        elif option==4:self.cexpchullstarDepth,self.cexpchullstarDir,self.allDepth,self.allDirections,self.dirIndiex=DC # assign value
        return self.cexpchullstarDepth
        
    def geometrical(self,x:np.ndarray,solver: str = "neldermead", NRandom: int = 1000, n_refinements: int = 10, 
                    sphcap_shrink: float = 0.5, alpha_Dirichlet: float = 1.25, cooling_factor: float = 0.95, 
                    cap_size: int = 1, start: str = "mean", space: str = "sphere", line_solver: str = "goldensection", bound_gc: bool = True,
                    output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir")->np.ndarray:
        """
        Compute geometrical depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Geometrical depth : {array like}
        """
        self._check_variables(
            x=x, NRandom=NRandom, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
            alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor,cap_size=cap_size,
            )# check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number 
        
        DG=mtv.geometrical(x=x,data=self.data, solver=solver, NRandom=NRandom, option=option, n_refinements=n_refinements, 
                          sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
                          cap_size=cap_size,start=start, space=space, line_solver=line_solver, bound_gc=bound_gc)
        
        if option==1:self.geometricalDepth=DG # assign value
        elif option==2:self.geometricalDepth,self.geometricalDir=DG # assign value
        elif option==3:self.geometricalDepth,self.geometricalDir,self.allDepth=DG # assign value
        elif option==4:self.geometricalDepth,self.geometricalDir,self.allDepth,self.allDirections,self.dirIndiex=DG # assign value
        return self.geometricalDepth

    def halfspace(self, x:np.ndarray,exact: bool = True,method: str = "recursive",solver: str = "neldermead",
                  NRandom: int = 1000,n_refinements: int = 10,sphcap_shrink: float = 0.5,alpha_Dirichlet: float = 1.25,cooling_factor: float = 0.95,
                  cap_size: int = 1,start: str = "mean",space: str = "sphere",line_solver: str = "goldensection",bound_gc: bool = True,
                  output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir")->np.ndarray:
        """
        Compute Halfspace depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Halfspace (Tukey) depth : {array like}
        """
        self._check_variables(x=x,NRandom=NRandom,
                              n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                              alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,) # check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number
        DH=mtv.halfspace(x=x,data=self.data,exact=exact,method=method,
            solver=solver,NRandom=NRandom,option=option,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,
            space=space,line_solver=line_solver,bound_gc=bound_gc,
        )
        if option==1:self.halfspaceDepth=DH # assign value
        elif option==2:self.halfspaceDepth,self.halfspaceDir=DH # assign value
        elif option==3:self.halfspaceDepth,self.halfspaceDir,self.allDepth=DH # assign value
        elif option==4:self.halfspaceDepth,self.halfspaceDir,self.allDepth,self.allDirections,self.dirIndiex=DH # assign value
        return self.halfspaceDepth
    
    def L2(self,x: np.ndarray, mah_estimate: str = 'moment', mah_parMcd: float = 0.75)->np.ndarray:
        """
        Compute L2 depth 
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth
        mah_estimate
        mah_parMcd

        Results
        ----------
        L2 depth : {array like}
        """
        self._check_variables(x=x,mah_estimate=mah_estimate, mah_parMcd=mah_parMcd) # check if parameters are valid
        self.L2Depth=mtv.L2(x=x,data=self.data,mah_estimate=mah_estimate, mah_parMcd=mah_parMcd)
        return self.L2Depth

    def potential(self,x:np.ndarray,pretransform: str = "1Mom", kernel: str = "EDKernel", 
                  mah_parMcd: float = 0.75, kernel_bandwidth: int = 0)->np.ndarray:
        """
        Compute potential depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Potential depth : {array like}
        """
        self._check_variables(x=x,mah_parMcd=mah_parMcd)# check if parameters are valid
        #x: Any, data: Any, pretransform: str = "1Mom", kernel: str = "EDKernel", mah_parMcd: float = 0.75, kernel_bandwidth: int = 0
        self.potentialDepth=mtv.potential(x=x, data=self.data, pretransform=pretransform, kernel=kernel, mah_parMcd=mah_parMcd, kernel_bandwidth=kernel_bandwidth)
        return self.potentialDepth
    
    def projection(self,x:np.ndarray,solver: str = "neldermead",NRandom: int = 1000,n_refinements: int = 10,
                  sphcap_shrink: float = 0.5,alpha_Dirichlet: float = 1.25,cooling_factor: float = 0.95,
                  cap_size: int = 1,start: str = "mean",space: str = "sphere",line_solver: str = "goldensection",bound_gc: bool = True,
                  CUDA:bool=False, output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir")->np.ndarray:
        """
        Compute projection depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Projection depth : {array like}
        """
        self._check_variables(x=x,NRandom=NRandom,
                              n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                              alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,) # check if parameters are valid
        CUDA=self._check_CUDA(CUDA,solver)
        option=self._determine_option(x,NRandom,output_option,CUDA) # determine option number
        DP=mtv.projection(x=x,data=self.data,solver=solver,NRandom=NRandom,option=option,
                          n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                          alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,
                          space=space,line_solver=line_solver,bound_gc=bound_gc,
        )
        if option==1:self.projectionDepth=DP # assign value
        elif option==2:self.projectionDepth,self.projectionDir=DP # assign value
        elif option==3:self.projectionDepth,self.projectionDir,self.allDepth=DP # assign value
        elif option==4:self.projectionDepth,self.projectionDir,self.allDepth,self.allDirections,self.dirIndiex=DP # assign value
        return self.projectionDepth
        
    def qhpeeling(self,x:np.ndarray)->np.ndarray:
        """
        Calculates the convex hull peeling depth
        
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth
        Results
        ----------
        Convex hull peeling depth : {array like}
        """
        self._check_variables(x=x)# check if parameters are valid
        self.qhpeelingDepth=mtv.qhpeeling(x=x,data=self.data)
        return self.qhpeelingDepth

    def simplicial(self,x:np.ndarray,exact:bool=True,k:float=0.05,)->np.ndarray:
        """
        Compute simplicial depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth
        exact:
        k:

        Results
        ----------
        Simplicial depth : {array like}
        """
        self._check_variables(x=x)# check if parameters are valid
        self.simplicialDepth=mtv.simplicial(x=x,data=self.data,exact=exact,k=k,seed=self.seed)
        return self.simplicialDepth

    def simplicialVolume(self,x:np.ndarray,exact: bool = True, k: float = 0.05, 
                         mah_estimate: str = "moment", mah_parMCD: float = 0.75,)->np.ndarray:
        """
        Compute simplicial volume depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Simplicial volume depth : {array like}
        """
        self._check_variables(x=x,mah_estimate=mah_estimate,mah_parMCD=mah_parMCD)
        self.simplicialVolumeDepth=mtv.simplicialVolume(x=x,data=self.data,exact=exact,k=k,mah_estimate=mah_estimate,
                                                        mah_parMCD=mah_parMCD,seed=self.seed)
        return self.simplicialVolumeDepth

    def spatial(self,x:np.ndarray,mah_estimate:str='moment',mah_parMcd:float=0.75)->np.ndarray:
        """
        Compute spatial depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Spatial depth : {array like}
        """
        self._check_variables(x=x,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd) # check if parameters are valid
        self.spatialDepth=mtv.spatial(x,self.data,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd)

        return self.spatialDepth
        
    def zonoid(self,x:np.ndarray, exact:bool=True,
               solver="neldermead",NRandom=1000,n_refinements=10,
               sphcap_shrink=0.5,alpha_Dirichlet=1.25,cooling_factor=0.95,cap_size=1,
               start="mean",space="sphere",line_solver="goldensection",bound_gc=True,
               output_option:Literal["lowest_depth","final_depht_dir",
                                     "all_depth","all_depth_directions"]="final_depht_dir")->np.ndarray:
        """
        Compute zonoide depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Zonoid depth : {array like}
        """
        self._check_variables(x=x,exact=exact, 
            NRandom=NRandom,n_refinements=n_refinements,
            sphcap_shrink=sphcap_shrink,alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,
            cap_size=cap_size,output_option=output_option) # check if parameters are valid
        
        # seedZ=seed if seed!=self.seed else self.seed #set seed value to default if seed is not passed
        option=self._determine_option(x,NRandom,output_option) # determine option number 
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
    #### Plot ####
    # def depth_mesh(self,notion = "halfspace",freq = [100, 100],xlim = None,ylim = None,mah_estimate = "moment",mah_parMCD = 0.75,beta = 2,distance = "Lp",Lp_p = 2,exact = True,method = "recursive",
    #                k = 0.05,solver = "neldermead",NRandom = 1000,n_refinements = 10,sphcap_shrink = 0.5,alpha_Dirichlet = 1.25,cooling_factor = 0.95, cap_size = 1,
    #                start = "mean", space = "sphere", line_solver = "goldensection", bound_gc = True)->np.ndarray:
    #     """
    #     TO DO
    #     """
    #     self._check_variables(mah_estimate=mah_estimate,mah_parMCD=mah_parMCD,NRandom=NRandom,n_refinements=n_refinements,
    #                           sphcap_shrink=sphcap_shrink,alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,)
    #     xs, ys, depth_grid=mtv.depth_mesh(data=self.data,notion=notion,freq=freq,xlim=xlim,ylim=ylim,mah_estimate=mah_estimate,mah_parMCD=mah_parMCD,beta=beta,option=1,
    #                    distance=distance,Lp_p=Lp_p,exact=exact,method=method,k=k,solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
    #                    alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,line_solver=line_solver,bound_gc=bound_gc,)
    #     return xs, ys, depth_grid
    
    # def depth_plot2d(self, notion:str = "halfspace",freq:list = [100, 100], xlim:List[int]|List[float]=None, ylim:List[int]|List[float]=None, cmap:str = "YlOrRd", 
    #                  ret_depth_mesh:bool= False,xs = None, ys = None,
    #                  val_mesh = None,mah_estimate = "moment",mah_parMCD = 0.75,beta = 2,distance = "Lp",Lp_p = 2,exact = True,method = "recursive",k = 0.05,
    #                  solver = "neldermead",NRandom = 1000,n_refinements = 10,sphcap_shrink = 0.5,alpha_Dirichlet = 1.25,cooling_factor = 0.95,
    #                  cap_size = 1, start = "mean", space = "sphere", line_solver = "goldensection", bound_gc = True):
    #     """
    #     TO DO
    #     """
    #     self._check_variables(mah_estimate=mah_estimate, mah_parMCD=mah_parMCD, beta=beta, distance=distance, NRandom=NRandom, n_refinements=n_refinements, 
    #                           sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, cap_size=cap_size,)
    #     fig, ax, im =mtv.depth_plot2d(data=self.data,
    #                      notion=notion, freq=freq, xlim=xlim, ylim=ylim, cmap=cmap, ret_depth_mesh=ret_depth_mesh, xs=xs, ys=ys, val_mesh=val_mesh, 
    #                      mah_estimate=mah_estimate, mah_parMCD=mah_parMCD, beta=beta, distance=distance, Lp_p=Lp_p, exact=exact, method=method, k=k, 
    #                      solver=solver, NRandom=NRandom, option=1, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, 
    #                      cooling_factor=cooling_factor, cap_size=cap_size, start=start, space=space, line_solver=line_solver, bound_gc=bound_gc, )
    #     return fig, ax, im 
    # def _calcDet(self,):
    #     """
    #     TO DO
    #     """
    #     self._check_variables
    #     mtv.calcDet
    #     pass
    def _MCD(self, h, seed=None, mfull: int = 10, nstep: int = 7, hiRegimeCompleteLastComp: bool = True)->None:
        """
        TO DO
        """
        self._check_variables
        if h<self.data.shape[0]*.5:
            h=int(self.data.shape[0]*.5)
            
        self.MCD=mtv.MCD(self.data,h,seed,mfull, nstep, hiRegimeCompleteLastComp)
        return 
    
    #### auxiliar functions #### 
    def set_seed(self,seed:int=2801)->None:
        """Set seed for computation"""
        self.seed=seed

    def _check_dataset(self,)->None:
        """Check if the dataset is loaded"""
        if type(self.data)==None:
            raise Exception("A dataset must be loaded before depth computation")
    def _create_selfRef(self,)->None:
        """Initialize all self.depth and self.directions"""
        # main direction and depth values
        self.cexpchullDir,self.cexpchullDepth=None, None
        self.cexpchullstarDir,self.cexpchullstarDepth=None, None
        self.geometricalDir,self.geometricalDepth=None, None
        self.halfspaceDir,self.halfspaceDepth=None, None
        self.mahalanobisDir,self.mahalanobisDepth=None, None
        self.projectionDir,self.projectionDepth=None, None
        self.aprojectionDir,self.aprojectionDepth=None,None
        self.zonoidDir,self.zonoidDepth=None,None
        self.potentialDepth=None
        self.qhpeelingDepth,self.betaSkeletonDepth,self.L2Depth=None,None,None
        self.simplicialVolumeDepth,self.simplicialDepth,self.spatialDepth=None,None,None
        # MCD
        self.MCD=None
        # approximate depth and direction
        self.allDepth,self.allDirections,self.dirIndiex=None,None,None
    def _determine_option(self,x:np.ndarray,NRandom:int,output_option:str,CUDA:bool=False)->int:
        """Determine which is the option number (following the 1 to 4 convention), 
        with a created criteria to compute option 4 - all depths and directions - of 1Gb to the direction matrix"""
        option=self.approxOption.index(output_option)+1 # define option for function return 
        memorySize=x.size*x.itemsize*NRandom//1048576 # compute an estimate of the memory amount used for option 4
        if memorySize>1 and option==4:
            print("output_option demands too much memory, output_option automatically set to 'final_direction'")
            option=2
        if CUDA and option>2:
            option=1
            print(f"{output_option} is not available for CUDA computation, output_option automatically set to 'lowest_depth'")
        return option
    
    def _check_variables(self,**kwargs)->None:
        """Check if passed parameters have valid values"""
        self._check_dataset() #check if dataset is loaded
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
            if key in ["NRandom","n_refinements"]:
                assert type(value)==int, f"{key} must be an integer, got {type(value)}"
            if key in ["mah_parMcd","sphcap_shrink","alpha_Dirichlet","cooling_factor"]: 
                assert type(value)==float, f"{key} must be a float, got {type(value)}"
            if key=="cap_size":
                assert type(value)==float or type(value)==int, f"cap_size must be a float or integer, got {type(value)}"
            if key=="output_option":
                assert type(value)==str, f"output_option must be a float, got {type(value)}"
                if value not in self.approxOption: 
                    raise ValueError(f"Only output_option possibilities are {self.approxOption}, got {value}.")
            
    def _check_CUDA(self,CUDA,solver):
        if solver not in ["simplerandom", "refinedrandom"] and CUDA==True:
            print(f"CUDA is only available for 'simplerandom', 'refinedrandom', solver is {solver}, CUDA is set to False")
            return False
        return CUDA
    # aprojection.__doc__=mtv.aprojection.__doc__
    # betaSkeleton.__doc__=mtv.betaSkeleton.__doc__
    # cexpchull.__doc__=mtv.cexpchull.__doc__
    # cexpchullstar.__doc__=mtv.cexpchullstar.__doc__
    # geometrical.__doc__=mtv.geometrical.__doc__
    # halfspace.__doc__=mtv.halfspace.__doc__
    # L2.__doc__=mtv.L2.__doc__
    # mahalanobis.__doc__=mtv.mahalanobis.__doc__
    # potential.__doc__=mtv.potential.__doc__
    # projection.__doc__=mtv.projection.__doc__
    # qhpeeling.__doc__=mtv.qhpeeling.__doc__
    # simplicial.__doc__=mtv.simplicial.__doc__
    # simplicialVolume.__doc__=mtv.simplicialVolume.__doc__
    # spatial.__doc__=mtv.spatial.__doc__
    # zonoid.__doc__=mtv.zonoid.__doc__
    # depth_mesh.__doc__=mtv.depth_mesh.__doc__
    # depth_plot2d.__doc__=mtv.depth_plot2d.__doc__
    # _calcDet.__doc__=mtv.calcDet.__doc__
    # _MCD.__doc__=mtv.MCD.__doc__
    
