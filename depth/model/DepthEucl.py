# Authors: Leonardo Leone

import numpy as np
from . import docHelp
from . import multivariate as mtv
from typing import Literal, List
import torch
import sys, os
try:os.environ['CUDA_HOME']=os.environ.get('CUDA_PATH').split(";")[0] # Force add cuda path
except:pass

class DepthEucl():
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
    
    mah_estimate : str, {"none", "moment", "mcd"}, default="moment"
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
    
    output_option : str {"lowest_depth","final_depht_dir","all_depth","all_depth_directions}, default = final_depht_dir
        Determines what will be computated alongside with the final depth
    
    evaluate_dataset: bool, default=False,
        Boolean to determine if the loaded dataset will be evaluate

    Attributes
    ----------
    data: {array-like}, default=None,
        Returns loaded dataset

    {depth-name}Depth : {array-like}, default=None,
        Returns the computed depth using {depth-name} notion.
        Available for all depth notions.
        Example: halfspaceDepth, projectionDepth

    {depth-name}Dir : {array-like}, default=None,
        Returns the directoion whose {depth-name}Depth corresponds using {depth-name} notion.
        Available only for projection-based depths.
        Example: halfspaceDir, projectionDir
    
    {depth-name}DepthDS : {array-like}, default=None,
        Returns the computed depth of the loaded dataset using {depth-name} notion.
        Available for all depth notions.
        Example: halfspaceDepthDS, projectionDepthDS

    {depth-name}DirDS : {array-like}, default=None,
        Returns the directoion whose {depth-name}DepthDS corresponds using {depth-name} notion.
        Available only for projection-based depths.
        Example: halfspaceDirDS, projectionDirDS

    """
    def __init__(self,):
        """
        Initialize depthModel instance for statistical depth computation.        
        """  
        self.data=None
        self.approxOption=["lowest_depth","final_depht_dir","all_depth","all_depth_directions"]
        self.set_seed() # set initial seed  
        self._create_selfRef() # create self. referecnces for storing depth and directions

    def load_dataset(self,data:np.ndarray=None,distribution:np.ndarray|None=None, CUDA:bool=False,y:np.ndarray|None=None)->None:
        """
        Load the dataset X for reference calculations. Depth is computed with respect to this dataset.

        Parameters
        ----------
        data : {array-like} of shape (n,d).
            Dataset that will be used for depth computation
        
        distribution : Ignored, default=None
            Not used, present for API consistency by convention.

        CUDA : bool, default=False
            Determine with device CUDA will be used
        
        y : Ignored, default=None
            Not used, present for API consistency by convention.

        Returns
        ---------
        loaded dataset
        """
        if type(data)==None:
            raise Exception("You must load a dataset")
        assert(type(data)==np.ndarray), "The dataset must be a numpy array"
        self._nSamples=data.shape[0] # define dataset size - n
        self._spaceDim=data.shape[1] # define space dimension - d
        if type(distribution)!=type(None):
            if distribution.shape[0]!=data.shape[0]:
                raise Exception(f"distribution and dataset must have same length, {distribution.shape[0]}!={data.shape[0]}")
            self.distribution=distribution # define distributions 
            self.distRef=np.unique(distribution) # define unique dist
        else:
            self.distribution=np.repeat(0,data.shape[0])
            self.distRef=np.array([0]) # define unique dist

        if type(y)!=type(None):
            if y.shape[0]!=data.shape[0]:
                raise Exception(f"y and dataset must have same length, {y.shape[0]}!={data.shape[0]}")
            self.y=y # define y
        else:self.y=None

        if CUDA==False:
            self.data=data
            device = torch.device("cpu")
        else: 
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                self.dataCuda=torch.tensor(data.T,device=device,dtype=torch.float32) 
                self.data=data
                # Tensor is transposed to facilitate projection and depth  computation
            else:
                self.data=data
                print("CUDA is set to True, but cuda is not available, CUDA is automatically set to False")
        return self

    def mahalanobis(self, x: np.ndarray|None = None, exact: bool = True, mah_estimate: Literal["none", "moment", "mcd"] = "moment",
                    mah_parMcd: float = 0.75,solver= "neldermead", NRandom= 1000,
                    n_refinements= 10, sphcap_shrink=0.5, 
                    alpha_Dirichlet= 1.25, cooling_factor=0.95, 
                    cap_size=1, start="mean", space= "sphere", 
                    line_solver="goldensection", bound_gc= True, 
                    output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir", evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            if output_option=="all_depth"or output_option=="all_depth_directions":
                print(f"output_option is set to {output_option}, only possible for lowest_depth or final_depth_dir, \
                      automaticaly set to lowest_depth")
                output_option="lowest_depth"
            self.mahalanobisDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:self.mahalanobisDepth=np.empty((self.distRef.shape[0],x.shape[0]))

        
        self._check_variables(x=x,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
            NRandom=NRandom, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
            alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
            cap_size=cap_size, output_option=output_option, ) # check if parameters are valid

        option=self._determine_option(x,NRandom,output_option,exact=exact) # determine option number 
        if option>=2:
            if evaluate_dataset:self.mahalanobisDirDS=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            else:self.mahalanobisDir=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            if option>=3:
                self.allDepth=np.empty((self.distRef.shape[0],x.shape[0],NRandom))
            if option==4:
                self.allDirections=np.empty((self.distRef.shape[0],x.shape[0],NRandom,x.shape[1]))

        for ind, d in enumerate(self.distRef):
            DM=mtv.mahalanobis(
                x,self.data[self.distribution==d],exact,mah_estimate.lower(),mah_parMcd,
                solver=solver, NRandom=NRandom, 
                n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
                alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
                cap_size=cap_size, start=start, space=space, 
                line_solver=line_solver, bound_gc=bound_gc,option=option, 
                            ) #compute depth value
            if evaluate_dataset==False:
                if exact or option==1:self.mahalanobisDepth[ind]=DM # assign value - exact or option 1
                elif option==2:self.mahalanobisDepth[ind],self.mahalanobisDir[ind]=DM # assign value option 2
                elif option==3:self.mahalanobisDepth[ind],self.mahalanobisDir[ind],self.allDepth[ind]=DM # assign value option 3
                elif option==4:self.mahalanobisDepth[ind],self.mahalanobisDir[ind],self.allDepth[ind],self.allDirections[ind],_=DM # assign value option 4
            elif evaluate_dataset==True:
                if exact or option==1:self.mahalanobisDepthDS[ind]=DM # assign value - exact or option 1
                elif option==2:self.mahalanobisDepthDS[ind],self.mahalanobisDirDS[ind]=DM # assign value option 2
        
        
        if self.distRef.shape[0]==1: #fix for one distribution
            if evaluate_dataset:
                self.mahalanobisDepthDS=self.mahalanobisDepthDS[0]
                if option==2:self.mahalanobisDirDS=self.mahalanobisDirDS[0]
            else:
                self.mahalanobisDepth=self.mahalanobisDepth[0]
                if option>=2:self.mahalanobisDir=self.mahalanobisDir[0]
            if option>=3:self.allDepth=self.allDepth[0]
            if option>=4:self.allDirections=self.allDirections[0]
        if evaluate_dataset==False:
            if exact or option==1:return self.mahalanobisDepth
            if option==2:return self.mahalanobisDepth,self.mahalanobisDir
            if option==3:return self.mahalanobisDepth,self.mahalanobisDir,self.allDepth
            if option==4:return self.mahalanobisDepth,self.mahalanobisDir,self.allDepth,self.allDirections
        elif evaluate_dataset==True:
            if exact or option==1:return self.mahalanobisDepthDS
            if option==2:return self.mahalanobisDepthDS,self.mahalanobisDirDS
        
            

    def aprojection(self,x:np.ndarray|None=None,solver: str = "neldermead", NRandom: int = 1000,
                    n_refinements: int = 10, sphcap_shrink: float = 0.5, alpha_Dirichlet: float = 1.25, 
                    cooling_factor: float = 0.95,cap_size: int = 1, start: str = "mean", space: str = "sphere", 
                    line_solver: str = "goldensection", bound_gc: bool = True,
                    output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir", evaluate_dataset:bool=False,
                                          CUDA:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            if output_option=="all_depth"or output_option=="all_depth_directions":
                print(f"output_option is set to {output_option}, only possible for lowest_depth or final_depth_dir, \
                      automaticaly set to lowest_depth")
                output_option="lowest_depth"
            self.aprojectionDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:self.aprojectionDepth=np.empty((self.distRef.shape[0],x.shape[0]))
        self._check_variables(x=x, solver=solver, NRandom=NRandom,n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
                              alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor,cap_size=cap_size,
                              ) # check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number
        if option>=2:
            if evaluate_dataset:self.aprojectionDirDS=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            else:self.aprojectionDir=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            if option>=3:
                self.allDepth=np.empty((self.distRef.shape[0],x.shape[0],NRandom))
            if option==4:
                self.allDirections=np.empty((self.distRef.shape[0],x.shape[0],NRandom,x.shape[1]))

        for ind,d in enumerate(self.distRef):
            if CUDA:DAP=mtv.aprojection(x=x,data=self.dataCuda[:,self.distribution==d],solver=solver,NRandom=NRandom,option=option,
                                n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
                                cap_size=cap_size,start=start,space=space,line_solver=line_solver,bound_gc=bound_gc,CUDA=CUDA) #compute depth value        
            else:DAP=mtv.aprojection(x=x,data=self.data[self.distribution==d],solver=solver,NRandom=NRandom,option=option,
                                n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
                                cap_size=cap_size,start=start,space=space,line_solver=line_solver,bound_gc=bound_gc,CUDA=CUDA) #compute depth value
            if evaluate_dataset==False:
                if option==1:self.aprojectionDepth[ind]=DAP # assign val option 1
                elif option==2:self.aprojectionDepth[ind],self.aprojectionDir[ind]=DAP # assign value option 2
                elif option==3:self.aprojectionDepth[ind],self.aprojectionDir[ind],self.allDepth[ind]=DAP # assign value option 3
                elif option==4:self.aprojectionDepth[ind],self.aprojectionDir[ind],self.allDepth[ind],self.allDirections[ind],_=DAP # assign value option 4    
            elif evaluate_dataset==True:
                if option==1:self.aprojectionDepthDS[ind]=DAP # assign val option 1
                elif option==2:self.aprojectionDepthDS[ind],self.aprojectionDirDS[ind]=DAP # assign value option 2

        if self.distRef.shape[0]==1: #fix for one distribution
            if evaluate_dataset:
                self.aprojectionDepthDS=self.aprojectionDepthDS[0]
                if option==2:self.aprojectionDirDS=self.aprojectionDirDS[0]
            else:
                self.aprojectionDepth=self.aprojectionDepth[0]
                if option>=2:self.aprojectionDir=self.aprojectionDir[0]
            if option>=3:self.allDepth=self.allDepth[0]
            if option>=4:self.allDirections=self.allDirections[0]
        if evaluate_dataset==False: # return correct value
            if option==1:return self.aprojectionDepth
            if option==2:return self.aprojectionDepth,self.aprojectionDir
            if option==3:return self.aprojectionDepth,self.aprojectionDir,self.allDepth
            if option==4:return self.aprojectionDepth,self.aprojectionDir,self.allDepth,self.allDirections
        elif evaluate_dataset==True:
            if option==1:return self.aprojectionDepthDS
            if option==2:return self.aprojectionDepthDS,self.aprojectionDirDS


    
    def betaSkeleton(self,x:np.ndarray|None=None, beta:int=2,distance: str = "Lp", 
                     Lp_p: int = 2, mah_estimate: str = "moment", mah_parMcd: float = 0.75, evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            self.betaSkeletonDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:
            self.betaSkeletonDepth=np.empty((self.distRef.shape[0],x.shape[0]))
        self._check_variables(x=x,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd) #check validity

        for ind,d in enumerate(self.distRef):
            DB=mtv.betaSkeleton(x=x,data=self.data[self.distribution==d],beta=beta,distance=distance, Lp_p=Lp_p,
                                                        mah_estimate=mah_estimate,mah_parMcd=mah_parMcd) # compute depth
            if evaluate_dataset==False: self.betaSkeletonDepth[ind]=DB
            if evaluate_dataset==True: self.betaSkeletonDepthDS[ind]=DB
        if self.distRef.shape[0]==1: 
            if evaluate_dataset==True:self.betaSkeletonDepthDS=self.betaSkeletonDepthDS[0] 
            else: self.betaSkeletonDepth=self.betaSkeletonDepth[0]
        return self.betaSkeletonDepthDS if evaluate_dataset==True else self.betaSkeletonDepth
        

    def cexpchull(self,x: np.ndarray|None=None,solver:str= "neldermead",NRandom:int = 1000,
                  n_refinements:int = 10, sphcap_shrink:float = 0.5,
                  alpha_Dirichlet:float = 1.25,cooling_factor:float = 0.95,
                  cap_size:int|float = 1,start:str = "mean",space:str = "sphere",
                  line_solver:str = "goldensection",bound_gc:bool = True,
                  output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir", evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            if output_option=="all_depth"or output_option=="all_depth_directions":
                print(f"output_option is set to {output_option}, only possible for lowest_depth or final_depth_dir, \
                      automaticaly set to lowest_depth")
                output_option="lowest_depth"
            self.cexpchullDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:self.cexpchullDepth=np.empty((self.distRef.shape[0],x.shape[0]))
        self._check_variables(
            x=x,NRandom =NRandom,output_option =output_option,n_refinements =n_refinements,
            sphcap_shrink=sphcap_shrink,alpha_Dirichlet =alpha_Dirichlet,
            cooling_factor=cooling_factor,cap_size =cap_size,
        ) # check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number 
        if option>=2:
            if evaluate_dataset:self.cexpchullDirDS=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            else:self.cexpchullDir=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            if option>=3:
                self.allDepth=np.empty((self.distRef.shape[0],x.shape[0],NRandom))
            if option==4:
                self.allDirections=np.empty((self.distRef.shape[0],x.shape[0],NRandom,x.shape[1]))
        
        for ind,d in enumerate(self.distRef):
            DC=mtv.cexpchull(
                x=x, data=self.data[self.distribution==d],solver=solver,NRandom=NRandom,option=option,n_refinements=n_refinements,
                sphcap_shrink=sphcap_shrink,alpha_Dirichlet =alpha_Dirichlet,cooling_factor=cooling_factor,
                cap_size =cap_size,start =start,space =space,line_solver =line_solver,bound_gc =bound_gc,
                ) # compute depth 
            if evaluate_dataset==False: 
                if option==1:self.cexpchullDepth[ind]=DC # assign value
                elif option==2:self.cexpchullDepth[ind],self.cexpchullDir[ind]=DC # assign value
                elif option==3:self.cexpchullDepth[ind],self.cexpchullDir[ind],self.allDepth[ind]=DC # assign value
                elif option==4:self.cexpchullDepth[ind],self.cexpchullDir[ind],self.allDepth[ind],self.allDirections[ind],_=DC # assign value
            if evaluate_dataset==True: 
                if option==1:self.cexpchullDepthDS[ind]=DC # assign value
                elif option==2:self.cexpchullDepthDS[ind],self.cexpchullDirDS[ind]=DC # assign value


        if self.distRef.shape[0]==1: #fix for one distribution
            if evaluate_dataset:
                self.cexpchullDepthDS=self.cexpchullDepthDS[0]
                if option==2:self.cexpchullDirDS=self.cexpchullDirDS[0]
            else:
                self.cexpchullDepth=self.cexpchullDepth[0]
                if option>=2:self.cexpchullDir=self.cexpchullDir[0]
            if option>=3:self.allDepth=self.allDepth[0]
            if option>=4:self.allDirections=self.allDirections[0]
        if evaluate_dataset==False: # return correct value
            if option==1:return self.cexpchullDepth
            if option==2:return self.cexpchullDepth,self.cexpchullDir
            if option==3:return self.cexpchullDepth,self.cexpchullDir,self.allDepth
            if option==4:return self.cexpchullDepth,self.cexpchullDir,self.allDepth,self.allDirections
        elif evaluate_dataset==True:
            if option==1:return self.cexpchullDepthDS
            if option==2:return self.cexpchullDepthDS,self.cexpchullDirDS

        
    def cexpchullstar(self,x: np.ndarray|None=None, solver: str = "neldermead", NRandom: int = 1000, 
        option: int = 1, n_refinements: int = 10, sphcap_shrink: float = 0.5, 
        alpha_Dirichlet: float = 1.25, cooling_factor: float = 0.95, cap_size: int = 1,
        start: str = "mean", space: str = "sphere", line_solver: str = "goldensection", bound_gc: bool = True,
        output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir", evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            if output_option=="all_depth"or output_option=="all_depth_directions":
                print(f"output_option is set to {output_option}, only possible for lowest_depth or final_depth_dir, \
                      automaticaly set to lowest_depth")
                output_option="lowest_depth"
            self.cexpchullstarDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:self.cexpchullstarDepth=np.empty((self.distRef.shape[0],x.shape[0]))
        self._check_variables(x=x,NRandom=NRandom, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink,
                            alpha_Dirichlet=alpha_Dirichlet, cooling_factor= cooling_factor, cap_size=cap_size,
                              ) # check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number 
        if option>=2:
            if evaluate_dataset:self.cexpchullstarDirDS=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            else:self.cexpchullstarDir=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            if option>=3:
                self.allDepth=np.empty((self.distRef.shape[0],x.shape[0],NRandom))
            if option==4:
                self.allDirections=np.empty((self.distRef.shape[0],x.shape[0],NRandom,x.shape[1]))
        for ind,d in enumerate(self.distRef):
            DC=mtv.cexpchullstar(x=x,data=self.data[self.distribution==d], solver=solver, NRandom=NRandom, option=option, n_refinements=n_refinements, 
                            sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
                            cap_size=cap_size,start=start, space=space, line_solver=line_solver, bound_gc=bound_gc)
            if evaluate_dataset==False:
                if option==1:self.cexpchullstarDepth[ind]=DC # assign value
                elif option==2:self.cexpchullstarDepth[ind],self.cexpchullstarDir[ind]=DC # assign value
                elif option==3:self.cexpchullstarDepth[ind],self.cexpchullstarDir[ind],self.allDepth[ind]=DC # assign value
                elif option==4:self.cexpchullstarDepth[ind],self.cexpchullstarDir[ind],self.allDepth[ind],self.allDirections[ind],_=DC # assign value
            if evaluate_dataset==True:
                if option==1:self.cexpchullstarDepthDS[ind]=DC # assign value
                elif option==2:self.cexpchullstarDepthDS[ind],self.cexpchullstarDirDS[ind]=DC # assign value
        
        if self.distRef.shape[0]==1: #fix for one distribution
            if evaluate_dataset:
                self.cexpchullstarDepthDS=self.cexpchullstarDepthDS[0]
                if option==2:self.cexpchullstarDirDS=self.cexpchullstarDirDS[0]
            else:
                self.cexpchullstarDepth=self.cexpchullstarDepth[0]
                if option>=2:self.cexpchullstarDir=self.cexpchullstarDir[0]
            if option>=3:self.allDepth=self.allDepth[0]
            if option>=4:self.allDirections=self.allDirections[0]
        if evaluate_dataset==False: # return correct value
            if option==1:return self.cexpchullstarDepth
            if option==2:return self.cexpchullstarDepth,self.cexpchullstarDir
            if option==3:return self.cexpchullstarDepth,self.cexpchullstarDir,self.allDepth
            if option==4:return self.cexpchullstarDepth,self.cexpchullstarDir,self.allDepth,self.allDirections
        elif evaluate_dataset==True:
            if option==1:return self.cexpchullstarDepthDS
            if option==2:return self.cexpchullstarDepthDS,self.cexpchullstarDirDS
        
    def geometrical(self,x:np.ndarray|None=None,solver: str = "neldermead", NRandom: int = 1000, n_refinements: int = 10, 
                    sphcap_shrink: float = 0.5, alpha_Dirichlet: float = 1.25, cooling_factor: float = 0.95, 
                    cap_size: int = 1, start: str = "mean", space: str = "sphere", line_solver: str = "goldensection", bound_gc: bool = True,
                    output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir", evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            if output_option=="all_depth"or output_option=="all_depth_directions":
                print(f"output_option is set to {output_option}, only possible for lowest_depth or final_depth_dir, \
                      automaticaly set to lowest_depth")
                output_option="lowest_depth"
            self.geometricalDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:self.geometricalDepth=np.empty((self.distRef.shape[0],x.shape[0]))
        self._check_variables(
            x=x, NRandom=NRandom, n_refinements=n_refinements, sphcap_shrink=sphcap_shrink, 
            alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor,cap_size=cap_size,
            )# check if parameters are valid
        option=self._determine_option(x,NRandom,output_option) # determine option number 
        if option>=2:
            if evaluate_dataset:self.geometricalDirDS=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            else:self.geometricalDir=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            if option>=3:
                self.allDepth=np.empty((self.distRef.shape[0],x.shape[0],NRandom))
            if option==4:
                self.allDirections=np.empty((self.distRef.shape[0],x.shape[0],NRandom,x.shape[1]))
        for ind,d in enumerate(self.distRef):
            DG=mtv.geometrical(x=x,data=self.data[self.distribution==d], solver=solver, NRandom=NRandom, option=option, n_refinements=n_refinements, 
                            sphcap_shrink=sphcap_shrink, alpha_Dirichlet=alpha_Dirichlet, cooling_factor=cooling_factor, 
                            cap_size=cap_size,start=start, space=space, line_solver=line_solver, bound_gc=bound_gc)
            if evaluate_dataset==False:
                if option==1:self.geometricalDepth[ind]=DG # assign value
                elif option==2:self.geometricalDepth[ind],self.geometricalDir[ind]=DG # assign value
                elif option==3:self.geometricalDepth[ind],self.geometricalDir[ind],self.allDepth[ind]=DG # assign value
                elif option==4:self.geometricalDepth[ind],self.geometricalDir[ind],self.allDepth[ind],self.allDirections[ind],_=DG # assign value
            if evaluate_dataset==True:
                if option==1:self.geometricalDepthDS[ind]=DG # assign value
                elif option==2:self.geometricalDepthDS[ind],self.geometricalDirDS[ind]=DG # assign value
        

        if self.distRef.shape[0]==1: #fix for one distribution
            if evaluate_dataset:
                self.geometricalDepthDS=self.geometricalDepthDS[0]
                if option==2:self.geometricalDirDS=self.geometricalDirDS[0]
            else:
                self.geometricalDepth=self.geometricalDepth[0]
                if option>=2:self.geometricalDir=self.geometricalDir[0]
            if option>=3:self.allDepth=self.allDepth[0]
            if option>=4:self.allDirections=self.allDirections[0]
        if evaluate_dataset==False: # return correct value
            if option==1:return self.geometricalDepth
            if option==2:return self.geometricalDepth,self.geometricalDir
            if option==3:return self.geometricalDepth,self.geometricalDir,self.allDepth
            if option==4:return self.geometricalDepth,self.geometricalDir,self.allDepth,self.allDirections
        elif evaluate_dataset==True:
            if option==1:return self.geometricalDepthDS
            if option==2:return self.geometricalDepthDS,self.geometricalDirDS

    def halfspace(self, x:np.ndarray|None=None,exact: bool = True,method: str = "recursive",solver: str = "neldermead",
                  NRandom: int = 1000,n_refinements: int = 10,sphcap_shrink: float = 0.5,alpha_Dirichlet: float = 1.25,cooling_factor: float = 0.95,
                  cap_size: int = 1,start: str = "mean",space: str = "sphere",line_solver: str = "goldensection",bound_gc: bool = True,
                  CUDA:bool=False,output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir", evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            if output_option=="all_depth"or output_option=="all_depth_directions":
                print(f"output_option is set to {output_option}, only possible for lowest_depth or final_depth_dir, \
                      automaticaly set to lowest_depth")
                output_option="lowest_depth"
            self.halfspaceDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:self.halfspaceDepth=np.empty((self.distRef.shape[0],x.shape[0]))
        CUDA=self._check_CUDA(CUDA,solver)
        if CUDA:exact=False
        self._check_variables(x=x,NRandom=NRandom,
                              n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                              alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,) # check if parameters are valid
        option=self._determine_option(x,NRandom,output_option, CUDA=CUDA,exact=exact) # determine option number
        if option>=2:
            if evaluate_dataset:self.halfspaceDirDS=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            else:self.halfspaceDir=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            if option>=3:
                self.allDepth=np.empty((self.distRef.shape[0],x.shape[0],NRandom))
            if option==4:
                self.allDirections=np.empty((self.distRef.shape[0],x.shape[0],NRandom,x.shape[1]))

        for ind,d in enumerate(self.distRef):
            if CUDA:DH=mtv.halfspace(x=x,data=self.dataCuda[:,self.distribution==d],exact=exact,method=method,
                solver=solver,NRandom=NRandom,option=option,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,
                space=space,line_solver=line_solver,bound_gc=bound_gc,CUDA=CUDA,
            )
            elif CUDA==False:DH=mtv.halfspace(x=x,data=self.data[self.distribution==d],exact=exact,method=method,
                solver=solver,NRandom=NRandom,option=option,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,
                space=space,line_solver=line_solver,bound_gc=bound_gc,CUDA=CUDA,
            )
            if evaluate_dataset==False:
                if option==1 or exact==True:self.halfspaceDepth[ind]=DH # assign value
                elif option==2:self.halfspaceDepth[ind],self.halfspaceDir[ind]=DH # assign value
                elif option==3:self.halfspaceDepth[ind],self.halfspaceDir[ind],self.allDepth[ind]=DH # assign value
                elif option==4:self.halfspaceDepth[ind],self.halfspaceDir[ind],self.allDepth[ind],self.allDirections[ind],_=DH # assign value
            if evaluate_dataset==True:
                if option==1:self.halfspaceDepthDS[ind]=DH # assign value
                elif option==2:self.halfspaceDepthDS[ind],self.halfspaceDirDS[ind]=DH # assign value

        if self.distRef.shape[0]==1: #fix for one distribution
            if evaluate_dataset:
                self.halfspaceDepthDS=self.halfspaceDepthDS[0]
                if option==2:self.halfspaceDirDS=self.halfspaceDirDS[0]
            else:
                self.halfspaceDepth=self.halfspaceDepth[0]
                if option>=2:self.halfspaceDir=self.halfspaceDir[0]
            if option>=3:self.allDepth=self.allDepth[0]
            if option>=4:self.allDirections=self.allDirections[0]
        if evaluate_dataset==False: # return correct value
            if option==1:return self.halfspaceDepth
            if option==2:return self.halfspaceDepth,self.halfspaceDir
            if option==3:return self.halfspaceDepth,self.halfspaceDir,self.allDepth
            if option==4:return self.halfspaceDepth,self.halfspaceDir,self.allDepth,self.allDirections
        elif evaluate_dataset==True:
            if option==1:return self.halfspaceDepthDS
            if option==2:return self.halfspaceDepthDS,self.halfspaceDirDS

    
    def L2(self,x: np.ndarray|None=None, mah_estimate: str = 'moment', mah_parMcd: float = 0.75, evaluate_dataset:bool=False)->np.ndarray:
        """
        Compute L2 depth 
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        L2 depth : {array like}
        """
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            self.L2DepthDS=np.zeros((self.distRef.shape[0], x.shape[0]))
        else: # create self 
            self.L2Depth=np.zeros((self.distRef.shape[0], x.shape[0])) 
        self._check_variables(x=x,mah_estimate=mah_estimate, mah_parMcd=mah_parMcd) # check if parameters are valid
        for ind,d in enumerate(self.distRef): # run distributions
            DL2=mtv.L2(x=x,data=self.data[self.distribution==d],
                    mah_estimate=mah_estimate, mah_parMcd=mah_parMcd)
            if evaluate_dataset:self.L2DepthDS[ind]=DL2
            else:self.L2Depth[ind]=DL2
        if self.distRef.shape[0]==1: # Fix size
            if evaluate_dataset==True: self.L2DepthDS=self.L2DepthDS[0] 
            else: self.L2Depth=self.L2Depth[0]
        return self.L2DepthDS if evaluate_dataset==True else self.L2Depth

    def potential(self,x:np.ndarray|None=None,pretransform: str = "1Mom", kernel: str = "EDKernel", 
                  mah_parMcd: float = 0.75, kernel_bandwidth: int = 0, evaluate_dataset:bool=False)->np.ndarray:
        """
        Compute potential depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth
        
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

        Results
        ----------
        Potential depth : {array like}
        """
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            self.potentialDepthDS=np.zeros((self.distRef.shape[0], x.shape[0]))
        else: # create self 
            self.potentialDepth=np.zeros((self.distRef.shape[0], x.shape[0])) 
        self._check_variables(x=x,mah_parMcd=mah_parMcd)# check if parameters are valid
        #x: Any, data: Any, pretransform: str = "1Mom", kernel: str = "EDKernel", mah_parMcd: float = 0.75, kernel_bandwidth: int = 0
        for ind,d in enumerate(self.distRef):
            DP=mtv.potential(x=x, data=self.data[self.distribution==d], pretransform=pretransform, kernel=kernel, mah_parMcd=mah_parMcd, kernel_bandwidth=kernel_bandwidth)
            if evaluate_dataset==True: # Dataset evaluation
                self.potentialDepthDS[ind]=DP
            else:self.potentialDepth[ind]=DP
        if self.distRef.shape[0]==1: # Fix size
            if evaluate_dataset==True:self.potentialDepthDS=self.potentialDepthDS[0] 
            else: self.potentialDepth=self.potentialDepth[0]
        return self.potentialDepthDS if evaluate_dataset==True else self.potentialDepth
        

    
    def projection(self,x:np.ndarray|None=None,solver: str = "neldermead",NRandom: int = 1000,n_refinements: int = 10,
                  sphcap_shrink: float = 0.5,alpha_Dirichlet: float = 1.25,cooling_factor: float = 0.95,
                  cap_size: int = 1,start: str = "mean",space: str = "sphere",line_solver: str = "goldensection",bound_gc: bool = True,
                  CUDA:bool=False, output_option:Literal["lowest_depth","final_depht_dir",
                                          "all_depth","all_depth_directions"]="final_depht_dir", evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            if output_option=="all_depth"or output_option=="all_depth_directions":
                print(f"output_option is set to {output_option}, only possible for lowest_depth or final_depth_dir, \
                      automaticaly set to lowest_depth")
                output_option="lowest_depth"
            self.projectionDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:self.projectionDepth=np.empty((self.distRef.shape[0],x.shape[0]))

        self._check_variables(x=x,NRandom=NRandom,
                              n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                              alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,) # check if parameters are valid
        CUDA=self._check_CUDA(CUDA,solver)
        option=self._determine_option(x,NRandom,output_option,CUDA) # determine option number

        if option>=2:
            if evaluate_dataset:self.projectionDirDS=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            else:self.projectionDir=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            if option>=3:
                self.allDepth=np.empty((self.distRef.shape[0],x.shape[0],NRandom))
            if option==4:
                self.allDirections=np.empty((self.distRef.shape[0],x.shape[0],NRandom,x.shape[1]))
        for ind,d in enumerate(self.distRef):
            if CUDA:DP=mtv.projection(x=x,data=self.dataCuda[:,self.distribution==d],solver=solver,NRandom=NRandom,option=option,
                            n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,
                            space=space,line_solver=line_solver,bound_gc=bound_gc,CUDA=CUDA
            )
            else:DP=mtv.projection(x=x,data=self.data[self.distribution==d],solver=solver,NRandom=NRandom,option=option,
                            n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,
                            space=space,line_solver=line_solver,bound_gc=bound_gc,CUDA=CUDA
            )
            if evaluate_dataset==False:
                if option==1:self.projectionDepth[ind]=DP # assign value
                elif option==2:self.projectionDepth[ind],self.projectionDir[ind]=DP # assign value
                elif option==3:self.projectionDepth[ind],self.projectionDir[ind],self.allDepth[ind]=DP # assign value
                elif option==4:self.projectionDepth[ind],self.projectionDir[ind],self.allDepth[ind],self.allDirections[ind],_=DP # assign value
            if evaluate_dataset==True:
                if option==1:self.projectionDepthDS[ind]=DP # assign value
                elif option==2:self.projectionDepthDS[ind],self.projectionDirDS[ind]=DP # assign value

        if self.distRef.shape[0]==1: #fix for one distribution
            if evaluate_dataset:
                self.projectionDepthDS=self.projectionDepthDS[0]
                if option==2:self.projectionDirDS=self.projectionDirDS[0]
            else:
                self.projectionDepth=self.projectionDepth[0]
                if option>=2:self.projectionDir=self.projectionDir[0]
            if option>=3:self.allDepth=self.allDepth[0]
            if option>=4:self.allDirections=self.allDirections[0]
        if evaluate_dataset==False: # return correct value
            if option==1:return self.projectionDepth
            if option==2:return self.projectionDepth,self.projectionDir
            if option==3:return self.projectionDepth,self.projectionDir,self.allDepth
            if option==4:return self.projectionDepth,self.projectionDir,self.allDepth,self.allDirections
        elif evaluate_dataset==True:
            if option==1:return self.projectionDepthDS
            if option==2:return self.projectionDepthDS,self.projectionDirDS
        
    def qhpeeling(self,x:np.ndarray|None=None, evaluate_dataset:bool=False)->np.ndarray:
        """
        Calculates the convex hull peeling depth.
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth

        Results
        ----------
        Convex hull peeling depth : {array like}
        """
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            self.qhpeelingDepthDS=np.zeros((self.distRef.shape[0], x.shape[0]))
        else: # create self 
            self.qhpeelingDepth=np.zeros((self.distRef.shape[0], x.shape[0])) 
        self._check_variables(x=x)# check if parameters are valid
        for ind,d in enumerate(self.distRef): # run distribution
            DQ=mtv.qhpeeling(x=x,data=self.data[self.distribution==d])
            if evaluate_dataset==True:self.qhpeelingDepthDS[ind]=DQ
            if evaluate_dataset==False:self.qhpeelingDepth[ind]=DQ
        if self.distRef.shape[0]==1: # Fix size
            if evaluate_dataset==True: self.qhpeelingDepthDS=self.qhpeelingDepthDS[0] 
            else: self.qhpeelingDepth=self.qhpeelingDepth[0]
        return self.qhpeelingDepthDS if evaluate_dataset==True else self.qhpeelingDepth

    def simplicial(self,x:np.ndarray|None=None,exact:bool=True,k:float=0.05,evaluate_dataset:bool=False)->np.ndarray:
        """
        Compute simplicial depth.
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth
            
        k: float, default=0.05
            Number (``k > 1``) or portion (if ``0 < k < 1``) of simplices that are considered if ``exact=False``.
            If ``k > 1``, then the algorithmic complexity is polynomial in d but is independent of the number of observations in data, given k. 
            If ``0 < k < 1``,then the algorithmic complexity is exponential in the number of observations in data, 
                but the calculation precision stays approximately the same.

        Results
        ----------
        Simplicial depth : {array like}
        """
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            self.simplicialDepthDS=np.zeros((self.distRef.shape[0], x.shape[0]))
        else: # create self 
            self.simplicialDepth=np.zeros((self.distRef.shape[0], x.shape[0])) 
        self._check_variables(x=x)# check if parameters are valid
        for ind,d in enumerate(self.distRef):
            DS=mtv.simplicial(x=x,data=self.data[self.distribution==d],exact=exact,k=k,seed=self.seed)
            if evaluate_dataset==False:
                self.simplicialDepth[ind]=DS
            if evaluate_dataset==True:
                self.simplicialDepthDS[ind]=DS
        if self.distRef.shape[0]==1: # Fix size
            if evaluate_dataset==True: self.simplicialDepthDS=self.simplicialDepthDS[0] 
            else: self.simplicialDepth=self.simplicialDepth[0]
        return self.simplicialDepthDS if evaluate_dataset==True else self.simplicialDepth
            
         
    def simplicialVolume(self,x:np.ndarray|None=None,exact: bool = True, k: float = 0.05, 
                         mah_estimate: str = "moment", mah_parMCD: float = 0.75,
                         evaluate_dataset:bool=False)->np.ndarray:
        """
        Compute simplicial volume depth
                
        Parameters
        ----------
        x : {array-like} of shape (n_samples,d).
            Samples matrix to compute depth
        
        k: float, default=0.05
            Number (``k > 1``) or portion (if ``0 < k < 1``) of simplices that are considered if ``exact=False``.
            If ``k > 1``, then the algorithmic complexity is polynomial in d but is independent of the number of observations in data, given k. 
            If ``0 < k < 1``,then the algorithmic complexity is exponential in the number of observations in data, 
                but the calculation precision stays approximately the same.

        Results
        ----------
        Simplicial volume depth : {array like}
        """
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            self.simplicialVolumeDepthDS=np.zeros((self.distRef.shape[0], x.shape[0]))
        else: # create self 
            self.simplicialVolumeDepth=np.zeros((self.distRef.shape[0], x.shape[0])) 
        self._check_variables(x=x)# check if parameters are valid
        for ind,d in enumerate(self.distRef):
            DS=mtv.simplicialVolume(x=x,data=self.data[self.distribution==d],
                                    exact=exact,k=k,mah_estimate=mah_estimate,mah_parMCD=mah_parMCD,seed=self.seed)
            if evaluate_dataset==True:self.simplicialVolumeDepthDS[ind]=DS
            elif evaluate_dataset==False:self.simplicialVolumeDepth[ind]=DS
        if self.distRef.shape[0]==1: # Fix size
            if evaluate_dataset==True: self.simplicialVolumeDepthDS=self.simplicialVolumeDepthDS[0] 
            else: self.simplicialVolumeDepth=self.simplicialVolumeDepth[0]
        return self.simplicialVolumeDepthDS if evaluate_dataset==True else self.simplicialVolumeDepth
        
    def spatial(self,x:np.ndarray|None=None,mah_estimate:str='moment',mah_parMcd:float=0.75,
                evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            self.spatialDepthDS=np.zeros((self.distRef.shape[0], x.shape[0]))
        else: # create self 
            self.spatialDepth=np.zeros((self.distRef.shape[0], x.shape[0]))
        self._check_variables(x=x,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd) # check if parameters are valid
        for ind,d in enumerate(self.distRef):
            DS=mtv.spatial(x,self.data[self.distribution==d],mah_estimate=mah_estimate,mah_parMcd=mah_parMcd)
            if evaluate_dataset==False:self.spatialDepth[ind]=DS
            if evaluate_dataset==True:self.spatialDepthDS[ind]=DS
        if self.distRef.shape[0]==1: # Fix size
            if evaluate_dataset==True: self.spatialDepthDS=self.spatialDepthDS[0] 
            else: self.spatialDepth=self.spatialDepth[0]
        return self.spatialDepthDS if evaluate_dataset==True else self.spatialDepth
        
        
    def zonoid(self,x:np.ndarray|None=None, exact:bool=True,
               solver="neldermead",NRandom=1000,n_refinements=10,
               sphcap_shrink=0.5,alpha_Dirichlet=1.25,cooling_factor=0.95,cap_size=1,
               start="mean",space="sphere",line_solver="goldensection",bound_gc=True,
               output_option:Literal["lowest_depth","final_depht_dir",
                                     "all_depth","all_depth_directions"]="final_depht_dir",
                evaluate_dataset:bool=False)->np.ndarray:
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
        if evaluate_dataset==True: # Dataset evaluation
            print("x value is set to the loaded dataset")
            x=self.data
            if output_option=="all_depth"or output_option=="all_depth_directions":
                print(f"output_option is set to {output_option}, only possible for lowest_depth or final_depth_dir, \
                      automaticaly set to lowest_depth")
                output_option="lowest_depth"
            self.zonoidDepthDS=np.empty((self.distRef.shape[0],x.shape[0]))
        else:self.zonoidDepth=np.empty((self.distRef.shape[0],x.shape[0]))

        self._check_variables(x=x,exact=exact, 
            NRandom=NRandom,n_refinements=n_refinements,
            sphcap_shrink=sphcap_shrink,alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,
            cap_size=cap_size,output_option=output_option) # check if parameters are valid
        
        # seedZ=seed if seed!=self.seed else self.seed #set seed value to default if seed is not passed
        option=self._determine_option(x,NRandom,output_option,exact=exact) # determine option number
        if option>=2:
            if evaluate_dataset:self.zonoidDirDS=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            else:self.zonoidDir=np.empty((self.distRef.shape[0],x.shape[0],x.shape[1]))
            if option>=3:
                self.allDepth=np.empty((self.distRef.shape[0],x.shape[0],NRandom))
            if option==4:
                self.allDirections=np.empty((self.distRef.shape[0],x.shape[0],NRandom,x.shape[1]))
        for ind,d in enumerate(self.distRef):
            DZ=mtv.zonoid(
                x,self.data[self.distribution==d],seed=self.seed,exact=exact, 
                solver=solver,NRandom=NRandom,n_refinements=n_refinements,
                sphcap_shrink=sphcap_shrink,alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,
                cap_size=cap_size,start=start,space=space,line_solver=line_solver,
                bound_gc=bound_gc,option=option) # compute zonoid depth
            if evaluate_dataset==False:
                if exact or option==1:self.zonoidDepth[ind]=DZ # assign value
                elif option==2:self.zonoidDepth[ind],self.zonoidDir[ind]=DZ # assign value
                elif option==3:self.zonoidDepth[ind],self.zonoidDir[ind],self.allDepth[ind]=DZ # assign value
                elif option==4:self.zonoidDepth[ind],self.zonoidDir[ind],self.allDepth[ind],self.allDirections[ind],_=DZ # assign value
            if evaluate_dataset==True:
                if exact or option==1:self.zonoidDepthDS[ind]=DZ # assign value
                elif option==2:self.zonoidDepthDS[ind],self.zonoidDirDS[ind]=DZ # assign value
        
        
        if self.distRef.shape[0]==1: #fix for one distribution
            if evaluate_dataset:
                self.zonoidDepthDS=self.zonoidDepthDS[0]
                if option==2:self.zonoidDirDS=self.zonoidDirDS[0]
            else:
                self.zonoidDepth=self.zonoidDepth[0]
                if option>=2:self.zonoidDir=self.zonoidDir[0]
                if option>=3:self.allDepth=self.allDepth[0]
                if option>=4:self.allDirections=self.allDirections[0]
        if evaluate_dataset==False: # return correct value
            if option==1:return self.zonoidDepth
            if option==2:return self.zonoidDepth,self.zonoidDir
            if option==3:return self.zonoidDepth,self.zonoidDir,self.allDepth
            if option==4:return self.zonoidDepth,self.zonoidDir,self.allDepth,self.allDirections
        elif evaluate_dataset==True:
            if option==1:return self.zonoidDepthDS
            if option==2:return self.zonoidDepthDS,self.zonoidDirDS

    def ACA(self,dim:int=2,
            sample_size: None = None,
            sample: None = None,
            notion: str = "projection",
            solver: str = "neldermead",
            NRandom: int = 100,
            n_refinements: int = 10,
            sphcap_shrink: float = 0.5,
            alpha_Dirichlet: float = 1.25,
            cooling_factor: float = 0.95,
            cap_size: int = 1,
            start: str = "mean",
            space: str = "sphere",
            line_solver: str = "goldensection",
            bound_gc: bool = True):
        """
        Computes the abnormal component analysis
            
        Parameters
        ----------
        dim: int, default=2
            Number of dimensions to keep in the reduction
        
        sample_size: int, default=None
            Size of the dataset (uniform sampling) to be used in the ACA calculation

        sample: list[int], default=None
            Indices for the dataset to be used in the computation 
        
        notion: str, default="projection"
            Chosen notion for depth computation

        Results
        --------
            ACA directions for dimensional reduction
            
        """
        ACA_tab=mtv.ACA(X=self.data,dim=dim,
                        sample_size=sample_size,
                        sample=sample,
                        notion=notion,
                        solver=solver,
                        NRandom=NRandom,
                        n_refinements=n_refinements,
                        sphcap_shrink=sphcap_shrink,
                        alpha_Dirichlet=alpha_Dirichlet,
                        cooling_factor=cooling_factor,
                        cap_size=cap_size,
                        start=start,
                        space=space,
                        line_solver=line_solver,
                        bound_gc=bound_gc)
        return ACA_tab

    ## Det and MCD 
    def _calcDet(self,mat:np.ndarray):
        """
        Computes the determinant of a matrix 

        Parametres 
        -----------
        mat: {array-like}
            Matrix to compute the determinant

        Results
        -----------
        Det: float
            determinant of the matrix
        """
        # self._check_variables
        return mtv.calcDet(mat)
        
    def computeMCD(self,mat:np.ndarray|None=None, h:int|float=1, mfull: int = 10, nstep: int = 7, hiRegimeCompleteLastComp: bool = True)->None:
        """
        Compute Minimum Covariance Determinant (MCD)

        Parametres 
        -----------
        mat: {array-like} or None, default=None
            Matrix to compute MCD. If set to None, compute the MCD of the loaded dataset

        h: int or float, default=1
            Represents the amount of data of the dataset used to compute the MCD. 
            If the value is in the interval [0,1], it is treated as the percentage of dataset,
            if the value is in the interval [n/2,n], it is treated as the amount of sample points.
            It in the interval ]1,n/2[, the amount is rounded to n/2.
        
        mfull: int, default=10

        nstep: int, default=7
            Amount of steps to compute MCD

        hiRegimeCompleteLastComp: bool, default=True
            
        Results 
        -----------
        Minimum Covariance Determinant (MCD): {array-like}
        """
        self._check_variables(h) # check if h is in the acceptable range
        if h>0 and h<=1: # transform h in the good value for MCD function
            h=int(h*self._nSamples)
        elif h<self._nSamples/2:
            h=int(self._nSamples/2)
        else:h=int(h)
        self.MCD=mtv.MCD(self.data,h=h,seed=self.seed,mfull=mfull, nstep=nstep, hiRegimeCompleteLastComp=hiRegimeCompleteLastComp)
        return self.MCD
    
    def change_dataset(self,newDataset:np.ndarray,newY:np.ndarray|None=None, newDistribution:np.ndarray|None=None,keepOld:bool=False,)->None:
        """

        Description
        ------------
            Modify loaded dataset.

        Arguments
        ---------
            newDataset:np.ndarray
                New dataset
            
            newDistribution:np.ndarray|None, default=None,
                Distribution related to the dataset
            
            newY:np.ndarray|None, default=None,
                Only for convention.
            
            keepOld:bool, default=False,
                Boolean to determine if current dataset is kept or not.
                If True, newDataset is added in the end of the old one.
        Returns 
        -------
            None
        """
        if keepOld: # keep old dataset
            if self.data.shape[1]!=newDataset.shape[1]:
                raise Exception(f"Dimensions must be the same, current dimension is {self.data.shape[1]} and new dimension is {newDataset.shape[1]}")
            self.data=np.concatenate((self.data,newDataset), axis=0)
            try:self.y=np.concatenate((self.y,newY)) # try for y
            except:pass
            try:
                self.distribution=np.concatenate((self.distribution,newDistribution)) # try for distribution
                self.distRef=np.unique(self.distribution)
            except:
                self.distribution=np.concatenate((self.distribution,np.repeat(0,newDataset.shape[0]))) # try for distribution
        else:
            self.data=newDataset
            try:self.y=newY # try for y
            except:pass
            try:
                self.distribution=newDistribution # try for distribution
                self.distRef=np.unique(self.distribution)
            except:
                self.distribution=np.repeat(0,newDataset.shape[0])
                self.distRef=np.unique(self.distribution)
        return self
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
        # depth and directions for dataset
        self.cexpchullDirDS,self.cexpchullDepthDS=None, None
        self.cexpchullstarDirDS,self.cexpchullstarDepthDS=None, None
        self.geometricalDirDS,self.geometricalDepthDS=None, None
        self.halfspaceDirDS,self.halfspaceDepthDS=None, None
        self.mahalanobisDirDS,self.mahalanobisDepthDS=None, None
        self.projectionDirDS,self.projectionDepthDS=None, None
        self.aprojectionDirDS,self.aprojectionDepthDS=None,None
        self.zonoidDirDS,self.zonoidDepthDS=None,None
        self.potentialDepthDS=None
        self.qhpeelingDepthDS,self.betaSkeletonDepthDS,self.L2DepthDS=None,None,None
        self.simplicialVolumeDepthDS,self.simplicialDepthDS,self.spatialDepthDS=None,None,None
        # MCD
        self.MCD=None
        # approximate depth and direction
        # self.allDepth,self.allDirections,self.dirIndiex=None,None,None
    def _determine_option(self,x:np.ndarray,NRandom:int,output_option:str,CUDA:bool=False, exact:bool=False)->int:
        """Determine which is the option number (following the 1 to 4 convention), 
        with a created criteria to compute option 4 - all depths and directions - of 1Gb to the direction matrix"""
        if exact==True: return 1 # only depth
        option=self.approxOption.index(output_option)+1 # define option for function return 
        memorySize=x.size*x.itemsize*NRandom*self.distRef.shape[0]//1048576 # compute an estimate of the memory amount used for option 4
        if type(self.distribution)==type(None):
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
                if value.lower() not in {"none", "moment", "mcd"}:
                    raise ValueError(f"Only mah_estimate possibilities are {{'none', 'moment', 'mcd'}}, got {value}.")
            if key in ["NRandom","n_refinements"]:
                assert type(value)==int, f"{key} must be an integer, got {type(value)}"
            if key in ["mah_parMcd","sphcap_shrink","alpha_Dirichlet","cooling_factor"]: 
                assert type(value)==float, f"{key} must be a float, got {type(value)}"
            if key=="cap_size":
                assert type(value)==float or type(value)==int, f"cap_size must be a float or integer, got {type(value)}"
            if key=="output_option":
                assert type(value)==str, f"output_option must be a str, got {type(value)}"
                if value not in self.approxOption: 
                    raise ValueError(f"Only output_option possibilities are {self.approxOption}, got {value}.")
            if key=="h":
                assert type(value)==int or type(value)==float, f"h must be a float or int, got {type(value)}"
                if value<=0 or value>self._nSamples: 
                    raise ValueError(f"h must be in the range from 0 to {self._nSamples}, got {value}.")

    def _check_CUDA(self,CUDA,solver):
        if solver not in ["simplerandom", "refinedrandom"] and CUDA==True:
            print(f"CUDA is only available for 'simplerandom', 'refinedrandom', solver is {solver}, CUDA is set to False")
            return False
        return CUDA
    
DepthEucl.mahalanobis.__doc__=docHelp.mahalanobis__doc__
DepthEucl.aprojection.__doc__=docHelp.aprojection__doc__
DepthEucl.betaSkeleton.__doc__=docHelp.betaSkeleton__doc__
DepthEucl.cexpchull.__doc__=docHelp.cexpchull__doc__
DepthEucl.cexpchullstar.__doc__=docHelp.cexpchullstar__doc__
DepthEucl.geometrical.__doc__=docHelp.geometrical__doc__
DepthEucl.halfspace.__doc__=docHelp.halfspace__doc__
DepthEucl.L2.__doc__=docHelp.L2__doc__
DepthEucl.potential.__doc__=docHelp.potential__doc__
DepthEucl.projection.__doc__=docHelp.projection__doc__
DepthEucl.qhpeeling.__doc__=docHelp.qhpeeling__doc__
DepthEucl.simplicial.__doc__=docHelp.simplicial__doc__
DepthEucl.simplicialVolume.__doc__=docHelp.simplicialVolume__doc__
DepthEucl.spatial.__doc__=docHelp.spatial__doc__
DepthEucl.zonoid.__doc__=docHelp.zonoid__doc__
DepthEucl.ACA.__doc__=docHelp.ACA__doc__
DepthEucl.change_dataset.__doc__=docHelp.change_dataset__doc__
    # depth_mesh.__doc__=mtv.depth_mesh.__doc__
    # depth_plot2d.__doc__=mtv.depth_plot2d.__doc__
    # _calcDet.__doc__=mtv.calcDet.__doc__
    
