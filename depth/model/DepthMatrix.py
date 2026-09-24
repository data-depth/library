
import numpy as np
from . import docHelp
from . import multivariate as mtv
from typing import Literal, List
try:import torch
except:torch=None
import sys, os
from matrix import *
import DepthEucl, DepthFunc

class DepthMatrix():
    """
    Matrix Data-Depth
    
    Return the depth of each sample w.r.t. a dataset, D(x,data) in a matrix space, using a chosen depth notion.

    Data depth computes the centrality (similarity, belongness) of a sample 'x' given a dataset 'data'.
    
    Notes
    -----
    Possible depth notions are : `mahalanobis`,`halfspace`,`zonoid`,`cexpchullstar`,`cexpchull`,`geometrical`,`potential`,`qhpeeling`,`simplicial`,`betaskeleton`,`L2`,`simplicialvolume`,`spatial`,`projection`,`aprojection`,`sprojection`
    Metric spaces : `Riemannian`, `logEuclidean`,`cholesky`, `Euclidean`, `rootEuclidean`
    
    There are two possibilits: 
    Pointwise depth:
        For each discretization matrix :
            - Extract the matrix `x[i, :, :]` (shape: d x d)
            - Compute the multivariate depth for a matrix relative to all matrices

    Integrated / functional depth:
        For each discretization point i = 1, ..., L:
                - Extract the data slice `data[:, i, :, :]` (shape: N_data x d x d)
                - Extract the query matrix `x[i, :, :]` (shape: d x d)
                - Compute the multivariate depth of the query matrix relative to the data slice
                - Average the results over all L time points
    """
    
    def __init__(self):
        self.set_seed()
        self._notionsDict={"potential":mtv.potential,
                             "qhpeeling":mtv.qhpeeling, 
                             "simplicial":mtv.simplicial,
                             "betaskeleton":mtv.betaSkeleton,
                             "L2":mtv.L2, 
                             "simplicialvolume":mtv.simplicialVolume,
                             "spatial":mtv.spatial,
                             "mahalanobis":mtv.mahalanobis, 
                             "halfspace":mtv.halfspace, 
                             "zonoid":mtv.zonoid,
                             "PorjBased":mtv.depth_approximation}
        self.DataLogeuclidean=None
        # self.ModelLogeuclidean=None
        self.DataCholesky=None
        # self.ModelCholesky=None
        self.DataEuclidean=None
        # self.ModelEuclidean=None
        self.DataRootEuclidean=None
        # self.ModelRootEuclidean=None
        
        
    
    
    def load_dataset(self,data:np.ndarray=None,distribution:np.ndarray=None, CUDA:bool=False,y:np.ndarray=None):
        """
        Load the dataset X for reference calculations. Depth is computed with respect to this dataset.

        Parameters
        ----------
        data : {array-like} of shape (n,d,d).
            Dataset that will be used for depth computation
        
        distribution : Ignored, default=None
            Not used, present for API consistency by convention.

        y : Ignored, default=None
            Not used, present for API consistency by convention.

        Returns
        -------
        self : DepthMatrix model object.
            Returns the instance itself.
        """
        if type(data)==None:
            raise Exception("You must load a dataset")
        assert(type(data)==np.ndarray), "The dataset must be a numpy array"
        if len(data.shape)==3: 
            self._compType="point"
            self.samples,self.dim,_=data.shape
        elif len(data.shape)==4: 
            self._compType="integrated"
            self.samples, self.timesteps,self.dim,_=data.shape
        else:
            raise ValueError(f"Shape of data must be 3 (n,d,d) for pointwise depth or 4 (n,t,d,d) for integrated depth")
        self.data=data
        
        return self
    
    
    def evaluate(self, matrix,notion='zonoid',metric ='riemannian', weights=None,**kwargs):
        
        self._checkSpaceMetric(metric=metric)
        if "exact" in kwargs.keys():exact=kwargs["exact"]
        else:exact=False
        notionDic=self._determine_depth_func(notion,exact)
        if self._compType=="point":
            if len(matrix.shape)==2:matrix=matrix.reshape(1,self.dim,self.dim)
            depth=self._pointDepth(matrix, notion, metric,notionDic,**kwargs)
        if self._compType=="integrated":
            if len(matrix.shape)==3:matrix=matrix.reshape(1,self.timesteps,self.dim,self.dim)
            depth=self._integratedDepth(matrix, notion, metric,notionDic, weights,**kwargs)
        return depth
    
    
    def _pointDepth(self,matrix,notion, metric,notionDic,**kwargs):
        depth=np.zeros(matrix.shape[0])
        solver,NRandom,n_refinements,sphcap_shrink,alpha_Dirichlet,cooling_factor,cap_size,start,space,\
            line_solver,bound_gc,exact,mah_estimate,mah_parMcd,beta,distance,Lp_p,method,pretransform,\
            kernel,kernel_bandwidth,k=self._check_hyperparDepth(**kwargs)
        if metric=='riemannian':
            for i in range(matrix.shape[0]):
                z=np.zeros((self.dim*self.dim))
                dataR=self._applyLogmECoeff(matrix[i], self.data)
                dpt,state=self._notionsDict[notionDic](z,dataR,notion=notion,
                                             solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                                             alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,
                                             line_solver=line_solver,bound_gc=bound_gc,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
                                             beta=beta,distance=distance,Lp_p=Lp_p,method=method,pretransform=pretransform,
                                             kernel=kernel,kernel_bandwidth=kernel_bandwidth,k=k,state=self.RNG.bit_generator.state)
                depth[i]+=dpt
                self.RNG.bit_generator.state=state
                
        if metric=="logeuclidean":
            if type(self.DataLogeuclidean)==type(None):
                self.DataLogeuclidean=self._applyLogmECoeff(np.eye(self.dim), self.data)
                # self.ModelLogeuclidean=DepthEucl().load_dataset(self.DataLogeuclidean)
            z=self._applyLogmECoeff(np.eye(self.dim), matrix)
            depth,state=self._notionsDict[notionDic](z,self.DataLogeuclidean,notion=notion,
                                            solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                                            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,
                                            line_solver=line_solver,bound_gc=bound_gc,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
                                            beta=beta,distance=distance,Lp_p=Lp_p,method=method,pretransform=pretransform,
                                            kernel=kernel,kernel_bandwidth=kernel_bandwidth,k=k,state=self.RNG.bit_generator.state)
            self.RNG.bit_generator.state=state
            
            
        if metric=="cholesky":
            if type(self.DataCholesky)==type(None):
                # self.DataCholesky=self._ECoeff(self.DataCholesky)
                pass
            print("Not implemented")
        if metric=="euclidean":
            if type(self.DataEuclidean)==type(None):
                self.DataEuclidean=self._ECoeff(self.DataEuclidean)
            z=self._ECoeff(matrix)
            depth,state=self._notionsDict[notionDic](z,self.DataEuclidean,notion=notion,
                                            solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                                            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,
                                            line_solver=line_solver,bound_gc=bound_gc,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
                                            beta=beta,distance=distance,Lp_p=Lp_p,method=method,pretransform=pretransform,
                                            kernel=kernel,kernel_bandwidth=kernel_bandwidth,k=k,state=self.RNG.bit_generator.state)
            self.RNG.bit_generator.state=state
            
        if metric=="rooteuclidean":
            if type(self.DataRootEuclidean)==type(None):
                self.DataRootEuclidean=self._ECoeff(np.sqrt(self.DataRootEuclidean))
            z=self._ECoeff(np.sqrt(matrix))
            depth,state=self._notionsDict[notionDic](z,self.DataRootEuclidean,notion=notion,
                                            solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                                            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,
                                            line_solver=line_solver,bound_gc=bound_gc,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
                                            beta=beta,distance=distance,Lp_p=Lp_p,method=method,pretransform=pretransform,
                                            kernel=kernel,kernel_bandwidth=kernel_bandwidth,k=k,state=self.RNG.bit_generator.state)
            self.RNG.bit_generator.state=state
        
        return depth

    def _integratedDepth(self,matrix,notion, metric,notionDic, weights,**kwargs):
        depth=np.zeros(matrix.shape[0])
        solver,NRandom,n_refinements,sphcap_shrink,alpha_Dirichlet,cooling_factor,cap_size,start,space,\
                line_solver,bound_gc,exact,mah_estimate,mah_parMcd,beta,distance,Lp_p,method,pretransform,\
                kernel,kernel_bandwidth,k=self._check_hyperparDepth(**kwargs)
        
        if metric=='riemannian':
            for i in range(matrix.shape[0]):
                for t in range(matrix.shape[1]):
                    z=np.zeros((self.dim*self.dim))
                    dataR=self._applyLogmECoeff(matrix[i,t],self.data[:,t])
                    dpt,state=self._notionsDict[notionDic](z,dataR,notion=notion,
                                solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                                alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,
                                line_solver=line_solver,bound_gc=bound_gc,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
                                beta=beta,distance=distance,Lp_p=Lp_p,method=method,pretransform=pretransform,
                                kernel=kernel,kernel_bandwidth=kernel_bandwidth,k=k)
                    depth[i]+=dpt*weights[i,t]
                    self.RNG.bit_generator.state=state
        if metric=="logeuclidean":
            for t in range(self.timesteps):
                DataLogeuclidean=self._applyLogmECoeff(np.eye(self.dim), self.data[:,t])
                z=self._applyLogmECoeff(np.eye(self.dim),matrix[:,t])
                dpt,state=self._notionsDict[notionDic](z,DataLogeuclidean,notion=notion,
                                            solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                                            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,
                                            line_solver=line_solver,bound_gc=bound_gc,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
                                            beta=beta,distance=distance,Lp_p=Lp_p,method=method,pretransform=pretransform,
                                            kernel=kernel,kernel_bandwidth=kernel_bandwidth,k=k,state=self.RNG.bit_generator.state)
                depth+=dpt*weights[:,t]
                self.RNG.bit_generator.state=state
        if metric=="cholesky":
            print("Not implemented")
        if metric=="euclidean":
            for t in range(self.timesteps):
                DataEuclidean=self._ECoeff(self.data[:,t])
                z=self._ECoeff(matrix[:,t])
                dpt,state=self._notionsDict[notionDic](z,DataEuclidean,notion=notion,
                                            solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                                            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,
                                            line_solver=line_solver,bound_gc=bound_gc,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
                                            beta=beta,distance=distance,Lp_p=Lp_p,method=method,pretransform=pretransform,
                                            kernel=kernel,kernel_bandwidth=kernel_bandwidth,k=k,state=self.RNG.bit_generator.state)
                depth+=dpt*weights[:,t]
                self.RNG.bit_generator.state=state
        if metric=="rooteuclidean":
            for t in range(self.timesteps):
                DataRooteuclidean=self._ECoeff(np.sqrt(self.data[:,t]))
                z=self._ECoeff(np.sqrt(matrix[:,t]))
                dpt,state=self._notionsDict[notionDic](z,DataRooteuclidean,notion=notion,
                                            solver=solver,NRandom=NRandom,n_refinements=n_refinements,sphcap_shrink=sphcap_shrink,
                                            alpha_Dirichlet=alpha_Dirichlet,cooling_factor=cooling_factor,cap_size=cap_size,start=start,space=space,
                                            line_solver=line_solver,bound_gc=bound_gc,exact=exact,mah_estimate=mah_estimate,mah_parMcd=mah_parMcd,
                                            beta=beta,distance=distance,Lp_p=Lp_p,method=method,pretransform=pretransform,
                                            kernel=kernel,kernel_bandwidth=kernel_bandwidth,k=k,state=self.RNG.bit_generator.state)
                depth+=dpt*weights[:,t]
                self.RNG.bit_generator.state=state
        return depth
    
    
    
    def _build_weight(self,weights, queryShape):
        if type(weights)==type(None):
            weights=np.ones((queryShape,self.data_array.shape[1]))/self.data_array.shape[1]
            return weights
        if len(weights.shape)==1:
            if weights.shape[0]!=self.data_array.shape[1]:
                raise ValueError(f"Size of weights is not the same of the time steps. \n {weights.shape[1]}!={self.data_array.shape[1]}")
            weights=weights/sum(weights)
            weights=np.repeat(weights[np.newaxis,...], queryShape, axis=0)
            return weights
        if weights.shape[1]!=self.data_array.shape[1]:
            raise ValueError(f"Size of weights is not the same of the time steps. \n {weights.shape[1]}!={self.data_array.shape[1]}") 
        elif weights.shape[0]==1:
            print("here")
            weights=weights/sum(weights,axis=1,keepdims=True)
            weights=np.repeat(weights[0][np.newaxis,...], queryShape, axis=0)
    
    def _applyLogmECoeff(self,y,X):
        out = np.empty((self.samples,self.dim*self.dim))
        for k in range(self.samples):
            out[:, k] = self._ECoeff(self._Logm(y, X[k]).reshape(1,self.dim,self.dim))
        return out
    def _sympd_funcm(self,mat,func):
        """Apply scalar function f to the eigenvalues of Hermitian PD matrix A."""
        w,v=np.linalg.eigh(mat)
        return (v*func(w))@v.conj().T
    def _sqrtm_sympd(self,mat):
        return self._sympd_funcm(mat,np.sqrt)
    def _invsqrtm_sympd(self,mat):
        return self._sympd_funcm(mat,lambda w: 1.0/np.sqrt(w))
    def _logm_sympd(self,mat):
        return self._sympd_funcm(mat,np.log)
    def _Logm(self,P, Q):
        if np.linalg.norm(P-np.eye(self.dim),ord=np.inf)<1e-10:
            return self._logm_sympd(Q)
        P1=self._sqrtm_sympd(P)
        P2=self._invsqrtm_sympd(P)
        P3=self._logm_sympd(P2@Q@P2)
        return P1@P3@P1
    
    # def _ECoeff(self,H):
    #     coeff=np.empty((self.dim, self.dim))
    #     di=np.diag_indices(self.dim)
    #     il=np.tril_indices(self.dim,-1)
    #     iu=np.triu_indices(self.dim,1)
    #     coeff[di]=H[di].real
    #     coeff[il]=np.sqrt(2)*H[il].real
    #     coeff[iu]=np.sqrt(2)*H[iu].imag
    #     return coeff.ravel()
    def _ECoeff(self,H):
        coeff=np.empty((H.shape[0],self.dim, self.dim))
        di=np.diag_indices(self.dim)
        il=np.tril_indices(self.dim,-1)
        iu=np.triu_indices(self.dim,1)
        coeff[:,di]=H[:,di].real
        coeff[:,il]=np.sqrt(2)*H[:,il].real
        coeff[:,iu]=np.sqrt(2)*H[:,iu].imag
        return coeff.reshape(H.shape[0],self.dim*self.dim)
    
    def check_HPD(self,X=None,tol=1e-10):
        """
        Check Hermitian positive-definiteness.
        Parameters
        ----------
            H  : {array-like} of shape (d,d).
                Matrix to check if it is hermitian.
                
            tol  : float
                tolerance for hermitian check. 
    
        Returns :
        ---------
            boolena to determine if it is hermitian or not
        """
        if type(X)==type(None):
            X=self.data
        if len(X.shape)==3:
            for i in range(self.samples):
                if isHPD(self.data[i],tol)==False:
                    return False
        if len(X.shape)==4:
            for i in range(self.samples):
                for j in range(self.timesteps):
                    if isHPD(self.data[i,j],tol)==False:
                        return False
        
        return True
            
    
    
    def _check_hyperparDepth(self,**kwargs):
    
        hyperValues={
        "solver":'neldermead',
        "NRandom":1000,
        "n_refinements": 10,
        "sphcap_shrink": 0.5,
        "alpha_Dirichlet": 1.25,
        "cooling_factor": 0.95,
        "cap_size": 1,
        "start": "mean",
        "space": "sphere",
        "line_solver": "goldensection",
        "bound_gc": True,
        "exact":False,
        "mah_estimate": "moment",
        "mah_parMcd": 0.75,
        "beta":2,
        "distance": "Lp",
        "Lp_p": 2,
        "method": "recursive",
        "pretransform": "1Mom",
        "kernel": "EDKernel" ,
        "kernel_bandwidth": 0,
        "k":0.05,
        }
        for key, value in kwargs.items():
            if key in hyperValues.keys():
                hyperValues[key]=value
            else:
                print(f"{key} is not a parameter for depth computation")
        return list(hyperValues.values())
    def _determine_depth_func(self,depth, exact):
        all_depthsProj = ["projection", "aprojection", "cexpchullstar", "cexpchull", "geometrical", "sprojection"]
        all_depthsExac = ["potential","qhpeeling", "simplicial","betaskeleton","L2", "simplicialvolume","spatial",]
        toCheck=["mahalanobis", "halfspace", "zonoid"]
        if depth in all_depthsExac:return depth
        elif depth in all_depthsProj:return "PorjBased"
        elif depth in toCheck and exact==False:return "PorjBased"
        elif depth in toCheck and exact==True:return depth
        return "PorjBased"
    
    def _checkSpaceMetric(self, metric):
        metrics=["riemannian", "logeuclidean", "cholesky", "euclidean", "rooteuclidean"]
        if metric not in metrics:
            raise ValueError(f"Metric must be one of the followings: {["riemannian", "logeuclidean", "cholesky", "euclidean", "rooteuclidean"]}, got {metric}")
    
    def set_seed(self,seed:int=None)->None:
        """Set seed for computation"""
        if type(seed) == type(None) : self.RNG = np.random.default_rng()
        elif type(seed)==int:self.RNG = np.random.default_rng(seed)
        else : raise TypeError("seed must be an integer.")
    
