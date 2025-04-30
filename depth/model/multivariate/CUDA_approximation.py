import numpy as np
from numba import cuda
import ctypes
import torch
from torch.nn.functional import normalize
import gc
import math


def cudaApprox(data:torch.tensor,x:np.ndarray|torch.tensor,notion:str,
            solver:str,option:int,NRandom:int,n_refinements:int,sphcap_shrink:float,cap_size:int|float,
            step:int):
    """Main function to compute approximated depth based on chosen notion
    
    Parameters
    ----------
    data
    x
    notion
    solver = "",
    NRandom = 1000,
    option = 1,
    n_refinements = 10,
    sphcap_shrink = 0.5,
    cap_size = 1,
    start = "mean",
    step - memory overload 
    """
    # IMPORTANT TO REMEMBER: data is a transposed matrix, spaceDim x nSamples 

    if len(x.shape)==1:x=x.reshape(1,-1)
    xCUDA=torch.tensor(x,dtype=torch.float32,device="cuda:0") # transfert x to cuda
    dirRef=math.ceil(NRandom/n_refinements) # amount of direction per refinement
    Pdata=torch.empty((dirRef,data.shape[1]),dtype=torch.float32, device="cuda:0") # allocate memory for data projection (largest matrix)
    dirs=torch.empty((dirRef,data.shape[0]),dtype=torch.float32, device="cuda:0") # allocate memory for directions matrix
    finalDepth=np.empty((x.shape[0])) # final depth matrix
    if option==2:finalDirections=np.empty((x.shape)) # direction matrix
    for z in xCUDA:
        pass
    pass

def RRS(data:torch.tensor,z:torch.tensor,notion:str,
        solver:str,option:int,dirRef:int,n_refinements:int,sphcap_shrink:float,
        cap_size:int|float,step:int,Pdata,dirs):
    """Compute (refined) Random search"""
    
    pole=normalize(z).reshape(z.shape)