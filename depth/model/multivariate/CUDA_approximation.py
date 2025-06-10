# Authors: Leonardo Leone

import numpy as np
import ctypes
import torch
from torch.nn.functional import normalize
import gc
import math

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def cudaApprox(data:torch.Tensor,x:np.ndarray|torch.Tensor,notion:str,
            solver:str,option:int,NRandom:int,n_refinements:int,sphcap_shrink:float,
            step:int=10000)->torch.Tensor:
    """Main function to compute approximated depth based on chosen notion 
    """
    torch.manual_seed(2801)
    # IMPORTANT TO REMEMBER: data is a transposed matrix, spaceDim x nSamples 
    if len(x.shape)==1:x=x.reshape(1,-1)
    # xCUDA=torch.tensor(x,dtype=torch.float32,device=device) # transfert x to cuda
    dirRef=math.ceil(NRandom/n_refinements) # amount of direction per refinement
    Pdata=torch.empty((dirRef,data.shape[1]),dtype=torch.float32, device=device) # allocate memory for data projection (largest matrix)
    dirs=torch.empty((dirRef,data.shape[0]),dtype=torch.float32, device=device) # allocate memory for directions matrix
    finalDepth=np.empty((x.shape[0])) # final depth matrix
    if solver=="simplerandom": sphcap_shrink=1
    if option==2:finalDirections=np.empty((x.shape)) # direction matrix
    for ind,z in enumerate(x):
        zCuda=torch.tensor(z.reshape(1,-1),dtype=torch.float32,device=device)
        D=RS(data,zCuda,notion,option,dirRef,n_refinements,sphcap_shrink,step,Pdata,dirs)
        if option==1:
            finalDepth[ind]=D
        elif option==2:
            finalDepth[ind],finalDirections[ind]=D
    
    try:
        torch.cuda.synchronize()
        del dirs,Pdata,D
        torch.cuda.empty_cache()
    except:
        torch.mps.synchronize()
        del dirs,Pdata,D
        torch.mps.empty_cache()

    if option==1:return finalDepth
    elif option==2:return finalDepth,finalDirections

def RS(data:torch.Tensor,z:torch.Tensor,notion:str,
        option:int,dirRef:int,n_refinements:int,sphcap_shrink:float,
        step:int,Pdata,dirs):
    """Compute (refined) Random search
    """
    eps=torch.tensor([torch.pi/2],dtype=torch.float32,device=device) # initial cap size
    pole=normalize(z).reshape(z.shape) # first pole 
    dMin=torch.ones((1,1),dtype=torch.float32,device=device)
    for ref in range(n_refinements):
        dirs=poleCuda(dirs,num_dir=dirRef, pole=pole,eps=eps,)
        torch.matmul(dirs,data,out=Pdata)
        Pz=torch.matmul(dirs,z.T).reshape(1,-1)
        Pz=depthCompNotion(z,data,Pz,Pdata,notion,step)
        eps*=sphcap_shrink
        index_min=torch.argmin(Pz, 1,)
        torch.index_select(Pz, 1,index_min,out=dMin)
        pole=torch.index_select(dirs, 0, index_min)
    if option==2: # return based on option (1 or 2)
        return dMin[0].cpu(), pole.cpu()
    else:return dMin[0].cpu()


def poleCuda(dirs:torch.Tensor,num_dir:int,pole:torch.Tensor, eps : float)->torch.Tensor:
    """
    creating multiple directions based on the north pole values in the p tensor
    
    Parameters
    ----------
        dirs: tensor
            Memory allocated directions
        
        num_dir: int
            Number of directions per refinement 
        
        pole: tensor
            center pole to direction creation

        eps: Tensor
            spherical cap size

    Returns
    ------
        dirs: tensor
            created directions inside region
    """
    def randVectorSphCuda(dirs,num_dir:int,p:torch.Tensor,eps)->torch.Tensor:
        """
        function to generate a random number from a spherical cap of size 'eps' around 'p'
        the new p is the minimium direction
        """
        torch.zeros(dirs.shape,device=device,out=dirs) 
        dirs[:,1:]=torch.normal(0,1,size=(num_dir*(p.shape[0]-1),), device=device).reshape(num_dir, -1)
        torch.nn.functional.normalize(dirs,dim=1,out=dirs)
        dirs=dirs.T
        dirs[0]=torch.rand(num_dir, device=device)
        dirs[0]*=eps
        dirs[0]=torch.cos(dirs[0])
        raiz=torch.multiply(dirs[0],dirs[0])
        raiz=1-raiz
        raiz=torch.sqrt(raiz).reshape(1,-1)
        dirs[1:] = torch.multiply(dirs[1:],raiz,)
        return householderTransfCuda(dirs,p)

    def householderTransfCuda(dirs:torch.Tensor,p:torch.Tensor)->torch.Tensor:
        """
        this function maps e1 to p, it is applied to x 
        
        """
        if p[0]==1:
            return dirs.T
        temp=torch.matmul(p.reshape(1,-1),dirs).reshape(-1)
        lamb=(temp-dirs[0].reshape(-1)) 
        lamb=lamb/(1-p[0]) # (<p,x> - x1)/(1-p1) 
        dirs[0] = dirs[0]+lamb
        dirs-=torch.matmul(p.reshape(-1,1),lamb.reshape(1,-1))
        dirs=dirs.T
        del lamb, temp
        return dirs
 
    dirs=randVectorSphCuda(dirs,num_dir, pole[0], eps) # call function
    for i in range(pole.shape[0]):
        dirs[i] = pole[i] # put pole back in the 
    return dirs

def depthCompNotion(z,data,Pz,Pdata,notion,step)->torch.Tensor:
    """Computes the appproximate depth based on the chosen projection-based notion
    """
    if notion=="projection":
        prjMED=torch.empty((1,Pdata.shape[0]), device=device) # Memory alloc
        prjMAD=torch.empty((1,Pdata.shape[0]), device=device) # Memory alloc
        for i in range(0,Pdata.shape[0],step): # Compute median
            prjMED[0][i:i+step]=torch.median(Pdata[i:i+step],1).values
        torch.subtract(Pz,prjMED,out=Pz)
        torch.subtract(Pdata,prjMED.reshape(-1,1),out=Pdata)
        for i in range(0,Pdata.shape[0],step): # Compute MAD
            prjMAD[0][i:i+step]=torch.median(torch.abs(Pdata[i:i+step]),1).values
        torch.divide(Pz,prjMAD,out=Pz)
        torch.abs(Pz,out=Pz)
        Pz=1/(1+Pz) # Compute final depth
        return Pz
    elif notion=="halfspace":
        refQuant=Pdata.shape[1]
        ge=torch.greater_equal(Pdata,Pz.T,)
        le=torch.less_equal(Pdata,Pz.T,)
        tempGe=torch.sum(ge,1)
        tempGe=torch.divide(tempGe,refQuant)
        tempLe=torch.sum(le,1)
        refCount=torch.minimum(tempGe,tempLe).reshape(z.shape[0],-1)
        torch.divide(refCount,refQuant,out=Pz)
        return Pz
    elif notion=="aprojection":
        prjMED=torch.empty((1,Pdata.shape[0]), device=device) # Memory alloc
        prjMAD=torch.empty((1,Pdata.shape[0]), device=device) # Memory alloc
        for i in range(0,Pdata.shape[0],step): # Compute median
            prjMED[0][i:i+step]=torch.median(Pdata[i:i+step],1).values
        torch.subtract(Pz,prjMED,out=Pz)
        torch.maximum(Pz, torch.tensor(0, device=device),out=Pz)
        torch.subtract(Pdata,prjMED.reshape(-1,1),out=Pdata)
        Pdata[Pdata<=0]=torch.nan
        for i in range(0,Pdata.shape[0],step): # Compute MAD
            prjMAD[0][i:i+step]=torch.nanmedian(torch.abs(Pdata[i:i+step]),1).values
        torch.divide(Pz,prjMAD,out=Pz)
        torch.abs(Pz,out=Pz)
        Pz=1/(1+Pz) # Compute final depth
        return Pz
        