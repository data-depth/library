"""
=================================================
Data depth computation for multiple distributions 
=================================================

Sample usage of depth computation for multiple distributions.
It will plot samples based on different depth notions.

"""

# %%

from depth.model.DepthEucl import DepthEucl 
import numpy as np
from matplotlib import pyplot as plt

# %%
np.random.seed(2801)
mat1=[[1, 0],[0, 2]]
X1=np.random.normal([0,0],1,(100,2))
dist1=np.repeat(0,100)
X2=np.random.normal([5,5],1,(100,2))
dist2=np.repeat(1,100)
X3=np.random.normal([-5,5],1,(100,2))
dist3=np.repeat(2,100)
X=np.concatenate((np.concatenate((X1,X2),axis=0),X3),axis=0) 
dists=np.concatenate((np.concatenate((dist1,dist2),axis=0),dist3),axis=0)

# %%
plt.figure()
plt.scatter(X1[:,0],X1[:,1],color="blue",label="First distribution",)
plt.scatter(X2[:,0],X2[:,1],color="red",label="Second distribution",)
plt.scatter(X3[:,0],X3[:,1],color="green",label="Third distribution")
plt.xlabel("First component")
plt.ylabel("Second component")
plt.title("Multiple distibution samples")
plt.legend()
plt.show()

# %%
modelMult=DepthEucl().load_dataset(X,distribution=dists)
modelMult.L2(X)
colors=["blue","red","green"]
plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,1,)
    plt.scatter(modelMult.L2Depth[0][modelMult.distribution==i],
                modelMult.L2Depth[1][modelMult.distribution==i],color=colors[i])
    plt.subplot(1,3,2)
    plt.scatter(modelMult.L2Depth[0][modelMult.distribution==i],
                modelMult.L2Depth[2][modelMult.distribution==i],color=colors[i])
    plt.subplot(1,3,3)
    plt.scatter(modelMult.L2Depth[1][modelMult.distribution==i],
                modelMult.L2Depth[2][modelMult.distribution==i],color=colors[i])
    
plt.subplot(1,3,1,)
plt.xlabel("Depth w.r.t. distribution 1")
plt.ylabel("Depth w.r.t. distribution 2")
plt.subplot(1,3,2)
plt.xlabel("Depth w.r.t. distribution 1")
plt.ylabel("Depth w.r.t. distribution 3")
plt.subplot(1,3,3)
plt.xlabel("Depth w.r.t. distribution 2")
plt.ylabel("Depth w.r.t. distribution 3")
plt.legend(["First distribution","Second distribution","Third distribution"])

plt.suptitle("Multiple distribution using L2 depth")
plt.show()


# %%
modelMult.halfspace(X)
colors=["blue","red","green"]
plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,1,)
    plt.scatter(modelMult.halfspaceDepth[0][modelMult.distribution==i],
                modelMult.halfspaceDepth[1][modelMult.distribution==i],color=colors[i])
    plt.subplot(1,3,2)
    plt.scatter(modelMult.halfspaceDepth[0][modelMult.distribution==i],
                modelMult.halfspaceDepth[2][modelMult.distribution==i],color=colors[i])
    plt.subplot(1,3,3)
    plt.scatter(modelMult.halfspaceDepth[1][modelMult.distribution==i],
                modelMult.halfspaceDepth[2][modelMult.distribution==i],color=colors[i])
    
plt.subplot(1,3,1,)
plt.xlabel("Depth w.r.t. distribution 1")
plt.ylabel("Depth w.r.t. distribution 2")
plt.subplot(1,3,2)
plt.xlabel("Depth w.r.t. distribution 1")
plt.ylabel("Depth w.r.t. distribution 3")
plt.subplot(1,3,3)
plt.xlabel("Depth w.r.t. distribution 2")
plt.ylabel("Depth w.r.t. distribution 3")
plt.legend(["First distribution","Second distribution","Third distribution"])

plt.suptitle("Multiple distribution using halfspace depth")
plt.show()

# %%
modelMult.simplicialVolume(X)
colors=["blue","red","green"]
plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,1,)
    plt.scatter(modelMult.simplicialVolumeDepth[0][modelMult.distribution==i],
                modelMult.simplicialVolumeDepth[1][modelMult.distribution==i],color=colors[i])
    plt.subplot(1,3,2)
    plt.scatter(modelMult.simplicialVolumeDepth[0][modelMult.distribution==i],
                modelMult.simplicialVolumeDepth[2][modelMult.distribution==i],color=colors[i])
    plt.subplot(1,3,3)
    plt.scatter(modelMult.simplicialVolumeDepth[1][modelMult.distribution==i],
                modelMult.simplicialVolumeDepth[2][modelMult.distribution==i],color=colors[i])
    
plt.subplot(1,3,1,)
plt.xlabel("Depth w.r.t. distribution 1")
plt.ylabel("Depth w.r.t. distribution 2")
plt.subplot(1,3,2)
plt.xlabel("Depth w.r.t. distribution 1")
plt.ylabel("Depth w.r.t. distribution 3")
plt.subplot(1,3,3)
plt.xlabel("Depth w.r.t. distribution 2")
plt.ylabel("Depth w.r.t. distribution 3")
plt.legend(["First distribution","Second distribution","Third distribution"])

plt.suptitle("Multiple distribution using simplicial volume depth")
plt.show()

# %%

plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1,)
    plt.scatter(modelMult.simplicialVolumeDepth[i][modelMult.distribution==i],
                modelMult.halfspaceDepth[i][modelMult.distribution==i],color=colors[i])

    plt.xlabel(f"SimplicialVolume depth w.r.t. distribution {i}")
    plt.ylabel(f"Halfspace depth w.r.t. distribution {i}")
    plt.title(fr"Distribution {i} depth$\times$depth plot")
plt.show()

