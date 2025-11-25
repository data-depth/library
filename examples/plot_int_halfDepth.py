"""
============================
Integrated halfspace depth 
============================

Sample usage of integrated halfspace depth computation.
It will plot samples and dataset based on integrated functional depth values.

"""


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from depth.model import DepthFunc

# %%

## Creating Brownian motion paths as dataset and storing them in a Pandas DataFrame.

## Simulation Parameters 
T = 1.0        
N = 501      # Number of time points
M = 100      # Number of realizations (paths/functions)
dt = T / (N - 1) 

## Generate dataset (independent Brownian motion paths) 
increments = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(N - 1, M))
initial_zeros = np.zeros((1, M))
dW = np.vstack([initial_zeros, increments])
bm_paths = np.cumsum(dW, axis=0) 

df_bm = pd.DataFrame({
    'time': np.tile(np.linspace(0, T, N), M) ,
    'path_id': np.repeat(np.arange(1, M + 1), N) ,
    'value': bm_paths.flatten(order='F')
})

## Generate a sample (independent Brownian motion paths)
increments_samp = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(N - 1, 10))
initial_zeros_samp = np.zeros((1, 10))
dW_samp = np.vstack([initial_zeros_samp, increments_samp])
bm_paths_samp = np.cumsum(dW_samp, axis=0)

df_bm_samp = pd.DataFrame({
    'time': np.tile(np.linspace(0, T, N), 10) ,
    'path_id': np.repeat(np.arange(1, 10 + 1), N) ,
    'value': bm_paths_samp.flatten(order='F')
})


## Display DataFrame
print(f"The DataFrame containing {M} functions is:")
print(df_bm.head) 
print(f"\nDataFrame shape: {df_bm.shape}")

# visualize dataset and samples
fig=plt.figure()
for case, group in df_bm.groupby("path_id"):
    plt.plot(group["time"], group["value"], color = "b", linewidth = 1, label = 'Dataset' if case == 1 else None)

for case, group in df_bm_samp.groupby("path_id"):
    plt.plot(group["time"], group["value"], color = "r", linewidth = 1, label = 'New functions' if case == 1 else None)

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Dataset and Sample visualization")
plt.legend()
plt.show()



# %%
# Create model and load dataset for depth computation
model = DepthFunc().load_dataset(data = df_bm, timestamp_col = "time", value_cols = ["value"], case_id = "path_id")
DepthDataset = model.projection_based_func_depth(query = df_bm, notion = 'projection')
DepthSample = model.projection_based_func_depth(query = df_bm_samp, notion = 'projection')
print(DepthDataset )

## visualization
cmap = cm.get_cmap('viridis')
norm = Normalize(vmin=DepthDataset.min(), vmax=DepthDataset.max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
for case, group in df_bm.groupby("path_id"):
    ax2.plot(group["time"], group["value"], linewidth = .5, c = 'b', label = "Dataset" if case == 1 else None)

for case, group in df_bm.groupby("path_id"):
    color_rgb = cmap(norm(DepthDataset[case-1]))
    ax1.plot(group["time"], group["value"], c = color_rgb, linewidth = 1)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.set_title("Dataset depth visualization")
for case, group in df_bm_samp.groupby("path_id"):
    color_rgb = cmap(norm(DepthSample[case-1]))
    ax2.plot(group["time"], group["value"], c = color_rgb, linewidth = 1, label = "Sample" if case == 1 else None)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.set_title("Sample depth visualization")

plt.legend()
plt.show()
