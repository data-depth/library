"""
=================
Functional  depth 
=================

Sample usage of Depth for functional data.
It will plot samples and dataset based on depth notions.

"""

# %%
from depth.model.DepthFunc import DepthFunc 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %%

## Creating dataset and samples
np.random.seed(2801)
n_cases = 100
rows = []
# fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
for case in range(1, n_cases + 1):
    n_points = 100#np.random.randint(3, 16)
    timestamps = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.sort(np.random.randint(0, 1000, n_points)), unit='s'
    )
    baseVal = np.linspace(0,7,n_points)+np.random.normal(0,.2,n_points)
    bruit = np.random.normal(0,1,1)
    value_1 = np.cos(baseVal)+bruit
    value_2 = np.sin(baseVal)+bruit*0.5
    value_3 = np.cos(baseVal)*np.sin(baseVal**1.2)+bruit*0.5#+np.sin(baseVal)#+decVal
    
    # value_2 = np.random.rand(n_points)
    # value_3 = np.random.rand(n_points)
    value_3[-1] = None
    for t, v1, v2, v3 in zip(timestamps, value_1, value_2, value_3):
        rows.append([case, t, v1, v2, v3])
    # ax1.plot(timestamps,value_1)
    # ax2.plot(timestamps,value_2)
    # ax3.plot(timestamps,value_3)
df = pd.DataFrame(rows, columns=["case_id", "timestamp", "value_1", "value_2", "value_3"])

# df[df.case_id==1].plot(x="timestamp", y="value_1")
df.head(10)
# %%

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
for case in range(1, n_cases + 1):
    ax1.plot(df[df.case_id==case].value_1.values)
    ax2.plot(df[df.case_id==case].value_2.values)
    ax3.plot(df[df.case_id==case].value_3.values)
plt.show()
# %%
# Create model and load dataset for depth computation 

model=DepthFunc().load_dataset(df,interpolate_grid=False)
Dpth=model.projection_based_func_depth(df,notion="projection",output_option="lowest_depth",NRandom=1000)
print("depth of first 10 functions:" , Dpth)

# %%

from matplotlib import cm
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
for case in range(1, n_cases + 1):
    ax1.plot(df[df.case_id==case].value_1.values,c=cm.plasma((Dpth[case-1]-Dpth.min())/(Dpth.max()-Dpth.min())))
    ax2.plot(df[df.case_id==case].value_2.values,c=cm.plasma((Dpth[case-1]-Dpth.min())/(Dpth.max()-Dpth.min())))
    ax3.plot(df[df.case_id==case].value_3.values,c=cm.plasma((Dpth[case-1]-Dpth.min())/(Dpth.max()-Dpth.min())))

plt.show()