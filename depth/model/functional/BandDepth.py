from ctypes import *
import numpy as np
from ..multivariate.import_CDLL import libExact


def bandDepth(z,X,modified=True,):

    
    try:
        n, t, d = X.shape
    except ValueError:
        n = X.shape[0]
        t = X.shape[1]
        d = 1
    n_z = z.shape[0]

    depths=pointer((c_double*len(z))(*np.zeros(len(z))))

    points_list=X.flatten()
    objects_list=z.flatten()
    points=(c_double*len(points_list))(*points_list)
    objects=(c_double*len(objects_list))(*objects_list)
    points=pointer(points)
    objects=pointer(objects)
    numPoints=pointer(c_int(n))
    numArgs=pointer(c_int(t))
    dimension=pointer(c_int(d))
    numObjects=pointer(c_int(n_z))
    

    if modified==True:
        libExact.ModifiedBandDepth(
                points,
                objects,
                numObjects,
                numArgs,
                dimension,
                numPoints,
                depths,
                )
    else:
        libExact.BandDepth(
                points,
                objects,
                numObjects,
                numArgs,
                dimension,
                numPoints,
                depths,)

    res=np.zeros(len(z))
    for i in range(len(z)):
        res[i]=depths[0][i]
    return res


    

# def SimplicialBandDepth(z,X,J=2,modified=True,state=None):
#     RNG=np.random.default_rng()
#     RNG.bit_generator.state = state
#     try:
#         n, t, d = X.shape
#     except ValueError:
#         n = X.shape[0]
#         t = X.shape[1]
#         d = 1
#     n_z = z.shape[0]

#     depths=pointer((c_double*len(z))(*np.zeros(len(z))))

#     points_list=X.flatten()
#     objects_list=z.flatten()
#     points=(c_double*len(points_list))(*points_list)
#     objects=(c_double*len(objects_list))(*objects_list)
#     points=pointer(points)
#     objects=pointer(objects)
#     numPoints=pointer(c_int(n))
#     numArgs=pointer(c_int(t))
#     dimension=pointer(c_int(d))
#     numObjects=pointer(c_int(n_z))
#     seed = pointer((c_int(RNG.integers(0,10000000,1)[0])))

#     libExact.SimplicialBandDepth(
#             points,
#             objects,
#             numObjects,
#             numArgs,
#             dimension,
#             numPoints,
#             seed,
#             c_int(J),
#             c_bool(modified),
#             depths,
#             )

#     res=np.zeros(len(z))
#     for i in range(len(z)):
#         res[i]=depths[0][i]
#     return res,RNG.bit_generator.state
    # libExact.SimplicialBandDepth(double *points,double *objects, int *numObjects, int *numArgs,
	# int *dimension,int *numPoints, int *seed,  int *J,bool modified, double *depths)
