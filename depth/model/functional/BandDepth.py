from ctypes import *
import numpy as np
from ..multivariate.import_CDLL import libExact


def bandDepth(point,data,modified="modified",state=None):
    RNG=np.random.default_rng()
    RNG.bit_generator.state = state
    try:
        n,t,d = data.shape
    except ValueError:
        n = data.shape[0]
        d = 1
    n_z = point.shape[0]

    if(d == 1):
        option = 1

    try:
        n_z, d_z = point.shape
    except ValueError:
        if(d == 1):
            try:
                n_z = point.shape[0]
            except IndexError:
                n_z = 1
        else:
            n_z = 1


    # libExact.ModifiedBandDepth(double *points,double *objects, int *numObjects, int *numArgs,
	# int *dimension,int *numPoints, double *depths)

    # libExact.BandDepth(double *points,double *objects, int *numObjects, int *numArgs,
	# int *dimension,int *numPoints, double *depths)


    pass

def SimplicialBandDepth(point,data,J=2,modified="modified",state=None):

    # libExact.SimplicialBandDepth(double *points,double *objects, int *numObjects, int *numArgs,
	# int *dimension,int *numPoints, int *seed,  int *J,bool modified, double *depths)

    pass