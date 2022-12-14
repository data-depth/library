
/******************************************************************************/
/* File:             APD.h                                                    */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains declarations of functions that compute the asymmetric projection  */
/* depth of a point w.r.t. a data cloud.                                      */
/*                                                                            */
/******************************************************************************/

#ifndef __AsymmetricProjectionDepth__
#define __AsymmetricProjectionDepth__

namespace DataDepth {
	#ifndef M_PI
	#define M_PI       3.14159265358979323846
	#endif

    /****************************************************************************/
    /* APD1 computes the asymmetric projection depth for univariate data.       */
    /*                                                                          */
    /* Args:                                                                    */
    /*   z - the point for which to calculate the depth,                        */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of     */
    /*       dimension n)                                                       */
    /*   n - number of the data points.                                         */
    /* Returns:                                                                 */
    /*   asymmetric projection depth of z w.r.t. x.                             */
    /****************************************************************************/
    double APD1(double z, const double* x, int n);

}

#endif
