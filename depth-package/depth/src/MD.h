/******************************************************************************/
/* File:             MD.h                                                     */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains declarations of functions that compute the Mahalanobis depth of a */
/* point w.r.t. a data cloud.                                                 */
/*                                                                            */
/******************************************************************************/

#ifndef __MahalanobisDepth__
#define __MahalanobisDepth__

#include "Matrix.h"

namespace DataDepth {

    /****************************************************************************/
    /* MD computes the Mahalanobis depth for multivariate data.                 */
    /*                                                                          */
    /* Args:                                                                    */
    /*   z  - the point for which to calculate the depth, (vector of dimension  */
    /*        d)                                                                */
    /*   xx - the data w.r.t. which the depth has to be computed, (matrix of    */
    /*        dimension n x d)                                                  */
    /*   n  - number of the data points,                                        */
    /*   d  - dimension of the Euclidean space.                                 */
    /* Returns:                                                                 */
    /*   Mahalanobis depth of z w.r.t. x.                                       */
    /****************************************************************************/
    double MD(const double* z, const double* const* xx, int n, int d);
	double MD(const double* z, const dyMatrixClass::cMatrix xx, int n, int d);

    /****************************************************************************/
    /* MD1 computes the Mahalanobis depth for univariate data.                  */
    /*                                                                          */
    /* Args:                                                                    */
    /*   z - the point for which to calculate the depth,                        */
    /*   x - the data w.r.t. which the depth has to be computed, (vector of     */
    /*       dimension n),                                                      */
    /*   n - number of the data points.                                         */
    /* Returns:                                                                 */
    /*   Mahalanobis depth of z w.r.t. x.                                        */
    /****************************************************************************/
    double MD1(double z, const double* x, int n);

}

#endif


