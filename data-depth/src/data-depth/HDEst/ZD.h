/******************************************************************************/
/* File:             ZD.h                                                     */
/* Created by:       Rainer Dyckerhoff                                        */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains declarations of functions that compute the zonoid depth of a      */
/* point w.r.t. a data cloud.                                                 */
/*                                                                            */
/******************************************************************************/

#ifndef __ZonoidDepth__
#define __ZonoidDepth__

#include <vector>
#include "Matrix.h"

namespace DataDepth {

	typedef std::vector<double> TPoint;

	/****************************************************************************/
	/* ZD computes the zonoid depth for multivariate data.                      */
    /*                                                                          */
    /* For a description of the algorithm, see:                                 */
    /* Dyckerhoff, R., Koshevoy, G., and Mosler, K. (1996)                      */
    /* Zonoid Data Depth : Theory and Computation,                              */
    /* in : Compstat - Proceedings in Computational Statistics, (Albert         */
    /* Prat, ed.), Physica-Verlag, Heidelberg, p. 235--240.                     */
	/*                                                                          */
	/* Args:                                                                    */
	/*   z - the point for which to calculate the depth, (vector of dimension   */
	/*        d)                                                                */
	/*   x - the data w.r.t. which the depth has to be computed, (matrix of     */
	/*        dimension n x d)                                                  */
	/*   nPoints - number of the data points,                                   */
	/*   nDim    - dimension of the Euclidean space,                            */
	/*   Error   - error indicator, possible error codes are                    */
    /*               0: no error,                                               */
	/*               1 : simplex algorithm did not terminate within 'MaxIt'     */
	/*					 iterations,                                            */
	/*	             2 : not enough memory available,                           */
	/*                                                                          */
	/* Returns:                                                                 */
	/*   zonoid depth of z w.r.t. x. (if error code is 0)                       */
	/*	 If the error code is 1, the return value is an lower bound to the      */
	/*   zonoid data depth of 'z'. If the error code is 2, the return value     */
	/*   is - 1.                                                                */
	/*                                                                          */
	/****************************************************************************/
	double ZD(const double* z, const double* const* x, int nPoints, int nDim, int& Error);
	double ZD(const double* z, const dyMatrixClass::cMatrix x, int nPoints, int nDim, int& Error);

	/****************************************************************************/
	/* ZD1 computes the zonoid depth for univariate data.                       */
	/*                                                                          */
	/* The algorithm used here is capable of computing the zonoid depth with a  */
	/* complexity of O(n). The algorithm will be published in a forthcoming     */
	/* paper. Do not use this algorithm without permission of the author.       */
	/*                                                                          */
	/* Args:                                                                    */
	/*   z - the point for which to calculate the depth,                        */
	/*   x - the data w.r.t. which the depth has to be computed, (vector of     */
	/*       dimension n),                                                      */
	/*   n - number of the data points.                                         */
	/*   sorted - optional parameter that indicates that the array x is sorted  */
	/*            in ascending order when 'sorted' is true. Otherwise, the      */
	/*            array x is not sorted                                         */
	/* Returns:                                                                 */
	/*   zonoid depth of z w.r.t. x.                                            */
	/****************************************************************************/
	double ZD1 (double z, const double* x, int n, bool sorted = false);


}
#endif

