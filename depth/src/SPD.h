
/******************************************************************************/
/* File:             SPD.h                                                  */
/* Created by:       Leonardo Leone                                           */
/* Last revised:     27.08.2026                                               */
/*                                                                            */
/* Contains functions that compute the Skew-adjusted projection depth of a    */
/* point w.r.t a data cloud.                                                  */
/*                                                                            */
/******************************************************************************/

#ifndef __SkewAdjustdeProjectionDepth__
#define __SkewAdjustedProjectionDepth__

namespace DataDepth {
	#ifndef M_PI
	#define M_PI       3.14159265358979323846
	#endif

	/****************************************************************************/
	/* SPD1 computes the skew-adjusted projection depth for univariate data.    */
	/*                                                                          */
	/* Args:                                                                    */
	/*   z - the point for which to calculate the depth,                        */
	/*   x - the data w.r.t. which the depth has to be computed, (vector of     */
	/*       dimension n)                                                       */
	/*   n - number of the data points.                                         */
	/* Returns:                                                                 */
	/*   skew-adjusted projection depth of z w.r.t. x.                          */
	/****************************************************************************/
    double SPD1(double z, const double* x, int n);

}

#endif
