/******************************************************************************/
/* File:             APD.cpp                                                  */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains functions that compute the asymmetric projection depth of a       */
/* point w.r.t a data cloud.                                                  */
/*                                                                            */
/******************************************************************************/

#include <algorithm>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include "APD.h"

using namespace std;

namespace DataDepth {

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

	double APD1(double z, const double* x, int n) {
		double* xCopy = new double[n]; // Copy of "x" as "median(...)" spoils it
		memcpy(xCopy, x, n * sizeof(double));
		// The median is computed as the ceil(n * 0.5)-th smallest element in the
		// array x
		int n2 = ceil(n * 0.5);
		nth_element(xCopy, xCopy + n2, xCopy + n);
		double med{ xCopy[n2] };
		double dev;
		if (z >= med) {
			// The median of positive deviations from the median is computed as
			// the difference between the third quartile and the median
			int n3 = ceil(n * 0.75);
			nth_element(xCopy + n2 + 1, xCopy + n3, xCopy + n);
			dev = xCopy[n3] - med;
		}
		else {
			// The median of negative deviations from the median is computed as
			// the difference between the median and the first quartile.
			int n1 = ceil(n * 0.25);
			nth_element(xCopy, xCopy + n1, xCopy + n2);
			dev = med - xCopy[n1];
		}
		delete[] xCopy;
		return 1.0 / (1.0 + fabs(z - med) / dev);
	}
	
}
