/******************************************************************************/
/* File:             PD.cpp                                                   */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains functions that compute the projection depth of a  point w.r.t     */
/* a data cloud.                                                              */
/*                                                                            */
/******************************************************************************/

#include <algorithm>
#include <cstring>
#include <stdlib.h>
#include <cmath>
#include "PD.h"

using namespace std;

namespace DataDepth {

	/****************************************************************************/
	/*                                                                          */
	/* 'med' computes the median of an array x of length n.                     */
	/*                                                                          */
	/****************************************************************************/

	double med(double* x, int n) {
		int m;
		if (n & 1) { // n odd
			m = (n - 1) / 2;
			nth_element(x, x + m, x + n);
			return x[m];
		}
		else {  // n even
			m = n / 2 - 1;
			nth_element(x, x + m, x + n);
			return(x[m] + *min_element(x + m + 1, x + n)) / 2;
		}
	}

	/****************************************************************************/
	/*                                                                          */
	/* 'mad' computes the median of absolute deviations from the median.        */
	/* The median of the data in x is passed as a parameter 'med'               */
	/*                                                                          */
	/****************************************************************************/

	double mad(double* x, int n, double med) {
		for (int i = 0; i < n; i++) {
			x[i] = fabs(x[i] - med);
		}
		int m = (n & 1) ? (n - 1) / 2 : n / 2;
		nth_element(x, x + m, x + n);
		return x[m];
	}

	
	/****************************************************************************/
	/* PD1 computes the projection depth for univariate data.                   */
	/*                                                                          */
	/* Args:                                                                    */
	/*   z - the point for which to calculate the depth,                        */
	/*   x - the data w.r.t. which the depth has to be computed, (vector of     */
	/*       dimension n)                                                       */
	/*   n - number of the data points.                                         */
	/* Returns:                                                                 */
	/*   projection depth of z w.r.t. x.                                        */
	/****************************************************************************/

	double PD1(double z, const double* x, int n) {
		double* xCopy = new double[n]; // Copy of "x" as "median(...)" spoils it
		memcpy(xCopy, x, n * sizeof(double));
		double xmed = med(xCopy, n);
		double xmad = mad(xCopy, n, xmed);
		delete[] xCopy;
		return xmad / (xmad + fabs(z - xmed));
	}

}
