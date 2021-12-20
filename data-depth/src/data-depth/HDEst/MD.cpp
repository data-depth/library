/******************************************************************************/
/* File:             MD.cpp                                                   */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains functions that compute the Mahalanobis depth of a point w.r.t.    */
/* a data cloud.                                                              */
/*                                                                            */
/******************************************************************************/

#include <algorithm>
#include <stdlib.h>
#include "auxLinAlg.h"
#include "MD.h"

using namespace std;
using namespace dyMatrixClass;

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

	double MD(const double* z, const double* const* x, int n, int d) {

		int rank, error;
		double* xquer = new double[d];
		double* y = new double[d];
		double** cov = new double*[d];
		for (int i = 0; i < d; i++) cov[i] = new double[d];

		for (int j = 0; j < d; j++) {
			xquer[j] = 0;
			for (int i = 0; i < n; i++) xquer[j] += x[i][j];
			xquer[j] /= n;
		}
		for (int i = 0; i < d; i++) y[i] = z[i] - xquer[i];

		for (int i = 0; i < d; i++) {
			for (int j = 0; j <= i; j++) {
				cov[i][j] = 0;
				for (int k = 0; k < n; k++) cov[i][j] += (x[k][i] - xquer[i]) * (x[k][j] - xquer[j]);
				cov[i][j] /= n;
			}
		}

		InversePosDef(cov, d, rank, error);

		double sum = 0;
		for (int i = 0; i < d; i++) {
			double sum2 = 0;
			for (int j = 0; j < i; j++) sum2 += cov[i][j] * y[j];
			sum += y[i] * (cov[i][i] * y[i] + 2 * sum2);
		}

		for (int i = 0; i < d; i++) delete[] cov[i];
		delete[] cov;
		delete[] xquer;
		delete[] y;

		return 1.0 / (1 + sum);
	}

	/****************************************************************************/
	/*                                                                          */
	/* Overloaded version of 'MD' that uses a matrix class for the parameter x. */
	/*                                                                          */
	/****************************************************************************/

	double MD(const double* z, const cMatrix x, int n, int d) {
		unique_ptr<double*[]> xx{ new double*[n] };
		for (int i = 0; i < n; i++) xx[i] = x[i];
		return MD(z, xx.get(), n, d);
	}

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

	double MD1(double z, const double* x, int n) {
		double xquer, var, sum = 0;
		for (int i = 0; i < n; i++) sum += x[i];
		xquer = sum / n;
		sum = 0;
		for (int i = 0; i < n; i++) sum += (x[i] - xquer) * (x[i] - xquer);
		var = sum / n;
		return 1.0 / (1 + (z - xquer) * (z - xquer) / var);
	}



}
