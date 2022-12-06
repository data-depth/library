/******************************************************************************/
/* File:             auxLinAlg.cpp                                            */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains declarations of several auxiliary functions from linear algebra.  */
/*                                                                            */
/******************************************************************************/

#include "auxLinAlg.h"

using namespace std;
using namespace dyMatrixClass;

/* Definition of constants */
const double eps = 1e-10;

/******************************************************************************/
/* 'SphericalToCartesian' converts spherical coordinates to Cartesian         */
/*   coordinates.                                                             */
/*                                                                            */
/* Args:                                                                      */
/*   theta - spherical coordinates of a poin on the unit sphere. Since the    */
/*           radius is unity in this case, theta is a vector of dimension     */
/*           d-1 that contains the angles only                                */
/*   d - dimension of the data                                                */
/* Returns:                                                                   */
/*   vector of dimension d that contains the Cartesian coordinates            */
/******************************************************************************/

unique_ptr<double[]> SphericalToCartesian(const double theta[], int d) {
	unique_ptr<double[]> result{ new double[d] };
	for (int i = 0; i < d; i++) result[i] = 1.0;
	for (int i = 0; i < d - 1; i++) {
		result[i] *= cos(theta[i]);
		double tmp = sin(theta[i]);
		for (int j = i + 1; j < d; j++) result[j] *= tmp;
	}
	return result;
}

/******************************************************************************/
/* 'InnerProduct' computes the inner product of two vectors x and y           */
/*                                                                            */
/* Args:                                                                      */
/*   x - vector of dimension d                                                */
/*   y - vector of dimension d                                                */
/*   d - dimension of the vectors                                             */
/* Returns:                                                                   */
/*   inner product of the two vectors, i.e., \sum_{i=1}^d x_i * y_i           */
/******************************************************************************/

double InnerProduct(const double x[], const double y[], int d) {
	double sum = 0;
	for (int i = 0; i < d; i++) sum += x[i] * y[i];
	return sum;
}

/******************************************************************************/
/* 'norm2' computes the Euclidean norm (2-norm) of a vector x                 */
/*                                                                            */
/* Args:                                                                      */
/*   x - vector of dimension d                                                */
/*   d - dimension of the vector                                              */
/* Returns:                                                                   */
/*   Euclidean norm of the vector x, i.e., \sqrt{\sum_{i=1}^d x_i^2}          */
/******************************************************************************/

double norm2(const double x[], int d) {
	double result = 0;
	for (int i = 0; i < d; i++) result += x[i] * x[i];
	return sqrt(result);
}

/******************************************************************************/
/* 'distance' computes the Euclidean distance between two vectors x and y     */
/*                                                                            */
/* Args:                                                                      */
/*   x - vector of dimension d                                                */
/*   y - vector of dimension d                                                */
/*   d - dimension of the vectors                                             */
/* Returns:                                                                   */
/*   Euclidean norm of the vector x, i.e., \sqrt{\sum_{i=1}^d (x_i -y_i)^2}   */
/******************************************************************************/

double distance(const double x[], const double y[], int d) {
	double sum = 0;
	for (int i = 0; i < d; i++) sum += (x[i] - y[i]) * (x[i] - y[i]);
	return sqrt(sum);
}

/******************************************************************************/
/* 'mean' computes the mean vector of n vectors of dimension d                */
/*                                                                            */
/* Args:                                                                      */
/*   x - matrix of dimension n x d                                            */
/*   n - number of vectors, i.e., rows in the matrix x                        */
/*   d - dimension of the vectors, i.e., columns of the matrix x              */
/* Returns:                                                                   */
/*   mean vector of the n rows of matrix x                                    */
/******************************************************************************/

unique_ptr<double[]> mean(const cMatrix x, int n, int d) {
	unique_ptr<double[]> result{ new double[d] };
	for (int j = 0; j < d; j++) {
		result[j] = 0;
		for (int i = 0; i < n; i++) result[j] += x(i, j);
		result[j] /= n;
	}
	return result;
}

/******************************************************************************/
/* 'Normalize' normalizes a vector x to have Eucliedan norm one, i.e,         */
/*  projects x on the unit sphere                                             */
/*                                                                            */
/* Args:                                                                      */
/*   x - vector of dimension d. On exit, x contains the normalized vector,    */
/*       i.e., x divided by the Euclidean norm of x                           */
/*   d - dimension of the vectors, i.e., columns of the matrix x              */
/******************************************************************************/

void Normalize(double x[], int d) {
	double Norm = 0;
	for (int i = 0; i < d; i++) Norm += x[i] * x[i];
	Norm = sqrt(Norm);
	for (int i = 0; i < d; i++) x[i] /= Norm;
}

/******************************************************************************/
/* 'Householder' performs a Housholder transformationn on the point z.        */
/* The Housholder matrix H that maps the first unit vector e_1 to the point   */
/* u on the unit sphere is computed. Then, H is applied to z, i.e., H * z     */
/* is computed. The result H * z overwrites the vector z.                     */
/*                                                                            */
/* Args:                                                                      */
/*   z - vector of dimension d. z is assumed to have Euclidean norm one.      */
/*       On exit, z is overwritten by H * z where H is the Householder        */
/*       transformation that maps e_1 to u                                    */
/*   u - vector of dimension u. u is assumed to have Euclidean norm one.      */
/*   d - dimension of the vectors z and u                                     */
/******************************************************************************/

void Householder(double z[], const double u[], int d) {
	if (u[0] < 1 - eps) {
		double lambda = (InnerProduct(z, u, d) - z[0]) / (1 - u[0]);
		z[0] += lambda;
		for (int i = 0; i < d; i++) z[i] -= lambda * u[i];
	}
}

/****************************************************************************/
/* 'Cholesky' computes the Cholesky decomposition of a symmetric positive   */
/*   semi-definite matrix: A = GG^t                                         */
/*                                                                          */
/* Args:                                                                    */
/*   a  - the matrix of dimension n x n which has to be decomposed,         */
/*        only the lower triangular part of 'a' is used in the procedure,   */
/*        on exit, the lower triangular part of 'a' contains the lower      */
/*        triangular matrix G for which A = GG^t holds,                     */
/*   n  - dimension of the matrix 'a',                                      */
/*   rank  - on exit contains the rank of the matrix 'a' (which is the      */
/*           same as the rank of the output matrix 'G',                     */
/*   error - error indicator with the following error codes:                */
/*             0: no error,                                                 */
/*             1: error, matrix 'a' is not positive semi-definite.          */
/*                                                                          */
/****************************************************************************/

void Cholesky(double** a, int n, int& rank, int& error) {
	const double eps = 1e-12;
	error = 1;
	rank = 0;
	for (int j = 0; j < n; j++) {
		double eps1 = eps * fabs(a[j][j]);
		for (int k = 0; k < j; k++) a[j][j] -= a[j][k] * a[j][k];
		if (fabs(a[j][j]) <= eps1) a[j][j] = 0;
		else if (a[j][j] > 0) {
			a[j][j] = sqrt(a[j][j]);
			rank++;
		}
		else return;
		for (int i = j + 1; i < n; i++) {
			double eps2 = eps * fabs(a[i][i]);
			for (int k = 0; k < j; k++) a[i][j] -= a[i][k] * a[j][k];
			if (a[j][j] > 0)
				a[i][j] /= a[j][j];
			else
				if (fabs(a[i][j]) <= eps1 * eps2)
					a[j][j] = 0;
				else
					return;
		}
	}
	error = 0;
}

/****************************************************************************/
/* 'InversePosDef' computes the (psuedo)-inverse of a symmetric positive    */
/*   semi-definite matrix 'a'.                                              */
/*                                                                          */
/* Args:                                                                    */
/*   a  - the matrix of dimension n x n which has to be inverted,           */
/*        only the lower triangular part of 'a' is used in the procedure.   */
/*        Since the inverse is again symmetric, on exit, the lower          */
/*        trinagular part of the inverse is stored in the lower triangular  */
/*        part of the matrix 'a'.                                           */
/*   n  - dimension of the matrix 'a',                                      */
/*   rank  - on exit contains the rank of the matrix 'a' (which is the      */
/*           same as the rank of the inverse,                               */
/*   error - error indicator with the following error codes:                */
/*             0: no error,                                                 */
/*             1: error, matrix 'a' is not positive semi-definite.          */
/*                                                                          */
/****************************************************************************/

void InversePosDef(double** a, int n, int& rank, int& error) {
	const double eps = 1e-12;
	double sum = 0;
	Cholesky(a, n, rank, error);
	for (int j = 0; j < n; j++) {
		if (a[j][j] > eps) {
			a[j][j] = 1 / a[j][j];
			for (int i = j + 1; i < n; i++) {
				if (a[i][i] > eps) {
					sum = 0;
					for (int k = j; k < i; k++) sum -= a[i][k] * a[k][j];
					a[i][j] = sum / a[i][i];
				}
				else a[i][j] = 0;
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j <= i; j++) {
			sum = 0;
			for (int k = i; k < n; k++) sum += a[k][i] * a[k][j];
			a[i][j] = sum;
		}
	}
}
