/******************************************************************************/
/* File:             mvrandom.cpp                                             */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains functions for the generation of pseudorandom numbres from         */
/* multivariate distributions.                                                */
/*                                                                            */
/******************************************************************************/

#include <cmath>
#include <memory>
#include "auxLinAlg.h"
#include "mvrandom.h"


/****************************************************************************/
/* 'cdfn' computes the cdf of the standard normal distribution.             */
/*                                                                          */
/* Args:                                                                    */
/*   x - value at which the cdf has to be evaluated                         */
/* Returns:                                                                 */
/*   value of the cdf of the standard normal distribution at x              */
/****************************************************************************/

double cdfn(double x) {
	return 0.5 * (1 + erf(x / sqrt(2)));
}

/****************************************************************************/
/* 'cElliptic' abstract base class for generating random numbers from       */
/*  elliptic distributions.                                                 */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

// constructor for a spherical distribution
cElliptic::cElliptic(int aDim) : dim(aDim), standard(true), mu(NULL), sigma(NULL) {}

// constructor for an elliptical distribution
cElliptic::cElliptic(int aDim, double* aMu, double** aSigma) : standard(false), dim(aDim), mu(aMu), sigma(aSigma) {
	int rank, error;
	Cholesky(sigma, dim, rank, error);
}

// destructor, free the memory allocated for mu and Sigma
cElliptic::~cElliptic() {
	delete[] mu;
	if (sigma != NULL)
		for (int i = 0; i < dim; i++) delete[] sigma[i];
	delete[] sigma;
}

// transform x by multiplying with the matrix G from a Cholesky decomposition
// of the scale matrix Sigma and adding the location vector mu
void cElliptic::transform(double x[]) const {
	for (int i = dim - 1; i >= 0; i--) {
		double tmp = mu[i];
		for (int j = 0; j <= i; j++) tmp += sigma[i][j] * x[j];
		x[i] = tmp;
	}
}

/****************************************************************************/
/* 'cCombination' class for generating a random combination without         */
/*  replacement of size k from the numbers 0:n-1.                          */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

// constructor, main task here is the allocation of the generators to sample
// from a uniform distribution on some set of integers
cCombination::cCombination(int aN, int aK) {
	n = aN;
	k = aK;
	unifints = new std::uniform_int_distribution<int>*[k];
	for (int i = 0; i < k; i++) {
		unifints[i] = new std::uniform_int_distribution<int>(0, n - 1 - i);
	}
}

// destructor, free the generators for the univariate dirstributions on sets
// of integers
cCombination::~cCombination() {
	for (int i = 0; i < k; i++) {
		delete unifints[i];
	}
	delete[] unifints;
}

