/******************************************************************************/
/* File:             mvrandom.h                                               */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Contains functions for the generation of pseudorandom numbres from         */
/* multivariate distributions.                                                */
/*                                                                            */
/******************************************************************************/

#ifndef __MVRandom__
#define __MVRandom__

#include <random>
#include <iostream>
#include "auxLinAlg.h"

/****************************************************************************/
/* 'cdfn' computes the cdf of the standard normal distribution.             */
/*                                                                          */
/* Args:                                                                    */
/*   x - value at which the cdf has to be evaluated                         */
/* Returns:                                                                 */
/*   value of the cdf of the standard normal distribution at x              */
/****************************************************************************/

double cdfn(double x);

/****************************************************************************/
/* 'cUniformSphere' class for generating random numbers from the uniform    */
/*  distribution on the unit sphere in R^d.                                 */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cUniformSphere {
	int dim;
	std::normal_distribution<double> normal;
public:
	// constructor: 'aDim' is the dimension d of the ambient space
	cUniformSphere(int aDim) : dim(aDim) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cUniformSphere::operator()(URNG& gen) {
	double norm = 0;
	double* x = new double[dim];
	// generate 'dim' independent standard normal variates and compute the squared 2-norm
	for (int i = 0; i < dim; i++) {
		x[i] = normal(gen);
		norm += x[i] * x[i];
	}
	// compute the norm
	norm = sqrt(norm);
	// normalize the vector by its norm
	for (int i = 0; i < dim; i++) x[i] /= norm;
	return x;
}

template <class URNG> void cUniformSphere::vector(double* const x, URNG& gen) {
	double norm = 0;
	// generate 'dim' independent standard normal variates and compute the squared 2-norm
	for (int i = 0; i < dim; i++) {
		x[i] = normal(gen);
		norm += x[i] * x[i];
	}
	// compute the norm
	norm = sqrt(norm);
	// normalize the vector by its norm
	for (int i = 0; i < dim; i++) x[i] /= norm;
}

/****************************************************************************/
/* 'cRandomPolarCap' class for generating random numbers from a spherical   */
/*  cap of size 'size' and pole 'u' in R^d.                                 */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cRandomPolarCap {
	int dim;
	std::normal_distribution<double> normal;
	std::uniform_real_distribution<double> unif;
public:
	double size;
	// constructor, 'aDim' is the dimension of the ambient space, 'size'
	// the size of the spherical cap in terms of the polar angle
	cRandomPolarCap(int aDim, double aSize) : dim(aDim), size(aSize) { };
	template <class URNG> double* operator()(URNG& gen, const double* u = nullptr);
	template <class URNG> void vector(double* const x, URNG& gen, const double* u = nullptr);
};

template <class URNG> double* cRandomPolarCap::operator()(URNG& gen, const double* u) {
	double* x = new double[dim];
	// the polar angle is randomly dran from a uniform distribution on [0,size]
	// the first coordinate of the random point is the cosine of the polar angle
	double c = cos(unif(gen) * size);
	x[0] = c;
	// draw the remaining d-1 coordinates from a uniform distribution on the 
	// sphere in R^{d-1} with radius sqrt(1-c^2)
	double norm{ 0 };
	for (int i = 1; i < dim; i++) {
		x[i] = normal(gen);
		norm += x[i] * x[i];
	}
	norm = sqrt((1 - c * c) / norm);
	for (int i = 1; i < dim; i++) x[i] *= norm;
	// if u is not the north pole, transform the cap aroun the nort pole to a
	// cap around u using a proper Householder transformation
	if (u != nullptr) Householder(x, u, dim);
	return x;
}

template <class URNG> void cRandomPolarCap::vector(double* const x, URNG& gen, const double* u) {
	// the polar angle is randomly dran from a uniform distribution on [0,size]
	// the first coordinate of the random point is the cosine of the polar angle
	double c = cos(unif(gen) * size);
	x[0] = c;
	// draw the remaining d-1 coordinates from a uniform distribution on the 
	// sphere in R^{d-1} with radius sqrt(1-c^2)
	double norm{ 0 };
	for (int i = 1; i < dim; i++) {
		x[i] = normal(gen);
		norm += x[i] * x[i];
	}
	norm = sqrt((1 - c * c) / norm);
	for (int i = 1; i < dim; i++) x[i] *= norm;
	// if u is not the north pole, transform the cap aroun the nort pole to a
	// cap around u using a proper Householder transformation
	if (u != nullptr) Householder(x, u, dim);
}

/****************************************************************************/
/* 'cHemisphericalShell' class for generating random numbers from a         */
/*  hemispherical shell in R^d.                                             */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cHemisphericalShell {
	int dim;
	double lo, hi;
	std::normal_distribution<double> normal;
	std::uniform_real_distribution<double> unif;
public:
	// constructor, 'aDim' is the dimension of the ambient space,
	// 'lo' and 'hi' are inner and outer radius of the shell
	cHemisphericalShell(int aDim, double aLo, double aHi) : dim(aDim), unif(aLo, aHi) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cHemisphericalShell::operator()(URNG& gen) {
	double rad = 0;
	double* x = new double[dim];
	for (int i = 0; i < dim; i++) {
		x[i] = normal(gen);
		rad += x[i] * x[i];
	}
	rad = unif(gen) / sqrt(rad);
	for (int i = 0; i < dim; i++) x[i] *= rad;
	x[dim - 1] = fabs(x[dim - 1]);
	return x;
}

template <class URNG> void cHemisphericalShell::vector(double* const x, URNG& gen) {
	double rad = 0;
	for (int i = 0; i < dim; i++) {
		x[i] = normal(gen);
		rad += x[i] * x[i];
	}
	rad = unif(gen) / sqrt(rad);
	for (int i = 0; i < dim; i++) x[i] *= rad;
	x[dim - 1] = fabs(x[dim - 1]);
}


/****************************************************************************/
/* 'cUniformCube' class for generating random numbers from the uniform      */
/*  distribution on the cube [-1,1]^d in R^d.                               */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cMUniformCube {
	int dim;
	std::uniform_real_distribution<> unif;
public:
	// constructor: 'aDim' is the dimension d of the ambient space
	cMUniformCube(int aDim) : dim(aDim), unif(-1.0,1.0) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cMUniformCube::operator()(URNG& gen) {
	double* x = new double[dim];
	// generate 'dim' independent U(-1,1) variates
	for (int i = 0; i < dim; i++) x[i] = unif(gen);
	return x;
}

template <class URNG> void cMUniformCube::vector(double* const x, URNG& gen) {
	// generate 'dim' independent U(-1,1) variates
	for (int i = 0; i < dim; i++) x[i] = unif(gen);
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

class cElliptic {
protected:
	bool standard;    // if 'standard' is true, then the standard version,
	                  // i.e, \mu = 0, \Sigma = I, is considered
	int dim;          // dimension of the space
	double* mu;       // location vector of the elliptic distribution
	double** sigma;   // 'sigma' contains the Cholesky decomposition of the
	                  // scale matrix \Sigma 
	// 'transform' preforms the transformation from a spherical distribution
	// to an an elliptic distribution
	void transform(double x[]) const;
public:
	// constructor for a spherical distribution
	cElliptic(int aDim);  
	// constructor for an elliptical distribution
	cElliptic(int aDim, double* aMu, double** aSigma);
	~cElliptic();
};

/****************************************************************************/
/* 'cMNormal' class for generating random numbers from the multivariate     */
/*  normal distribution in R^d.                                             */
/*                                                                          */
/* 'cMNormal' is derived from 'cElliptic'                                   */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cMNormal : public cElliptic {
protected:
	std::normal_distribution<double> normal;
public:
	// constructor for a multivariate standard normal distribution
	cMNormal(int aDim) : cElliptic(aDim) { };
	// constructor for a multivariate normal distribution with  general
	// paramters mu and Sigma
	cMNormal(int aDim, double* aMu, double** aSigma) : cElliptic(aDim, aMu, aSigma) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cMNormal::operator()(URNG& gen) {
	double* x = new double[dim];
	// generate 'dim' independent standard normal variates
	for (int i = 0; i < dim; i++) x[i] = normal(gen);
	// if necessary transform to an elliptical distribution
	if (!standard) transform(x);
	return x;
}

template <class URNG> void cMNormal::vector(double* const x, URNG& gen) {
	// generate 'dim' independent standard normal variates
	for (int i = 0; i < dim; i++) x[i] = normal(gen);
	// if necessary transform to an elliptical distribution
	if (!standard) transform(x);
}

/****************************************************************************/
/* 'cBimodalNormal' class for generating random numbers from a mixture of   */
/*  two multivariate normal distributions with equal weights in R^d.        */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cBimodalNormal : public cMNormal {
protected:
	double dist;
	std::uniform_real_distribution<double> unif;
public:
	// constructor: 'aDim' is the dimension d of the ambient space
	// 'aDist' is the shifting value. The first component is shifted
	// by aDist * e_1, the second by -aDist * e_1. (e_1 is the first
	// canonical unit vector.             
	cBimodalNormal(int aDim, double aDist) : cMNormal(aDim), dist(aDist) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cBimodalNormal::operator()(URNG& gen) {
	double* x = new double[dim];
	cMNormal::vector(x, gen);
	if (unif(gen) > 0.5) x[0] += dist; else x[0] -= dist;
	return x;
}

template <class URNG> void cBimodalNormal::vector(double* const x, URNG& gen) {
	cMNormal::vector(x, gen);
	if (unif(gen) > 0.5) x[0] += dist; else x[0] -= dist;
}

/****************************************************************************/
/* 'cMultimodalNormal' class for generating random numbers from a mixture   */
/*  of d multivariate normal distributions with equal weights in R^d.       */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cMultimodalNormal : public cMNormal {
protected:
	double dist;
	std::uniform_int_distribution<int> unif;
public:
	// constructor: 'aDim' is the dimension d of the ambient space
	// 'aDist' is the shifting value. The j-th component is shifted
	// by aDist * e_j, where e_j is the first canonical unit vector.             
	cMultimodalNormal(int aDim, double aDist) : cMNormal(aDim), unif(0, aDim - 1), dist(aDist) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cMultimodalNormal::operator()(URNG& gen) {
	double* x = new double[dim];
	cMNormal::vector(x, gen);
	x[unif(gen)] += dist;
	return x;
}

template <class URNG> void cMultimodalNormal::vector(double* const x, URNG& gen) {
	cMNormal::vector(x, gen);
	x[unif(gen)] += dist;
}

/****************************************************************************/
/* 'cMCauchy' class for generating random numbers from the multivariate     */
/*  Cauchy distribution in R^d.                                             */
/*                                                                          */
/* 'cMCauchy' is derived from 'cMNormal'                                    */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cMCauchy : public cMNormal {
public:
	// constructor for a multivariate standard Cauchy distribution
	cMCauchy(int aDim) : cMNormal(aDim) { };
	// constructor for a multivariate Cauchy distribution with  general
	// location parameter mu and scale matrix Sigma
	cMCauchy(int aDim, double* aMu, double** aSigma) : cMNormal(aDim, aMu, aSigma) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cMCauchy::operator()(URNG& gen) {
	double* x = new double[dim];
	double s = fabs(normal(gen));
	// generate 'dim' independent standard normal variates and divide them by
	// an independent standard nmormal variate s
	for (int i = 0; i < dim; i++) x[i] = normal(gen) / s;
	// if necessary transform to an elliptical distribution
	if (!standard) transform(x);
	return x;
}

template <class URNG> void cMCauchy::vector(double* const x, URNG& gen) {
	double s = fabs(normal(gen));
	// generate 'dim' independent standard normal variates and divide them by
	// an independent standard nmormal variate s
	for (int i = 0; i < dim; i++) x[i] = normal(gen) / s;
	// if necessary transform to an elliptical distribution
	if (!standard) transform(x);
}

/****************************************************************************/
/* 'cMt' class for generating random numbers from the multivariate          */
/*  t distribution in R^d.                                                  */
/*                                                                          */
/* 'cMt' is derived from 'cMNormal'                                         */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cMt : public cMNormal {
private:
	int _df;  // degrees of freedom
	std::chi_squared_distribution<double> _chisqr;
public:
	// constructor for a multivariate standard t-distribution
	cMt(int aDim, int df) : cMNormal(aDim), _chisqr(df), _df(df) { };
	// constructor for a multivariate t-distribution with general
	// location parameter mu and scale matrix Sigma
	cMt(int aDim, int df, double* aMu, double** aSigma) : cMNormal(aDim, aMu, aSigma), _df(df), _chisqr(df) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cMt::operator()(URNG& gen) {
	double* x = new double[dim];
	// s is the square root of a chi squared variate divided by its degrees of freedom
	double s = sqrt(_chisqr(gen) / _df);
	// generate 'dim' independent standard normal variates and divide them by s
	for (int i = 0; i < dim; i++) x[i] = normal(gen) / s;
	// if necessary transform to an elliptical distribution
	if (!standard) transform(x);
	return x;
}

template <class URNG> void cMt::vector(double* const x, URNG& gen) {
	// s is the square root of a chi squared variate divided by its degrees of freedom
	double s = sqrt(_chisqr(gen) / _df);
	// generate 'dim' independent standard normal variates and divide them by s
	for (int i = 0; i < dim; i++) x[i] = normal(gen) / s;
	// if necessary transform to an elliptical distribution
	if (!standard) transform(x);
}

/****************************************************************************/
/* 'cMSkewNormal' class for generating random numbers from the multivariate */
/*  skew normal distribution in R^d.                                        */
/*                                                                          */
/* 'cMSkewNormal' is derived from 'cMNormal'                                */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cMSkewNormal : public cMNormal {
private:
	std::unique_ptr<double[]> _delta; // multivariate skewness parameter
	std::uniform_real_distribution<double> _unif;
public:
	// constructor for a multivariate standard skew normal distribution 
	cMSkewNormal(int aDim, double* delta) : cMNormal(aDim), _unif(0, 1), _delta{ delta } { };
	// constructor for a multivariate skew normal distribution with general
	// location parameter mu and scale matrix Sigma
	cMSkewNormal(int aDim, double* delta, double* aMu, double** aSigma) : cMNormal(aDim, aMu, aSigma), _unif(0, 1), _delta{ delta } { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cMSkewNormal::operator()(URNG& gen) {
	double u = _unif(gen);
	double* x = new double[dim];
	// generate 'dim' independent standard normal variates
	for (int i = 0; i < dim; i++) x[i] = normal(gen);
	// compute inner product of vector x and the skewness parameter
	double sum{ 0 };
	for (int i = 0; i < dim; i++) sum += _delta[i] * x[i];
	// with probability 1-cdfn(sum) invert the signs of x
	if (u > cdfn(sum)) {
		for (int i = 0; i < dim; i++) x[i] = -x[i];
	}
	// if necessary transform with location parameter mu and scale matrix Sigma
	if (!standard) transform(x);
	return x;
}

template <class URNG> void cMSkewNormal::vector(double* const x, URNG& gen) {
	double u = _unif(gen);
	// generate 'dim' independent standard normal variates
	for (int i = 0; i < dim; i++) x[i] = normal(gen);
	// compute inner product of vector x and the skewness parameter
	double sum{ 0 };
	for (int i = 0; i < dim; i++) sum += _delta[i] * x[i];
	// with probability 1-cdfn(sum) invert the signs of x
	if (u > cdfn(sum)) {
		for (int i = 0; i < dim; i++) x[i] = -x[i];
	}
	// if necessary transform with location parameter mu and scale matrix Sigma
	if (!standard) transform(x);
}

/****************************************************************************/
/* 'cMExponential' class for generating random numbers from the             */
/*  multivariate exponential distribution (with independent marginals)      */
/*  in R^d.                                                                 */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cMExponential {
	int dim;
	std::exponential_distribution<double> _exp;
	std::unique_ptr<double[]> _lambda;
public:
	// constructor for a multivariate standard exponential distribution, 
	// i.e., margins are independently Exp(1) distributed
	cMExponential(int aDim) : dim{ aDim }, _exp(), _lambda{ new double[aDim] } {
		for (int i = 0; i < dim; i++) _lambda[i] = 1.0;
	};
	// constructor for a multivariate exponential distribution, 
	// i.e., margins are independently Exp(\lambda_i) distributed
	cMExponential(int aDim, double* lambda) : dim{ aDim }, _exp(), _lambda{ lambda } {};
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cMExponential::operator()(URNG& gen) {
	double* x = new double[dim];
	// generate 'dim' independent exponential variates
	for (int i = 0; i < dim; i++) x[i] = _exp(gen) / _lambda[i];
	return x;
}

template <class URNG> void cMExponential::vector(double* const x, URNG& gen) {
	// generate 'dim' independent exponential variates
	for (int i = 0; i < dim; i++) x[i] = _exp(gen) / _lambda[i];
}

/****************************************************************************/
/* 'cDirichletSym' class for generating random numbers from the symmetric   */
/*  Dirichlet distribution in R^d.                                          */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cDirichletSym {
private:
	int dim;
	double alpha;
	std::gamma_distribution<double> gamma;
public:
	// constructor for a symmetric Dirichlet distribution, 
	cDirichletSym(int aDim, double alpha) : dim(aDim), gamma(alpha, 1.) { };
	template <class URNG> double* operator()(URNG& gen);
	template <class URNG> void vector(double* const x, URNG& gen);
};

template <class URNG> double* cDirichletSym::operator()(URNG& gen) {
	double* x = new double[dim];
	vector(x, gen);
	return x;
}

template <class URNG> void cDirichletSym::vector(double* const x, URNG& gen) {
	double sum = 0;
	// generate 'dim' independent gamma variates and compute their sum
	for (int i = 0; i < dim; i++) {
		x[i] = gamma(gen);
		sum += x[i];
	}
	// divide the gamma variates by their sum
	for (int i = 0; i < dim; i++) x[i] /= sum;
}

/****************************************************************************/
/* 'cCombination' class for generating a random combination without         */
/*  replacement of size k from the numbers 0:n-1.                           */
/*                                                                          */
/* Two public methods are defined for the generation of random vectors.     */
/*   1) The first method returns a vector of dimension d. The memory for    */
/*      the vector is allocated by the called routine.                      */
/*   2) The second method has no return value. Instead, the random vector   */
/*      is returned in an output parameter. The memory for this vector is   */
/*      allocated by the calling routine.                                   */
/****************************************************************************/

class cCombination {
private:
	int n;
	int k;
	std::uniform_int_distribution<int>** unifints;
public:
	cCombination(int aN, int aK);  // constructor
	template <class URNG> int* operator()(URNG& gen);
	template <class URNG> void vector(int* const x, URNG& gen);
	~cCombination();               // destructor
};

template <class URNG> int* cCombination::operator()(URNG& gen) {
	int* x = new int[k];
	vector(x, gen);
	return x;
}

template <class URNG> void cCombination::vector(int* const x, URNG& gen) {
	for (int i = 0; i < k; i++) {
		x[i] = unifints[i]->operator()(gen);
	}
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++) {
			if (x[i] >= x[j]) { x[j]++; }
		}
	}
}

#endif


