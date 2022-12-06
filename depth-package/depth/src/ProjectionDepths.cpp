/******************************************************************************/
/* File:             ProjectionDepths.cpp                                     */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Implementation of the methods in class 'cProjection'                       */
/*                                                                            */
/* Further, some helper functions are defined.                                */
/*                                                                            */
/******************************************************************************/

#define _USE_MATH_DEFINES
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cmath>
#include <cfloat>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "MD.h"
#include "ZD.h"
#include "HD.h"
#include "PD.h"
#include "APD.h"
#include "ProjectionDepths.h"
#include "Matrix.h"
#include "auxLinAlg.h"
#include "mvrandom.h"


using namespace std;
using namespace DataDepth;
using namespace dyMatrixClass;

// constant 'debug' controls the debug level, possible values are 
//     0 : no debugging
//     1 : statistics regarding depth values
//     2 : debug iterations (console output)
//     3 : debug methods (console output)
const int debug = 1;

// seed for the RNG
#define _seed 1234

// error indicator used when computing the zonoid depth
int error;

// Compare functions for objects of class 'Feval' and 'fVal', repsectively
// these functions are used for sorting corresponding arrays
int Compare(Feval& fe1, Feval& fe2) { return fe1.val < fe2.val; }
int cmp(const fVal& a, const fVal& b) { return a.val < b.val; }

// array of functions for exactly computing multivariate depths
// For PD and APD no exact algorithms are implemented, so we return always -1
function<double(const double*, const cMatrix, int, int)> multiDepths[]
= { 
	[](const double* z, const cMatrix x, int n, int d) { return MD(z, x, n, d); }, 
	[](const double* z, const cMatrix x, int n, int d) { return HD_Comb2(z, x, n, d); },
	[](const double* z, const cMatrix x, int n, int d) { return ZD(z, x, n, d, error); },
	[](const double* z, const cMatrix x, int n, int d) { return -1.; },
	[](const double* z, const cMatrix x, int n, int d) { return -1.; } };

// array of functions for exactly computing univariate depths
function<double(double, const double*, int)> uniDepths[] 
	= { MD1, HD1, 
	[](double z, const double* x, int n){ return ZD1(z, x, n, false); }, PD1, APD1 };


/****************************************************************************/
/*                                                                          */
/* 'cProjection' is the main class for computing several projection depths  */
/*   using different approximation strategies.                              */
/*                                                                          */
/****************************************************************************/

// constructor

cProjection::cProjection(const cMatrix& x, int n, int d, int NRandom)
	: x{ x }, n{ n }, d{ d }, _nProjections{ 0 }, _nRandom{ NRandom }, gen{ _seed }, rnd{ 0.0, 1.0 },
xp{ new double[n] {} }
{
	// initialization of array 'Method' that contains the different approximations methods
	Method[0] = [this](const double* z) { return SimpleGrid(z); };
	Method[1] = [this](const double* z) { return RefinedGrid(z); };
	Method[2] = [this](const double* z) { return SimpleRandom(z); };
	Method[3] = [this](const double* z) { return RefinedRandom(z); };
	Method[4] = [this](const double* z) { return CoordinateDescentMultiStart(z); };
	Method[5] = [this](const double* z) { return RandomSimplices(z); };
	Method[6] = [this](const double* z) { return NelderMeadMultiStart(z); };
	Method[7] = [this](const double* z) { return SimulatedAnnealingMultiStart(z); };
	Method[8] = [this](const double* z) { return CoordinateDescentGCMultiStart(z); };
	Method[9] = [this](const double* z) { return NelderMeadGCMultiStart(z); };
	// initialization of array 'MethodCD' that contains the different line search strategies for CoordDesc
	MethodCD[0] = [this](const double* z, double* u, double* dir) { return LineSearchUnif(z, u, dir); };
	MethodCD[1] = [this](const double* z, double* u, double* dir) { return LineSearchGS(z, u, dir); };
	// initialization of array 'MethodCDGC' that contains the different line search strategies for CoordDescGC
	MethodCDGC[0] = [this](const double* z, double* u, double* dir) { return LineSearchUnifGC(z, u, dir); };
	MethodCDGC[1] = [this](const double* z, double* u, double* dir) { return LineSearchGSGC(z, u, dir); };
	records = nullptr;
}

// destructor

cProjection::~cProjection() {
	delete[] records;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::initStatistics': initializes the structures used for keeping  */
/*   track of the development of depth values during an approximation          */
/*                                                                             */
/*******************************************************************************/

void cProjection::initStatistics() {
	// Restart counters
	_nProjections = 0;
	_Depths.clear();
	_MinDepths.clear();
	_BestDirection.clear();
	_Directions.clear();
	_DirectionsCard.clear();
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::SetDepthNotion': setter method for the depth notion '_depth'  */
/*                                                                             */
/*******************************************************************************/

void cProjection::SetDepthNotion(eDepth notion) {
	_depth = notion;
	// set 'UniDepth' according to the selected depth notion
	UniDepth = uniDepths[(int)notion];
	// set 'MultiDepth' according to the selected depth notion
	MultiDepth = multiDepths[(int)notion];
	if(d == 1){for (int i = 0; i < n; i++)xp[i] = x[0][i];}
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::Depth': computes the multivariate depth of a point 'z' for    */
/*   the currently selected depth notion and using the currently selected      */
/*   approximation method; further, records some statistics                    */
/*                                                                             */
/*******************************************************************************/

double cProjection::Depth(const double* z) {
	initStatistics();
	double depth;
	// Calculate approximate depth recording time
	clock_t starttime = clock();
	if(d == 1){
		double zp = z[0];
		depth = UniDepth(zp, xp.get(), n); // depth exact calculation for d=1
	}
	else{
		depth = Method[(int)_Method](z); // depth approximation calculation for d>1
	}
	clock_t laptime = clock() - starttime;
	_lastDepthDuration = (double)laptime / CLOCKS_PER_SEC;
	return depth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::ExactDepth': exactly computes the multivariate depth of a     */
/*   point 'z' using the currently selected depth notion; further, records     */
/*   some statistics. if no exact algorithm is implemented -1 is returned.     */
/*                                                                             */
/*******************************************************************************/

double cProjection::ExactDepth(const double* z) {
	clock_t starttime = clock();
	double depth = MultiDepth(z, x, n, d);
	clock_t laptime = clock() - starttime;
	_lastDepthDuration = (double)laptime / CLOCKS_PER_SEC;
	return depth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::ProjectedDepth': project 'z' in direction 'u' and compute its */
/*   univariate depth for the currently selected depth notion; further,        */
/*   records some statistics.                                                  */  
/*                                                                             */
/*******************************************************************************/

double cProjection::ProjectedDepth(const double* z, const double u[]) {

	// Increment number of used direcions (= projections)
	_nProjections++;
	// compute projections of data points 'x[i]' in direction 'u'
	for (int i = 0; i < n; i++) xp[i] = InnerProduct(x[i], u, d);
	// compute projection of 'z' in direction 'u'
	double zp = InnerProduct(z, u, d);
	// compute the univariate depth
	double prjDepth = UniDepth(zp, xp.get(), n);
	// save statistics (if necessary)
	if (debug >= 1) {
        switch(_option){
	    case 1:
		    break;
        case 2:
            if (_nProjections <= 1 || prjDepth < _MinDepths[_nProjections - 2]) {
                _MinDepths.push_back(prjDepth);
                _BestDirection = std::vector<double>(u, u + d);
            }
            else {
                _MinDepths.push_back(_MinDepths[_nProjections - 2]);
            }
            break;
        case 3:
            _Depths.push_back(prjDepth);
            if (_nProjections <= 1 || prjDepth < _MinDepths[_nProjections - 2]) {
                _MinDepths.push_back(prjDepth);
                _BestDirection = std::vector<double>(u, u + d);
            }
            else {
                _MinDepths.push_back(_MinDepths[_nProjections - 2]);
            }
            break;
        case 4:
            _Depths.push_back(prjDepth);
            _Directions.push_back(std::vector<double>(u, u + d));
            if (_nProjections <= 1 || prjDepth < _MinDepths[_nProjections - 2]) {
                _MinDepths.push_back(prjDepth);
                _BestDirection = std::vector<double>(u, u + d);
            }
            else {
                _MinDepths.push_back(_MinDepths[_nProjections - 2]);
            }
            break;
        }
	}
	// return computed univeriate depth
	return prjDepth;
}

/*******************************************************************************/
/*                                                                             */ 
/* 'cProjection::SimpleRandom': simple random search (RS)                      */
/*   This is essentially 'Algorithm 2' in the article 'Approximate computation */
/*   of projection depths'                                                     */
/*                                                                             */
/*******************************************************************************/

double cProjection::SimpleRandom(const double* z) {
	// random numbers from a uniform distribution on the sphere
	cUniformSphere rndSphere(d);  
	double minDepth = DBL_MAX;
	_nProjections = 0;
	for (int i = 0; i < _nRandom; i++) {
		unique_ptr<double[]> u{ rndSphere(gen) };
		double depth = ProjectedDepth(z, u.get());
		minDepth = min(depth, minDepth);
	}
	if (debug >= 2) cout << "RS:   " << _nProjections << endl;
	return minDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::RefinedRandom': refined random search (RRS)                   */
/*   This is essentially 'Algorithm 5' in the article 'Approximate computation */
/*   of projection depths'                                                     */
/*                                                                             */
/*******************************************************************************/

double cProjection::RefinedRandom(const double* z) {
	nRecords = 0;
	records = nullptr;
	unique_ptr<double[]> uOpt{ new double[d] {} };
	// We start with a spherical cap that is an entire hemisphere
	cRandomPolarCap rndPolarCap(d, M_PI / 2); 
	// and that is centered at the north pole, i.e, u_opt = (1,0,...,0)
	uOpt[0] = 1;                              
	double MinDepth{ DBL_MAX };
	int Count{};
	int Refinement{};
	_nProjections = 0;
	int MaxDirs = 0;
	double eps{ M_PI / 2 };
	do {
		// start a new refinement step
		Refinement++;
		// make sure to run exactly _nrandom directions
		if(Refinement < _maxRefinesRand){
			MaxDirs = _nRandom / _maxRefinesRand;
		}
		else{
			MaxDirs = _nRandom - _nProjections;
		}
		// perform a random search in the spherical cap
		for (int i = 0; i < MaxDirs; i++) {
			unique_ptr<double[]> u{ rndPolarCap(gen, uOpt.get()) };
			Count++;
			double depth{ ProjectedDepth(z, u.get()) };
			if (depth < MinDepth) {
				// if a new minimum is found we search in the spherical
				// cap centered at this new minimum
				MinDepth = depth;
				uOpt = move(u);
			}
		}
		// decrease the size of the spherical cap
		eps *= _alphaRand;   
		rndPolarCap.size *= _alphaRand;
	} while (Refinement < _maxRefinesRand);
	if (debug >= 2) cout << "RRS:  " << _nProjections << endl;
	return MinDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'nextGridPoint' is used to enumerate all points of a k-dimensional grid     */
/*   on the unit sphere. The values in 'grid' are later multiplied with some   */
/*   factors to give a point on the unit sphere expressed in spherical         */
/*   coordinates, phi_1,...,phi_k. This routine also takes care of the prolem  */
/*   that if some angle phi_j = 0, then regardless of the values phi_{j+1},    */ 
/*   ...,phi_k we always get the same point on the sphere. This routine avoids */
/*   the problem of generating the same point on the sphere over and over      */
/*   again. At the beginning 'grid[i] = 0' should hold for all i. The array    */
/*   'limits' gives the maximum values that we want to achieve at the          */
/*   corresponding position.                                                   */
/*                                                                             */
/* This routine is called in 'cProjection::GridSearch'                         */
/*                                                                             */
/*******************************************************************************/

bool nextGridPoint(int grid[], int limits[], int k) {
	int index{ 0 };
	while ((index < k) && (grid[index] != 0)) index++;
	if (index < k) {
		grid[index]++;
		return true;
	}
	index = k - 1;
	while ((index >= 0) && (grid[index] == limits[index])) index--;
	if (index < 0) return false;
	grid[index]++;
	for (int i = index + 1; i < k; i++) grid[i] = 0;
	return true;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::GridSearch': this is the actual grid search which is used     */
/*   both for the simple grid search (GS) and for the refined grid search      */
/*   (RGS). We construct a grid in spherical coordinates over a spherical cap  */
/*   with size 'size' centered at 'u'. The grid has 'nStep' subdivisions in    */
/*   each dimension, apart from the last dimension where we have '2*nStep'     */
/*   subdivisions. 'z' is the point for which we want to compute the depth.    */
/*                                                                             */ 
/* Args:																	   */
/*   z - point for which the depth has to be computed						   */
/*   u - center of the spherical cap that is searched						   */
/*   size - size of the spherical cap in terms of the polar angle			   */
/*   nStep - number of steps in the first d-2 dimensions, in the last		   */
/*           dimension there are 2*nStep steps								   */
/*                                                                             */
/* This routine is called in 'cProjection::SimpleGrid' (GS) and                */
/*   'cProjection::RefinedGrid' (RGS)                                          */
/*                                                                             */
/*******************************************************************************/

double cProjection::GridSearch(const double* z, double* u, const double size, int nStep) {
	// initialize the following arrays
	unique_ptr<double[]> phi{ new double[d - 1]{} };
	unique_ptr<double[]> uOpt{ new double[d]{} };
	unique_ptr<int[]> grid{ new int[d - 1]{} };
	unique_ptr<int[]> limits{ new int[d - 1]{} };
	// set the limits for the grid points
	for (int i = 0; i < d - 2; i++) limits[i] = nStep;
	limits[d - 2] = 2 * nStep;
	double deltaPhi{ M_PI / nStep };  // step size 
	double MinDepth{ DBL_MAX };
	do {
		// Compute the next grid point in spherical coordinates
		phi[0] = grid[0] * size / nStep;
		for (int i = 1; i < d - 1; i++) phi[i] = grid[i] * deltaPhi;
		// Transform to Cartesian coordinates
		unique_ptr<double[]> x = SphericalToCartesian(phi.get(), d);
		// Householder transformation that transforms the north pole to u
		Householder(x.get(), u, d);
		// Compute the univariate depth
		double depth = ProjectedDepth(z, x.get());
		if (depth < MinDepth) {
			MinDepth = depth;
			for (int i = 0; i < d; i++) uOpt[i] = x[i];
		}
	} while (nextGridPoint(grid.get(), limits.get(), d - 1));
	// Best direction is returned in u
	for (int i = 0; i < d; i++) u[i] = uOpt[i];
	return MinDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::SimpleGrid': simple grid search (GS)                          */
/*   For the simple grid search (GS) we call 'GridSearch' to search a		   */
/*   spherical cap of size 'M_PI/2', i.e., a hemisphere, and a randomly		   */
/*   chosen center 'u'. The number of subdivisions is chosen such that		   */
/*   we get approximately '_nRandom' grid points.							   */
/*                                                                             */
/*******************************************************************************/

double cProjection::SimpleGrid(const double* z) {
	_nProjections = 0;
	// For 'd > 10 the grid is to coarse. Therefore, we perform a grid
	// only for dimensions D <= 10. If d > 10 the value 2 is returned.
	if (d > 10) return 2;
	cUniformSphere rndSphere(d);
	unique_ptr<double[]> u{ rndSphere(gen) };
	int nStep = round(pow(_nRandom / 2, 1.0 / (d - 1))) - 1;
	// if the computation of 'nStep' yields a value of 0, then a grid
	// search for which the number of grid points is approximately 
	// '_nRandom' is not possible. To signal this, we return 2.
	if (nStep == 0) return 2;
	double result = GridSearch(z, u.get(), M_PI / 2, nStep);
	if (debug >= 2) cout << "GS:   " << _nProjections << endl;
	return result;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::RefinedGrid': refined grid search (RGS)                       */
/*   In the refined grid search (RGS) we search over grids on spherical caps   */
/*   which are centered at the current best point and whose size is decreased  */
/*   in each refinement step.                                                  */
/*                                                                             */
/*******************************************************************************/

double cProjection::RefinedGrid(const double* z) {
	_nProjections = 0;
	// For 'd > 10 the grid is to coarse. Therefore, we perform a grid
	// only for dimensions D <= 10. If d > 10 the value 2 is returned.
	if (d > 10) return 2;
	// We start with a sphericak cap that is an entire hemisphere
	double size{ M_PI / 2 }; 
	// and that is centered at a randomly chosen point on the whole sphere
	cUniformSphere rndSphere(d);
	unique_ptr<double[]> u{ rndSphere(gen) };
	int nStep = round(pow((_nRandom / 2) / _maxRefinesGrid, 1.0 / (d - 1))) - 1;
	// if the computation of 'nStep' yields a value of 0, then a grid
	// search for which the number of grid points is approximately 
	// '_nRandom' is not possible. To signal this, we return 2.
	if (nStep == 0) return 2;
	double minDepth{ DBL_MAX };
	for (int count = 0; count < _maxRefinesGrid; count++) {
		// search a grid over a spherical cap of size 'size' centered at 'u'
		minDepth = min(GridSearch(z, u.get(), size, nStep), minDepth);
		// decrease the size of the spherical cap
		size *= _alphaGrid;
	};
	if (debug >= 2) cout << "RGS:  " << _nProjections << endl;
	return minDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::RandomSimplices': random simplices (RaSi)                     */
/*   This is essentially 'Algorithm 6' in the article 'Approximate computation */
/*   of projection depths'                                                     */
/*                                                                             */
/*******************************************************************************/

double cProjection::RandomSimplices(const double* z) {
	cCombination rndCombination(n, d + 1);
	cDirichletSym rndDirichletSym(d, _alphaRaSi);
	_nProjections = 0;
	double minDepth = DBL_MAX;
	int* cmb = new int[d + 1];
	double* weights = new double[d];
	double* u = new double[d];
	for (int i = 0; i < d; i++) u[i] = 0;
	for (int i = 0; i < _nRandom; i++) {
		// choose d+1 data points at random
		rndCombination.vector(cmb, gen);
		// choose the weights from a symmetric Dirichlet distribution
		rndDirichletSym.vector(weights, gen); 
		// compute the direction which goes from the last of the selected
		// points to the convex combination of the first d selected points
		// that is defined by the chosen weights
		double sqSum = 0;
		for (int j = 0; j < d; j++) {
			for (int k = 0; k < d; k++)
				u[j] += weights[k] * x[cmb[k]][j];
			u[j] -= x[cmb[d]][j];
			sqSum += pow(u[j], 2);
		}
		sqSum = sqrt(sqSum);
		// normalize the direction
		for (int j = 0; j < d; j++) u[j] /= sqSum;
		// compute the univariate depth of point 'z' in direction 'u'
		double depth = ProjectedDepth(z, u);
		minDepth = min(depth, minDepth);
	}
	// Release memory
	delete[] cmb;
	delete[] weights;
	delete[] u;
	if (debug >= 2) cout << "RaSi: " << _nProjections << endl;
	return minDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::SimulatedAnnealing': simulated annealing (SA)                 */
/*   This is essentially 'Algorithm 7' in the article 'Approximate computation */
/*   of projection depths'                                                     */
/*                                                                             */
/* This routine is called in 'cProjection::SimulatedAnnealingMultiStart'       */
/*                                                                             */
/*******************************************************************************/

double cProjection::SimulatedAnnealing(const double* z) {
	const double TMax = 1;      // temperature at the start of the algorithm
	const double TMin = 0.001;  // temperature at which to stop the algorithm
	// at the beginning the neiughborhood is a spherical cap of size (M_PI / 2)
	// divided by _betaSA
	cRandomPolarCap rndPolarCap(d, (M_PI / 2) / _betaSA);
	cUniformSphere rndSphere(d);
	// compute 'nIt', the number of iterations before the size of the
	// neighborhood is decreased, such that the number of depth evaluations
	// is approximately equal to '_nRandom'
	int nIt = round(_nRandom / (log(TMin / TMax) / log(_alphaSA)));
	nIt = min(nIt, _nRandom);
	double T{ TMax };
	// the starting value is chosen according to the parameter '_startSA'
	// _startSA = 0 => starting point is the mean of the data points
	// _startSA = 1 => starting point is randomly chosen
	unique_ptr<double[]> u{ rndSphere(gen) };
	if (_startSA == 0) {
		unique_ptr<double[]> xquer{ mean(x, n, d) };
		if (distance(xquer.get(), z, d) >= 1e-8) {
			for (int i = 0; i < d; i++)  u[i] = z[i] - xquer[i];
			Normalize(u.get(), d);
		}
	}
	double depth{ ProjectedDepth(z, u.get()) };
	double MinDepth{ depth };
	do {
		nIt = min(nIt, _nRandom - _nProjections);
		for (int l = 1; l <= nIt; l++) {
			// randomly choose a new point 'uNew' from the spherical 
			// cap around 'u'
			unique_ptr<double[]> uNew{ rndPolarCap(gen, u.get()) };
			// and compute the univariate depth of 'z' in direction 'uNew'
			double depth_new{ ProjectedDepth(z, uNew.get()) };
			MinDepth = min(depth_new, MinDepth);
			// accept the new direction'uNew' with a certain probability 'r'
			double r{ exp((depth - depth_new) / T) };
			if (rnd(gen) <= r) {
				u = move(uNew);
				depth = depth_new;
			}
		}
		T *= _alphaSA;
	} while ((T >= TMin) && (_nProjections < _nRandom));
	// loop ends if temperatue is lower than 'TMin' or if the maximum numbe
	// iterations 'NRandom' is reached 
	return MinDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::SimulatedAnnealingMultiStart'                                 */
/*   In applying 'SA' in our simulations we use a multistart strategy.         */
/*   in order to come to a fair comparison of the methods we want to           */
/*   guarantee that for every method we have the same number '_nRandom'        */
/*   of univariate depth evaluations. Therefore, if the actual number of       */
/*   iterations in the first run of SA is less than '_nRandom', we start       */
/*   the algorithm again. We do this as long as the maximum number of          */
/*   univariate depth evaluations is not reached.                              */
/*                                                                             */
/*******************************************************************************/

double cProjection::SimulatedAnnealingMultiStart(const double* z) {
	double minDepth{ DBL_MAX };
	_nProjections = 0;
	int DirectionsSize = 0;
	do {
		minDepth = min(SimulatedAnnealing(z), minDepth);
        if(_option == 4){
		_DirectionsCard.push_back(_Directions.size() - DirectionsSize);
		DirectionsSize = _Directions.size();
        }
	} while (_nProjections < _nRandom);
	if (debug >= 2) cout << "SA:   " << _nProjections << "  " << minDepth << endl;
	return minDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::LineSearchUnif'                                               */
/*   This routine performs a line search over a uniform grid. The line         */
/*   along which the search proceeds goes through the point 'u' on the         */
/*   unit sphere and proceeds along the direction 'dir'. However, since        */
/*   'dir' needs not to be a direction in the tangent hyperplane at 'u' we     */
/*   decompose 'dir' in a direction 'v' in the tangent hyperplane and a        */
/*   direction orthogonal to the tangent hyperplane. Then, we only use the     */
/*   direction 'v' and search along a great semi-circle defined by 'u' and     */
/*   'v'. On this great semi-circle we put a unform grid and search for the    */
/*   minimum.                                                                  */
/*                                                                             */
/* Args:																	   */
/*   z - point for which the depth has to be computed					 	   */
/*   u - point on the unit sphere from wehich the line search starts           */
/*   dir - direction of the line search, dir is an arbitrary direction in R^d  */
/*                                                                             */
/* This routine is called in 'cProjection::CoordinateDescent'                  */
/*                                                                             */
/*******************************************************************************/

double cProjection::LineSearchUnif(const double* z, double* u, double* dir) {
	unique_ptr<double[]> v{ new double[d] };
	unique_ptr<double[]> uOpt{ new double[d] };
	// find the direction v in the tangent hyperplane at point u
	double s = InnerProduct(u, dir, d);
	for (int i = 0; i < d; i++) v[i] = dir[i] - s * u[i];
	s = norm2(v.get(), d);
	for (int i = 0; i < d; i++) v[i] /= s;
	// perform the search along a great semi-circle defined by u and the direction v
	// the grid consists of _nLineSearch + 1 points
	double MinDepth{ DBL_MAX };
	for (int i = 0; i <= _nLineSearch; i++) {
		if (_nProjections < _nRandom) {
			double lambda{ -M_PI / 2 + i * M_PI / _nLineSearch };
			unique_ptr<double[]> w{ new double[d] };
			// construct the i-th grid point on the semi-circle
			for (int k = 0; k < d; k++) w[k] = cos(lambda) * u[k] + sin(lambda) * v[k];
			// compute the univariate depth of 'z' in direction 'w'
			double f = ProjectedDepth(z, w.get());
			if (f < MinDepth) {
				MinDepth = f;
				uOpt = move(w);
			}
		}
	}
	for (int i = 0; i < d; i++) u[i] = uOpt[i];
	return MinDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::LineSearchGS'                                                 */
/*   This routine performs a line search using the golden-section search.      */
/*   The line along which the search proceeds goes through the point 'u' on    */
/*   the unit sphere and proceeds along the direction 'dir'. However, since    */
/*   'dir' needs not to be a direction in the tangent hyperplane at 'u' we     */
/*   decompose 'dir' in a direction 'v' in the tangent hyperplane and a        */
/*   direction orthogonal to the tangent hyperplane. Then, we only use the     */
/*   direction 'v' and search along a great semi-circle defined by 'u' and     */
/*   'v'. On this great semi-circle a golden-section search for the minimum    */
/*   is performed.                                                             */
/*                                                                             */
/* Args:																	   */
/*   z - point for which the depth has to be computed					 	   */
/*   u - point on the unit sphere from wehich the line search starts           */
/*   dir - direction of the line search, dir is an arbitrary direction in R^d  */
/*                                                                             */
/* This routine is called in 'cProjection::CoordinateDescent'                  */
/*                                                                             */
/*******************************************************************************/

double cProjection::LineSearchGS(const double* z, double* u, double* dir) {
	const double alpha{ (sqrt(5) - 1) / 2 };
	unique_ptr<double[]> v{ new double[d] };
	// find the direction v in the tangent hyperplane at point u
	double s = InnerProduct(u, dir, d);
	for (int i = 0; i < d; i++) v[i] = dir[i] - s * u[i];
	s = norm2(v.get(), d);
	for (int i = 0; i < d; i++) v[i] /= s;
	// perform a golden-section search along a great semi-circle defined by u 
	//and the direction v
	double a{ -M_PI / 2 }, b{ M_PI / 2 };
	double lambda{ a + (1 - alpha)*(b - a) };
	double mu{ a + alpha*(b - a) };

	double c1 = cos(lambda);
	double c2 = sin(lambda);
	unique_ptr<double[]> w{ new double[d] };
	for (int i = 0; i < d; i++) w[i] = c1*u[i] + c2*v[i];
	double f1{ DBL_MAX / 4 };
	if (_nProjections < _nRandom) f1 = ProjectedDepth(z, w.get());

	c1 = cos(mu);
	c2 = sin(mu);
	for (int i = 0; i < d; i++) w[i] = c1*u[i] + c2*v[i];
	double f2{ DBL_MAX / 4 };
	if (_nProjections < _nRandom) f2 = ProjectedDepth(z, w.get());

	// stop, when the distance between two successive points is less than '_epsGS'
	// or when the maximum number of depth evaluations is reached
	while ((b - a > _epsGS) && (_nProjections < _nRandom)) {
		if (f1 > f2) {
			a = lambda;
			lambda = mu;
			mu = a + alpha*(b - a);
			c1 = cos(mu);
			c2 = sin(mu);
			for (int i = 0; i< d; i++) w[i] = c1*u[i] + c2*v[i];
			f2 = ProjectedDepth(z, w.get());
		}
		else {
			b = mu;
			mu = lambda;
			lambda = a + (1 - alpha)*(b - a);
			c1 = cos(lambda);
			c2 = sin(lambda);
			for (int i = 0; i< d; i++) w[i] = c1*u[i] + c2*v[i];
			f1 = ProjectedDepth(z, w.get());
		}
	}
	for (int i = 0; i < d; i++) u[i] = w[i];
	return (f1 + f2) / 2;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::CoordinateDescent': coordinate descent (CD) in R^d            */
/*   Coordinate descent combines line searches along the coordinate axes to    */
/*   find a minimum of the objective. Here we do coordinate descent in the     */
/*   space R^d, i.e., in each step we have to do d line searches. The          */
/*   iteration is stopped when the distance between two succesive points is    */
/*   less than a prescribed tolerance or when the maximum number of depth      */
/*   evaluations is reached.                                                   */
/*                                                                             */
/* This routine is called in 'cProjection::CoordinateDescentMultiStart'        */
/*                                                                             */
/*******************************************************************************/

double cProjection::CoordinateDescent(const double* z) {
	const double eps{ 1e-4 };
	cUniformSphere rndSphere(d);
	// the starting point is chosen according to the parameter '_startCD'
	// _startCD = 0 => starting point is the mean of the data points
	// _startCD = 1 => starting point is randomly chosen
	unique_ptr<double[]> u{ rndSphere(gen) }; 
	if (_startCD == 0) {
		unique_ptr<double[]> xquer{ mean(x, n, d) };
		if (distance(xquer.get(), z, d) >= 1e-8) {
			for (int i = 0; i < d; i++)  u[i] = z[i] - xquer[i];
			Normalize(u.get(), d);
		}
	}
	unique_ptr<double[]> uOld{ new double[d] {} };
	double depth{ DBL_MAX };
	do {
		for (int i = 0; i < d; i++) uOld[i] = u[i];
		// perform d line searches along the d coordinate axes 
		for (int j = 1; j < d; j++) {
			if (_nProjections < _nRandom) {
				unique_ptr<double[]> v{ new double[d] {} };
				v[j] = 1;
				// perform the line search along axis j
				depth = min(MethodCD[(int)_lineSearchCD](z, u.get(), v.get()), depth);
			}
		}
	} while ((distance(u.get(), uOld.get(), d) >= eps) && (_nProjections < _nRandom));
	// stop, when the distance between two successive points is less than 'eps'
	// or when the maximum number of depth evaluations is reached
	if (debug >= 2) cout << "CD:   " << _nProjections << endl;
	return depth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::CoordinateDescentMultiStart'                                  */
/*   In applying 'CD' in our simulations we use a multistart strategy.         */
/*   in order to come to a fair comparison of the methods we want to           */
/*   guarantee that for every method we have the same number '_nRandom'        */
/*   of univariate depth evaluations. Therefore, if the actual number of       */
/*   iterations in the first run of 'CD' is less than '_nRandom', we start     */
/*   the algorithm again. We do this as long as the maximum number of          */
/*   univariate depth evaluations is not reached.                              */
/*                                                                             */
/*******************************************************************************/

double cProjection::CoordinateDescentMultiStart(const double* z) {
	double minDepth{ DBL_MAX };
	_nProjections = 0;
	_startCD = 0;
	int DirectionsSize = 0;
	do {
		minDepth = min(CoordinateDescent(z), minDepth);
        if(_option == 4){
            _DirectionsCard.push_back(_Directions.size() - DirectionsSize);
            DirectionsSize = _Directions.size();
        }
		_startCD = 1;
	} while (_nProjections < _nRandom);
	if (debug >= 2) cout << "CD:   " << _nProjections << endl;
	return minDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::LineSearchUnif'                                               */
/*   This routine performs a line search over a uniform grid. The line         */
/*   along which the search proceeds goes through the point 'u' on the         */
/*   unit sphere and proceeds along the direction 'dir'. Since 'dir' is        */
/*   assumed to be a direction in the tangent hyperplane at 'u', contrary to   */
/*   'LineSearchUnif' no correction of 'dir' has to be made. Therefore, we     */
/*   search along a great semi-circle defined by 'u' and the direction 'dir'.  */
/*   On this great semi-circle we put a unform grid and search for the         */
/*   minimum.                                                                  */
/*                                                                             */
/* Args:																	   */
/*   z - point for which the depth has to be computed					 	   */
/*   u - point on the unit sphere from wehich the line search starts           */
/*   dir - direction of the line search (in the tangent hyperplane at 'u')     */
/*                                                                             */
/* This routine is called in 'cProjection::CoordinateDescentGC'                */
/*                                                                             */
/* This routine is essentially 'Algorithm 8' in the article 'Approximate       */
/* computaion of projection depths'                                            */
/*                                                                             */
/*******************************************************************************/

double cProjection::LineSearchUnifGC(const double* z, double* u, double* dir) {
	unique_ptr<double[]> uOpt{ new double[d] };
	// construct a uniform grid on the great semi-circle definde by u and the direction dir
	// the grid consists of _nLineSearch + 1 points
	double MinDepth{ DBL_MAX };
	for (int i = 0; i <= _nLineSearch; i++) {
		if (_nProjections < _nRandom) {
			double lambda{ -M_PI / 2 + i * M_PI / _nLineSearch };
			unique_ptr<double[]> w{ new double[d] };
			// construct the i-th grid point on the semi-circle
			for (int k = 0; k < d; k++) w[k] = cos(lambda) * u[k] + sin(lambda) * dir[k];
			// compute the univariate depth of point 'z' in direction 'w'
			double f = ProjectedDepth(z, w.get());
			if (f < MinDepth) {
				MinDepth = f;
				uOpt = move(w);
			}
		}
	}
	for (int i = 0; i < d; i++) u[i] = uOpt[i];
	return MinDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::LineSearchGSGC'                                               */
/*   This routine performs a line search using the golden-section search.      */
/*   The lien along which the search proceeds goes through the point 'u' on    */
/*   the unit sphere and proceeds along the direction 'dir'. Since 'dir' is    */
/*   assumed to be a direction in the tangent hyperplane at 'u', contrary to   */
/*   'LineSearchGS' no correction of 'dir' has to be made. Therefore, we       */
/*   search along a great semi-circle defined by 'u' and the direction 'dir'.  */
/*   On this great semi-circle  a golden-section search for the minimum is     */
/*   performed.                                                                */
/*                                                                             */
/* Args:																	   */
/*   z - point for which the depth has to be computed					 	   */
/*   u - point on the unit sphere from wehich the line search starts           */
/*   dir - direction of the line search (in the tangent hyperplane at 'u')     */
/*                                                                             */
/* This routine is called in 'cProjection::CoordinateDescentGC'                */
/*                                                                             */
/* This routine is essentially 'Algorithm 9' in the article 'Approximate       */
/* computaion of projection depths'                                            */
/*                                                                             */
/*******************************************************************************/

double cProjection::LineSearchGSGC(const double* z, double* u, double* dir) {
	const double alpha{ (sqrt(5) - 1) / 2 };
	unique_ptr<double[]> v{ new double[d] };
	// perform a golden-section search along a great semi-circle defined by u 
    // and the direction dir
	double a{ -M_PI / 2 }, b{ M_PI / 2 };
	double lambda{ a + (1 - alpha) * (b - a) };
	double mu{ a + alpha * (b - a) };
	double minf{ DBL_MAX };

	double c1 = cos(lambda);
	double c2 = sin(lambda);
	unique_ptr<double[]> w{ new double[d] };
	for (int i = 0; i < d; i++) w[i] = c1 * u[i] + c2 * dir[i];
	double f1{ DBL_MAX };
	if (_nProjections < _nRandom) f1 = ProjectedDepth(z, w.get());
	minf = min(f1, minf);

	c1 = cos(mu);
	c2 = sin(mu);
	for (int i = 0; i < d; i++) w[i] = c1 * u[i] + c2 * dir[i];
	double f2{ DBL_MAX };
	if (_nProjections < _nRandom) f2 = ProjectedDepth(z, w.get());
	minf = min(f2, minf);

	// stop, when the distance between two successive points is less than '_epsGS'
	// or when the maximum number of depth evaluations is reached
	while ((b - a > _epsGS) && (_nProjections < _nRandom)) {
		if (f1 > f2) {
			a = lambda;
			lambda = mu;
			f1 = f2;
			mu = a + alpha * (b - a);
			c1 = cos(mu);
			c2 = sin(mu);
			for (int i = 0; i < d; i++) w[i] = c1 * u[i] + c2 * dir[i];
			f2 = ProjectedDepth(z, w.get());
			minf = min(f2, minf);
		}
		else {
			b = mu;
			mu = lambda;
			f2 = f1;
			lambda = a + (1 - alpha) * (b - a);
			c1 = cos(lambda);
			c2 = sin(lambda);
			for (int i = 0; i < d; i++) w[i] = c1 * u[i] + c2 * dir[i];
			f1 = ProjectedDepth(z, w.get());
			minf = min(f1, minf);
		}
	}
	for (int i = 0; i < d; i++) u[i] = w[i];
	return minf;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::CoordinateDescent': coordinate descent (CD) on S^{d-1}        */
/*   Coordinate descent combines line searches along the coordinate axes to    */
/*   find a minimum of the objective. Here we use a version of coordinate      */
/*   that is particularly adapted to the case that the domain of objective is  */
/*   the unit sphere S^{d-1}. Here we combine d-1 line searches which are      */
/*   defined by an orthonormal base of the tangent hyperplane at the unit      */
/*   sphere at 'u'. The iteration is stopped when the distance between two     */
/*   succesive points is less than a prescribed tolerance or when the maximum  */
/*   number of depth evaluations is reached.                                   */
/*                                                                             */
/* This routine is called in 'cProjection::CoordinateDescentGCMultiStart'      */
/*                                                                             */
/* This routine is essentially 'Algorithm 10' in the article 'Approximate      */
/* computaion of projection depths'                                            */
/*                                                                             */
/*******************************************************************************/

double cProjection::CoordinateDescentGC(const double* z) {
	const double eps{ 1e-4 };
	cUniformSphere rndSphere(d);
	// the starting point is chosen according to the parameter '_startCD'
	// _startCD = 0 => starting point is the mean of the data points
	// _startCD = 1 => starting point is randomly chosen
	unique_ptr<double[]> u{ rndSphere(gen) };
	if (_startCD == 0) {
		unique_ptr<double[]> xquer{ mean(x, n, d) };
		if (distance(xquer.get(), z, d) >= 1e-8) {
			for (int i = 0; i < d; i++)  u[i] = z[i] - xquer[i];
			Normalize(u.get(), d);
		}
	}
	unique_ptr<double[]> uOld{ new double[d] {} };
	double depth{ DBL_MAX };
	do {
		for (int i = 0; i < d; i++) uOld[i] = u[i];
		// perform d-1 line searches along the directions of an orthonormal
		// base of the tangent hyperplane at u
		for (int j = 0; j < d-1; j++) {
			if (_nProjections < _nRandom) {
				// compute the j-th direction v
				auto v = make_unique<double[]>(d);
				for (int k = 0; k < d - 1; k++) v[k] = -uOld[j] * uOld[k] / (1 - uOld[d - 1]);
				v[j] += 1;
				v[d - 1] = uOld[j];
				// and perform the line search along the great circle defined by u and v
				depth = min(MethodCDGC[(int)_lineSearchCDGC](z, u.get(), v.get()), depth);
			}
		}
	} while ((distance(u.get(), uOld.get(), d) >= eps) && (_nProjections < _nRandom));
	// stop, when the distance between two successive points is less than 'eps'
	// or when the maximum number of depth evaluations is reached
	if (debug >= 2) cout << "CDGC: " << _nProjections << endl;
	return depth;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::CoordinateDescentGCMultiStart'                                */
/*   In applying 'CDGC' in our simulations we use a multistart strategy.       */
/*   in order to come to a fair comparison of the methods we want to           */
/*   guarantee that for every method we have the same number '_nRandom'        */
/*   of univariate depth evaluations. Therefore, if the actual number of       */
/*   iterations in the first run of 'CDGC' is less than '_nRandom', we         */
/*   start the algorithm again. We do this as long as the maximum number       */
/*   of univariate depth evaluations is not reached.                           */
/*                                                                             */
/*******************************************************************************/

double cProjection::CoordinateDescentGCMultiStart(const double* z) {
	double minDepth{ DBL_MAX };
	_nProjections = 0;
	_startCD = 0;
	int DirectionsSize = 0;
	do {
		minDepth = min(CoordinateDescentGC(z), minDepth);
        if(_option == 4){
            _DirectionsCard.push_back(_Directions.size() - DirectionsSize);
            DirectionsSize = _Directions.size();
        }
		_startCD = 1;
	} while (_nProjections < _nRandom);
	if (debug >= 2) cout << "CDGC: " << _nProjections << endl;
	return minDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'centroid' computes the spherical mean of n points on the unit sphere       */
/*   Here, we simply compute the mean 'z' of the data points in 'x' and        */
/*   normalize 'z' to guarantee that it lies on the unit sphere.               */
/*                                                                             */
/* Args:																	   */
/*   x - array of points (together with their depth values)  			 	   */
/*   n - number of points in x                                                 */
/*   d - dimension of the points                                               */
/*   z - on exit, z contains the spherical mean                                */
/*                                                                             */
/* This routine is called in 'cProjection::NelderMead'                         */
/*                                                                             */
/*******************************************************************************/

void centroid(Feval* x, int n, int d, double z[]) {
	for (int j = 0; j < d; j++) z[j] = 0;
	// compute the mean of the n data points in x
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++) z[j] += x[i].arg[j];
	for (int j = 0; j < d; j++) z[j] /= n;
	// compute the norm of z
	double sum{};
	for (int j = 0; j < d; j++) sum += z[j] * z[j];
	sum = sqrt(sum);
	// normalize z
	for (int j = 0; j < d; j++) z[j] /= sum;
}


/*******************************************************************************/
/*                                                                             */
/* 'linComb' computes the point z / ||z|| where z = x + w * (y - x)            */
/*   This can be imagined as follows: Start at point x and make w steps in     */
/*   direction y - x, where the length of one step is the length of y - x.     */
/*   Finally, z is normalized so that it lies on the unit sphere.              */
/*                                                                             */
/* Args:																	   */
/*   x - starting point                                     			 	   */
/*   y - point that defines the direction and the step size                    */
/*   w - number of steps                                                       */
/*   z - on exit, z contains the point z = x + w * (y - x)                     */
/*                                                                             */
/* This routine is called in 'cProjection::NelderMead'                         */
/*                                                                             */
/*******************************************************************************/

void linComb(double* x, double* y, int d, double w, double* z) {
	// compute z = x + w * (y - x)
	for (int j = 0; j < d; j++) z[j] = x[j] + w * (y[j] - x[j]);
	// compute the norm of z
	double sum{};
	for (int j = 0; j < d; j++) sum += z[j] * z[j];
	sum = sqrt(sum);
	// and normalize z
	for (int j = 0; j < d; j++) z[j] /= sum;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::NelderMead': Nelder-Mead algorithm (NM) in R^d                */
/*   This routine performs the Nelder_mead algorithm in R^d. Therefore,        */
/*   the simplex used here is defined by d+1 points.                           */
/*                                                                             */
/* This routine is called in 'cProjection::NelderMeadMultiStart'               */
/*                                                                             */
/*******************************************************************************/

double cProjection::NelderMead(const double* z){
	// Set optimization parameters
	double alpha = 1;
	double gamma = 2;
	double beta = 0.5;
	double sigma = 0.5;
	double eps = 1e-4;
	// Allocate memory
	Feval* fevals = new Feval[d + 1]; // function evaluations
	for (int i = 0; i < d + 1; i++){
		fevals[i].arg = new double[d];
	}
	double* x_o = new double[d]; // centroid
	double* x_r = new double[d]; double f_r; // reflected point
	double* x_e = new double[d]; double f_e; // expanded point
	double* x_c = new double[d]; double f_c; // contracted point
	double* x_h = new double[d]; double f_h; // best of x_r and x_{N+1}
	// Generate and evaluate initial simplex
	cUniformSphere rndSphere(d);
	for (int i = 0; i < d + 1; i++){
		rndSphere.vector(fevals[i].arg, gen);
		fevals[i].val = ProjectedDepth(z, fevals[i].arg);
		if(_nProjections >= _nRandom) break;
	}
	// Main optimization cycle
	int iter = 0;
	bool converged = false;
	while (!converged && (_nProjections < _nRandom)) {
		iter++;
		// Order the values
		sort(fevals, fevals + d+1, Compare);
		// Check convergence criterium
		double maxDist = 0;
		for (int i = 0; i < d; i++) {
			double tmpVal = fabs(fevals[0].arg[i] - fevals[d].arg[i]);
			if (tmpVal > maxDist) {
				maxDist = tmpVal;
			}
		}
		if (maxDist < eps) {
			converged = true;
			break;
		}
		// Calculate the centroid of the d best points
		centroid(fevals, d, d, x_o);
		// Calculate and evaluate reflected point
		linComb(x_o, fevals[d].arg, d, -alpha, x_r);
		f_r = ProjectedDepth(z, x_r);
		if(_nProjections >= _nRandom) break;
		
		// Choose what to do
		if ((fevals[0].val <= f_r) && (f_r < fevals[d - 1].val)) {
			// Reflection
			memcpy(fevals[d].arg, x_r, d * sizeof(double));
			fevals[d].val = f_r;
		}
		else {
			if (f_r < fevals[0].val) {
				// Calculate and evaluate expanded point
				linComb(x_o, x_r, d, gamma, x_e);
				f_e = ProjectedDepth(z, x_e);
				if(_nProjections >= _nRandom) break;
				
				if (f_e < f_r) {
					// Expansion
					memcpy(fevals[d].arg, x_e, d * sizeof(double));
					fevals[d].val = f_e;
				}
				else {
					// Still (just) reflection
					memcpy(fevals[d].arg, x_r, d * sizeof(double));
					fevals[d].val = f_r;
				}
			}
			else {
				if (f_r < fevals[d].val) {
					memcpy(x_h, x_r, d * sizeof(double));
					f_h = f_r;
				}
				else {
					memcpy(x_h, fevals[d].arg, d * sizeof(double));
					f_h = fevals[d].val;
				}
				// Calculate and evaluate contracted point
				linComb(x_o, x_h, d, beta, x_c);
				f_c = ProjectedDepth(z, x_c);
				if(_nProjections >= _nRandom) break;
				
				if (f_c < fevals[d].val) {
					// Contraction
					memcpy(fevals[d].arg, x_c, d * sizeof(double));
					fevals[d].val = f_c;
				}
				else {
					// Reduction
					for (int i = 1; i < d + 1; i++) {
						linComb(fevals[0].arg, fevals[i].arg, d, sigma, fevals[i].arg);
						fevals[i].val = ProjectedDepth(z, fevals[i].arg);
						if(_nProjections >= _nRandom) break;
					}
				}
			}
		}
	}
	// Extract result
	sort(fevals, fevals + d+1, Compare);
	double res = fevals[0].val;
	// Release memory
	delete[] fevals;
	delete[] x_o;
	delete[] x_r;
	delete[] x_e;
	delete[] x_c;
	delete[] x_h; 
	if (debug >= 2) cout << "NM:   " << _nProjections << endl;
	return res;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::NelderMeadMultiStart'                                         */
/*   In applying 'NM' in our simulations we use a multistart strategy.         */
/*   in order to come to a fair comparison of the methods we want to           */
/*   guarantee that for every method we have the same number '_nRandom'        */
/*   of univariate depth evaluations. Therefore, if the actual number of       */
/*   iterations in the first run of 'NM' is less than '_nRandom', we start     */
/*   the algorithm again. We do this as long as the maximum number of          */
/*   univariate depth evaluations is not reached.                              */
/*                                                                             */
/*******************************************************************************/

double cProjection::NelderMeadMultiStart(const double* z) {
	double minDepth{ DBL_MAX };
	_nProjections = 0;
	int DirectionsSize = 0;
	do {
		minDepth = min(NelderMead(z), minDepth);
        if(_option == 4){
            _DirectionsCard.push_back(_Directions.size() - DirectionsSize);
            DirectionsSize = _Directions.size();
        }
	} while (_nProjections < _nRandom);
	if (debug >= 2) cout << "NM:   " << _nProjections << endl;
	return minDepth;
}

/*******************************************************************************/
/*                                                                             */
/* 'sphericalMean' computes the spherical mean of n points on the unit sphere  */
/*   Here, we simply compute the mean of the data points in 'x' and            */
/*   normalize it to guarantee that it lies on the unit sphere.                */
/*                                                                             */
/* Args:																	   */
/*   x - array of points (together with their depth values)  			 	   */
/*   n - number of points in x                                                 */
/*   d - dimension of the points                                               */
/* Returns:                                                                    */
/*   spherical mean of the data points                                         */
/*                                                                             */
/* This routine is called in 'cProjection::NelderMeadGC'                       */
/*                                                                             */
/*******************************************************************************/

unique_ptr<double[]> sphericalMean(fVal* x, int n, int d) {
	unique_ptr<double[]> res{ new double[d] {} };
	double sum{ 0 };
	for (int j = 0; j < d; j++) {
		for (int i = 0; i < n; i++)	res[j] += x[i].arg[j];
		res[j] /= n;
		sum += res[j] * res[j];
	}
	for (int j = 0; j < d; j++) res[j] /= sqrt(sum);
	return res;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::greatCircle':                                                 */
/*   This routine computes the point z on the great circle through x and y     */
/*   whose distance from x is w times the distance between x and y             */
/*                                                                             */
/* Args:																	   */
/*   x - starting point on the unit sphere                          	 	   */
/*   y - point on the unit sphere that defines the direction and step size     */
/*   w - number of steps                                                       */
/* Returns:                                                                    */
/*   the point z on the great circle through x and y whose distance from       */
/*   x is w times the distance between x and y                                 */
/*                                                                             */
/* This routine is called in 'cProjection::NelderMeadGC'                       */
/*                                                                             */
/* This routine is essentially 'Algorithm 11' in the article 'Approximate      */
/* computaion of projection depths'                                            */
/*                                                                             */
/*******************************************************************************/

unique_ptr<double[]> cProjection::greatCircle(const double x[], const double y[], int d, double w) {
	double xy{}, sum{}, alpha{}, sina{};
	for (int i = 0; i < d; i++) xy += x[i] * y[i];

	if (xy >= 0) {
		for (int i = 0; i < d; i++) sum += (x[i] - y[i]) * (x[i] - y[i]);
		alpha = 2 * asin(0.5 * sqrt(sum));
		sina = sqrt(sum * (1 + xy) / 2);
	}
	else {
		for (int i = 0; i < d; i++) sum += (x[i] + y[i]) * (x[i] + y[i]);
		alpha = M_PI - 2 * asin(0.5 * sqrt(sum));
		sina = sqrt(sum * (1 - xy) / 2);
	}
    double gammax = alpha * (1 - w);
    double gammay = alpha * w;
	if (_boundNM == 1) {
		if (fabs(gammay) >= M_PI / 2) {
			gammay = signbit(gammay) ? -M_PI / 2 : M_PI / 2;
			gammax = alpha - gammay;
		}
	}
    double cx = sin(gammax) / sina;
    double cy = sin(gammay) / sina;
	unique_ptr<double[]> res{ new double[d] };
	for (int i = 0; i < d; i++) res[i] = cx * x[i] + cy * y[i];
	return res;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::NelderMead': Nelder-Mead algorithm (NM) on S^{d-1}            */
/*   Here we use a version of the Nelder-mead algorithm that is particularly   */
/*   adapted to the case that the domain of the objective is the unit sphere   */
/*   S^{d-1}. Since S^{d-1} is a (d-1)-dimensional manifold the simplex        */
/*   used here is defined by d points on the unit sphere.                      */
/*                                                                             */
/* This routine is called in 'cProjection::NelderMeadGCMultiStart'             */
/*                                                                             */
/* This routine is essentially 'Algorithm 12' in the article 'Approximate      */
/* computaion of projection depths'                                            */
/*                                                                             */
/*******************************************************************************/

double cProjection::NelderMeadGC(const double z[]) {
	// Set optimization parameters
	const double alpha{ 1 }, gamma{ 2 }, rho{ 0.5 }, sigma{ 0.5 }, eps{ 1e-4 };
	// Allocate memory
	fVal* fevals = new fVal[d]; // function evaluations
	fVal x_o(d), x_r(d), x_e(d), x_c(d), *x_h;
	int iter{ 0 }; // number of iterations
	// Generate and evaluate initial simplex
	cUniformSphere rndSphere(d);
	// the starting simplex is drawn randomly from a spherical cap centered at
	// a point 'u'. 'u' is chosen according to the parameter '_startNM'
	// _startNM = 0 => 'u' is the mean of the data points
	// _startNM = 1 => 'u' is randomly chosen
	unique_ptr<double[]> u{ rndSphere(gen) };
	if (_startNM == 0) {
		unique_ptr<double[]> xquer{ mean(x, n, d) };
		if (distance(xquer.get(), z, d) >= 1e-8) {
			for (int i = 0; i < d; i++)  u[i] = z[i] - xquer[i];
			Normalize(u.get(), d);
		}
	}
	cRandomPolarCap rndPolarCap(d, (M_PI / 2) / _betaNM);
	// Generate and evaluate initial simplex
	for (int i = 0; i < d; i++) {
		fevals[i].arg = unique_ptr<double[]>{ rndPolarCap(gen, u.get()) };
		fevals[i].val = ProjectedDepth(z, fevals[i].p());
		if(_nProjections >= _nRandom) break;
	}
	
	// sort the points of the simplex according to their depth
	sort(fevals, fevals + d, cmp);
	if (debug >= 3) {
		for (int i = 0; i < d; i++) cout << fevals[i].val << "  ";
		cout << endl;
	}
	// Main optimization cycle
	double maxDist{};
	do{
		if(_nProjections >= _nRandom) break;
		iter++;
		if (debug >= 3) {
			for (int i = 0; i < d; i++) cout << fevals[i].val << "  ";
			cout << endl;
		}
		// Calculate the spherical mean of the d-1 best points
		x_o.arg = sphericalMean(fevals, d - 1, d);
		// Calculate and evaluate reflected point
		x_r.arg = greatCircle(x_o.p(), fevals[d-1].p(), d, -alpha);
		x_r.val = ProjectedDepth(z, x_r.p());
		if(_nProjections >= _nRandom) break;
		// Choose what to do
		if ((fevals[0].val <= x_r.val) && (x_r.val < fevals[d - 2].val)) {
			fevals[d - 1] = move(x_r); // Reflection
			if (debug >= 3) cout << "Reflection" << endl;
		}
		else {
			if (x_r.val < fevals[0].val) { // Calculate and evaluate expanded point
				x_e.arg = greatCircle(x_o.p(), x_r.p(), d, gamma);
				x_e.val = ProjectedDepth(z, x_e.p());
				if(_nProjections >= _nRandom) break;	
				if (x_e.val < x_r.val) fevals[d - 1] = move(x_e);
				else fevals[d - 1] = move(x_r);
				if (debug >= 3) cout << "Expansion" << endl;
			}
			else {
				if (x_r.val < fevals[d - 1].val) x_h = &x_r; else x_h = &(fevals[d - 1]);
				// Calculate and evaluate contracted point
				x_c.arg = greatCircle(x_o.p(), x_h->p(), d, rho);
				x_c.val = ProjectedDepth(z, x_c.p());
				if(_nProjections >= _nRandom) break;
				
				if (x_c.val < fevals[d - 1].val) {
					fevals[d - 1] = move(x_c);
					if (debug >= 3) cout << "Contraction" << endl;
				}
				else { // Reduction
					int MaxDim = 0;
					if(_nProjections < _nRandom - d){
						MaxDim = d;
					}
					else{
						MaxDim = _nRandom - _nProjections;
					}
					for (int i = 1; i < MaxDim; i++) {
						fevals[i].arg = greatCircle(fevals[0].p(), fevals[i].p(), d, sigma);
						fevals[i].val = ProjectedDepth(z, fevals[i].p());
						if(_nProjections >= _nRandom) break;
						
					}
					sort(fevals, fevals + d-1, cmp);
					if (debug >= 3) cout << "Shrink" << endl;
				}
			}
		}
		inplace_merge(fevals, fevals + d - 1, fevals + d, cmp);
		// Check convergence criterium
		maxDist = 0;
		for (int i = 0; i < d; i++) {
			double tmpVal = fabs(fevals[0].arg[i] - fevals[d - 1].arg[i]);
			if (tmpVal > maxDist) maxDist = tmpVal;
		}
	}while ((maxDist >= eps) && (_nProjections < _nRandom));
	// Extract result
	double res = fevals[0].val;
	// Release memory
	delete[] fevals;
	// return (maxDist < eps) ? res : 2;
	return res;
}

/*******************************************************************************/
/*                                                                             */
/* 'cProjection::NelderMeadGCMultiStart'                                       */
/*   In applying 'NMGC' in our simulations we use a multistart strategy.       */
/*   in order to come to a fair comparison of the methods we want to           */
/*   guarantee that for every method we have the same number '_nRandom'        */
/*   of univariate depth evaluations. Therefore, if the actual number of       */
/*   iterations in the first run of 'NMGC' is less than '_nRandom', we         */
/*   start the algorithm again. We do this as long as the maximum number       */
/*   of univariate depth evaluations is not reached.                           */
/*                                                                             */
/*******************************************************************************/

double cProjection::NelderMeadGCMultiStart(const double* z) {
	double minDepth{ DBL_MAX };
	_nProjections = 0;
	int DirectionsSize = 0;
	do {
		minDepth = min(NelderMeadGC(z), minDepth);
        if(_option == 4){
            _DirectionsCard.push_back(_Directions.size() - DirectionsSize);
            DirectionsSize = _Directions.size();
        }
	} while (_nProjections < _nRandom);
	if (debug >= 2) cout << "NMGC: " << _nProjections << endl;
	return minDepth;
}


 
