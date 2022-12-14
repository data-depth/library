/******************************************************************************/
/* File:             ProjectionDepths.h                                       */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Defines a class 'cProjection' that is able to compute several depths that  */
/* satisfy the projection property using different approximation algorithms.  */
/*                                                                            */
/* Further, some helper classes are defined.                                  */
/*                                                                            */
/******************************************************************************/

#include <random>
#include <functional>
#include <memory>
#include "Matrix.h"
#include "mvrandom.h"

// Enumeration class for the approximation algorithm
enum class eProjMeth { SimpleGrid, RefinedGrid, SimpleRandom, RefinedRandom, 
	CoordDesc, RandSimp, NelderMead, SimAnn, CoordDescGC, NelderMeadGC
};

// Enumeration class for the line search used in coordinate descent (CoordDesc)
enum class eLineSearchCD { LineSearchUnif, LineSearchGS };

// Enumeration class for the line search used in coordinate descent on great
// circles (CoordDescGC)
enum class eLineSearchCDGC { LineSearchUnifGC, LineSearchGSGC };

// Enumeration class for the depth notions
enum class eDepth { MD, HD, ZD, PD, APD };

const int nProjMeth = 10;  // number of approximation algorithms
const int nCDMeth   =  2;  // number of line serach strategies for CoordDesc
const int nCDGCMeth =  2;  // number of line serach strategies for CoordDescGC
const int nDepth    =  5;  // number of depth notions

// Structure for finding new records. 'it' is the number of the iteration where 
// a new record occurred and 'depth' the new record, i.e.m, new minimum depth 
struct cRecord {
	int it;
	double depth;
};

// Structure that contains a point (arg) and the corresponding depth value (val)
struct Feval {
	double* arg;
	double val = 1.1;  // init value superior to the maximum possible value
};

// Structure that contains a point (arg) and the corresponding depth value (val)
struct fVal {
	std::unique_ptr<double[]> arg = nullptr;
	double val = 1.1;                        // init value superior to the maximum possible value
	fVal() = default;                        // default constructor
	fVal(int d) : arg{ new double[d]{} } {}; // constructor when d is known in advance
	void setDimension(int d) {               // alloacte memory for array arg  
		arg = std::make_unique<double[]>(d);
	}; 
	double* p() { return arg.get(); }        // access array arg as double*
};

/****************************************************************************/
/*                                                                          */
/* 'cProjection' is the main class for computing several projection depths  */
/* using different approximation strategies.                                */
/*                                                                          */
/****************************************************************************/

class cProjection {
private:
	int _MultiStartConverged = 0;    // +1 everytime MultiStart algorithms converged
	// First we define some private members that control the different 
	// approximation methods
    int _option = 1;                // what is returned with depth_approximation
	int _nRandom = 1000;            // number of directions
	int _maxRefinesRand = 5;         // RRS: Number of re?nement steps
	int _maxRefinesGrid = 5;         // RGS: Number of re?nement steps
	double _alphaRand = 0.2;         // RRS: Shrinking factor of the spherical cap
	double _alphaGrid = 0.1;         // RGS: Shrinking factor of the spherical cap
	double _alphaRaSi = 1.25;        // RaSi: Parameter of the Dirichlet distribution
	double _alphaSA = 0.95;          // SA: Cooling factor
	double _betaSA = 10.0;           // SA: Size of the spherical cap  
	int _startSA = 0;                // SA: Starting value (0 = mean, 1 = random)
	int _startCD = 0;                // CD: Starting value (0 = mean, 1 = random)
	eLineSearchCD _lineSearchCD = eLineSearchCD::LineSearchUnif;
	                                 // CD: Line search: equally spaced or golden section 
	eLineSearchCDGC _lineSearchCDGC = eLineSearchCDGC::LineSearchUnifGC;
	                                 // CD: Line search: equally spaced or golden section 
	int _startNM = 0;                // NM: Starting value (0 = mean, 1 = random)
	double _betaNM = 1.0;            // NM: Size of the spherical cap
	int _boundNM = 0;                // NM: Bound movement on great circles (0 = no, 1 = yes)
	double _epsGS = 1e-3;            // epsilon used in the golden section search to terminate
	                                 // the search
	double _nLineSearch = 100;       // maximum number of depth evaluations in the line search
	int n, d;                        // n = number of points, d = dimension
	int _nProjections;               // counter for the number of projections ( = depth evaluations)
	eDepth _depth;                   // depth notion currently considered
	eProjMeth _Method;               // approximation method currently considered
	std::vector<double> _Depths;     // vector containing the univariate depth values computed so far
	std::vector<double> _MinDepths;  // vector containing the minimum depth values computed so far
	std::vector<std::vector<double>> _Directions;  // vector containing the projection vectors considered so far
	std::vector<double> _BestDirection;  // vector containing the best projection vector
	std::vector<int> _DirectionsCard;// vector containing the number of directions for every convergence
	double _lastDepthDuration;       // time used for computing the depth with a ( = time used 
	const dyMatrixClass::cMatrix& x; // matrix x of data points, x is a n x d matrix
	std::unique_ptr<double[]> xp;    // vector conatining the projections of the data points
	int nRecords;                    // number of records, i.e., entries in array 'records' 
	cRecord* records;                // array that contains all the records
	std::mt19937 gen;                // Mersenne twister is used as RNG
	std::uniform_real_distribution<double> rnd; // generation of U(0,1) random numbers
	// array of functions for the different approximation methods
	std::function<double(const double*)> Method[nProjMeth];  
	// array of functions for the two different line search strategies used in CoordDesc
	std::function<double(const double*, double*, double*)> MethodCD[nCDMeth];
	// array of functions for the two different line search strategies used in CoordDescGC
	std::function<double(const double*, double*, double*)> MethodCDGC[nCDGCMeth];
	void initStatistics();  // initializes the variables used to keep track of the progress
	// helper function for grid searches (GS and RGS
	double GridSearch(const double* z, double* u, const double size, int nStep);
	// line search strategies for CoordinateDescent
	double LineSearchUnif(const double* z, double* u, double* dir);
	double LineSearchGS(const double* z, double* u, double* dir);
	// line search strategies for CoordinateDescentGC
	double LineSearchUnifGC(const double* z, double* u, double* dir);
	double LineSearchGSGC(const double* z, double* u, double* dir);
	// helper function for CoordinateDescentMultiStart
	double CoordinateDescent(const double* z);
	// helper function for CoordinateDescentGCMultiStart
	double CoordinateDescentGC(const double* z);
	// helper function for SimulatedAnnealingMultiStart
	double SimulatedAnnealing(const double* z);
	// helper function for NelderMeadGCMultiStart
	double NelderMeadGC(const double* z);
	// helper function for NelderMeadMultiStart
	double NelderMead(const double* z);
	// compute the point on a great circle through x and y whose distance from x is w times
	// the distance between x and y
	std::unique_ptr<double[]> greatCircle(const double x[], const double y[], int d, double w);
	// function used to compute the univariate depth, this variable will be set to a function
	// that can be used to compute the univariate depth for the depth notion '_depth'
	std::function<double(const double, const double*, int)> UniDepth;
	// function used to compute the multivariate depth, this variable will be set to a function
	// that can be used to compute the univariate depth for the depth notion '_depth'
	std::function<double(const double*, const dyMatrixClass::cMatrix, int, int)> MultiDepth;
public:
	// constructor
	cProjection(const dyMatrixClass::cMatrix& x, int n = 0, int d = 0, int NRandom = 1000);
	// destructor
	~cProjection();
	// setter methods for most of the private members declared above
    void SetOption(int option) { _option = option; }
	void SetNRandom(int nRandom) { _nRandom = nRandom; }
	void SetMaxRefinesRand(int maxRefinesRand) { _maxRefinesRand = maxRefinesRand; }
	void SetMaxRefinesGrid(int maxRefinesGrid) { _maxRefinesGrid = maxRefinesGrid; }
	void SetAlphaRand(double alphaRand) { _alphaRand = alphaRand; }
	void SetAlphaGrid(double alphaGrid) { _alphaGrid = alphaGrid; }
	void SetAlphaRaSi(double alphaRaSi) { _alphaRaSi = alphaRaSi; }
	void SetAlphaSA(double alphaSA) { _alphaSA = alphaSA; }
	void SetBetaSA(double betaSA) { _betaSA = betaSA; }
	void SetStartSA(int startSA) { _startSA = startSA; }
	void SetLineSearchCD(eLineSearchCD lineSearchCD) { _lineSearchCD = lineSearchCD; }
	void SetLineSearchCDGC(eLineSearchCDGC lineSearchCDGC) { _lineSearchCDGC = lineSearchCDGC; }
	void SetStartNM(int startNM) { _startNM = startNM; }
	void SetBetaNM(double betaNM) { _betaNM = betaNM;  }
	void SetBoundNM(int boundNM) { _boundNM = boundNM; }
	void SetEpsGS(double epsGS) { _epsGS = epsGS; }
	void SetNLineSearch(int nLineSearch) { _nLineSearch = nLineSearch; }
	// set the depth notion that will be considered in the following 
	void SetDepthNotion(eDepth d);
	// set the approximation method that will be used in the following
	void SetMethod(eProjMeth m) { _Method = m;};
	// getter methods for some of the private members declared above
	int NProjections() { return _nProjections; };
	std::vector<double> Depths() { return std::vector<double>(_Depths.begin(), _Depths.end()); };
	std::vector<double> MinDepths() { return std::vector<double>(_MinDepths.begin(), _MinDepths.end()); };
	std::vector<std::vector<double>> Directions() { return std::vector<std::vector<double>>(_Directions.begin(), _Directions.end()); };
	std::vector<int> DirectionsCard(){return std::vector<int>(_DirectionsCard.begin(), _DirectionsCard.end());};
	double LastDepthDuration() { return _lastDepthDuration; };
	eProjMeth GetMethod() { return _Method; };
	// funtions to perform the different approximation methods
	double SimpleRandom(const double* z);                   // RS: simple random search  
	double SimpleGrid(const double* z);                     // GS: simple grid search
	double RefinedRandom(const double* z);                  // RRS: refined random search  
	double RefinedGrid(const double* z);                    // RGS: refined grid search 
	double RandomSimplices(const double* z);                // RaSi: random simplices 
	double SimulatedAnnealingMultiStart(const double* z);   // SA: simulated annealing with multistart strategy
	double CoordinateDescentMultiStart(const double* z);    // CD: coordinate descent with multistart strategy
	double CoordinateDescentGCMultiStart(const double* z);  // CDGC: coordinate descent along great circles with multistart strategy
	double NelderMeadMultiStart(const double* z);           // NM: Nelder-Mead with multistart strategy
	double NelderMeadGCMultiStart(const double* z);         // NMGC: Nelder-Mead along great cdircles with multistart strategy  
	// function to compute the multivariate depth using the currently selected approximation method
	double Depth(const double* z);
	// function to compute the multivariate depth using an exact algorithm
	double ExactDepth(const double* z);
	// function to project 'z' in direction 'u' and compute its univariate depth
	double ProjectedDepth(const double* z, const double* u);
	// return single best direction
	std::vector<double> BestDirection() { return std::vector<double>(_BestDirection.begin(), _BestDirection.end()); };
};

