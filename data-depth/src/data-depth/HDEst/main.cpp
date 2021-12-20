/******************************************************************************/
/* File:             main.cpp                                                 */
/* Created by:       Rainer Dyckerhoff, Pavlo Mozharovskyi                    */
/* Last revised:     04.07.2020                                               */
/*                                                                            */
/* Main file that contains all the routines necessray to simulate the data,   */
/* fine tune the parameters and run the simulations.                          */  
/*                                                                            */
/******************************************************************************/

#define _USE_MATH_DEFINES
#define NOMINMAX
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <functional>
#include <numeric>
#include <cmath>
#include <cstring>
#include <float.h>
#include "mvrandom.h"
#include "Matrix.h"
#include "ProjectionDepths.h"


using namespace std;
using namespace dyMatrixClass;

// Definition of constant string arrays for names of depths, distributions and approximation methods
const string Algorithm[]{ "GS", "RGS", "RS", "RRS", "CC", "RaSi", "NM", "SA", "CCGC", "NMGC", "Exact" };
const string DistName[] = { "Normal", "t-Dist", "Cauchy", "Uniform", "SkewNormal", "Exponential", "Shell", "Bimodal", "Multimodal" };
const string LongDistName[] = { "normal distribution", "Cauchy distribution", "uniform distribution",
	"t-distribution", "skew normal distribution", "exponential distribution",
	"hemispherical shell distribution", "bimodal normal distribution", "multimodal normal distribution" };
const string DepthName[] = { "MD", "HD", "ZD", "PD", "APD" };
const string LongDepthName[] = { "Mahalanobis depth", "Halfspace depth", "Zonoid depth", "Projection depth", "Asymmetric projection depth" };
// Definition of additional constants.
const int Dimensions[] = { 5, 10, 15, 20 };
const int Projections[] = { 100, 1000, 10000 };
const int maxDatasets{ 1000 };



// Structure used in the fine-tuning of parameters
struct depthRec {
	int method;
	int it;
	double val;
};

/*******************************************************************************/
/*                                                                             */
/* 'analyzeDoubleList' is used in parsing the command line.                    */
/*   The first argument is a constant string of the form [num_1,...,num_k],    */
/*   where num_1,...,num_k is a list of doubles.                               */
/*   The routine returns a pointer to an array of doubles that contains the    */
/*   values num_1,...,num_k.                                                   */
/*   On exit, the number k of entries in the list is stored in 'num'.          */
/*                                                                             */
/*******************************************************************************/

double* analyzeDoubleList(const char* s, int* num) {
	// Treat case of an emply list
	string params = s;
	if (params.compare("[]") == 0) {
		num[0] = 0;
		return 0;
	}
	// Prepare the input string
	char* theValues = new char[strlen(s) - 1];
	memcpy(theValues, s + 1, strlen(s) - 2);
	theValues[strlen(s) - 2] = 0;
	std::stringstream ss(theValues);
	// First count the number of entries
	int counter = 0;
	string curTok;
	while (getline(ss, curTok, ',')) counter++;
	num[0] = counter;
	// Now fill the allocated array
	double* vals = new double[counter];
	ss.clear();
	ss.str(theValues);
	counter = 0;
	while (getline(ss, curTok, ',')) {
		vals[counter++] = stod(curTok.data());
	}
	// The return
	return vals;
}

/*******************************************************************************/
/*                                                                             */
/* 'analyzeIntList' is used in parsing the command line.                       */
/*   The first argument is a constant string of the form [num_1,...,num_k],    */
/*   where num_1,...,num_k is a list of integers.                              */
/*   The routine returns a pointer to an array of ints that contains the       */
/*   values num_1,...,num_k.                                                   */
/*   On exit, the number k of entries in the list is stored in 'num'.          */
/*                                                                             */
/*******************************************************************************/

int* analyzeIntList(const char* s, int* num) {
	// We simply call analyzeDoubleList to analyze the list
	double* tmp = analyzeDoubleList(s, num);
	// The double values returned are now converted to integers
	int* vals = new int[*num];
	for (int i = 0; i < *num; i++) vals[i] = tmp[i];
	delete[] tmp;
	return vals;
}

/*******************************************************************************/
/*                                                                             */
/* 'analyzeStringList' is used in parsing the command line.                    */
/*   The first argument is a constant string of the form [str_1,...,str_k],    */
/*   where str_1,...,str_k is a list of comma separated strings.               */
/*   The routine returns a pointer to an array of strings that contains the    */
/*   values str_1,...,str_k.                                                   */
/*   On exit, the number k of entries in the list is stored in 'num'.          */
/*                                                                             */
/*******************************************************************************/

string* analyzeStringList(const char* s, int* num) {
	// Treat case of an emply list
	string params = s;
	if (params == "[]") {
		num[0] = 0;
		return nullptr;
	}
	// Prepare the input string
	char* theValues = new char[strlen(s) - 1];
	memcpy(theValues, s + 1, strlen(s) - 2);
	theValues[strlen(s) - 2] = 0;
	std::stringstream ss(theValues);
	// First count the number of entries
	int counter = 0;
	string curTok;
	while (getline(ss, curTok, ',')) counter++;
	num[0] = counter;
	// Now fill the allocated array
	string* vals = new string[counter];
	ss.clear();
	ss.str(theValues);
	counter = 0;
	while (getline(ss, curTok, ',')) {
		vals[counter++] = curTok.data();
	}
	// The return
	return vals;
}

/*******************************************************************************/
/*                                                                             */
/* 'SetDepthPars' is used to set the parameters for the slected approximation  */
/*   method. An object of class 'cProjection' is passed as the first           */
/*   parameter. The second pareameter is a string containing the list of       */
/*   selected parameters for the selected approximation method. The list is    */
/*   of the form [par_1,...,par_k] where par_1,.._par_k is a list of doubles.  */
/*                                                                             */
/*******************************************************************************/

int SetDepthPars(cProjection& depthObj, string param) {
	int nPars = -1;
	// Parse the list of parameters
	double* allPars = analyzeDoubleList(param.data(), &nPars);
	// Set the paremters depending on which approximation method is selected
	switch (depthObj.GetMethod()) {
	case eProjMeth::RefinedRandom:
		depthObj.SetMaxRefinesRand((int)allPars[0]);
		depthObj.SetAlphaRand(allPars[1]);
		break;
	case eProjMeth::RefinedGrid:
		depthObj.SetMaxRefinesGrid((int)allPars[0]);
		depthObj.SetAlphaGrid(allPars[1]);
		break;
	case eProjMeth::SimAnn:
		depthObj.SetAlphaSA(allPars[0]);
		depthObj.SetBetaSA(allPars[1]);
		depthObj.SetStartSA((int)allPars[2]);
		break;
	case eProjMeth::CoordDescGC:
		depthObj.SetLineSearchCDGC((eLineSearchCDGC)(int)allPars[0]);
		break;
	case eProjMeth::NelderMeadGC:
		depthObj.SetBetaNM(allPars[0]);
		depthObj.SetBoundNM((int)allPars[1]);
		depthObj.SetStartNM((int)allPars[2]);
		break;
	case eProjMeth::RandSimp:
		depthObj.SetAlphaRaSi(allPars[0]);
		break;
	default:
		break;
	}
	return 0;
}

/*******************************************************************************/
/*                                                                             */
/* 'TuneRefinedRandom' is used to select the 'best' parameters for             */
/*   the refined random search (RRS).                                          */
/*                                                                             */
/*   For the selected sample size (n), dimension (dim), distribution           */
/*   (nDistribution), depth (nDepth) and number of projections (nProjections)  */
/*   'nSim' samples are simulated. For each of these samples the selected      */
/*   approximation method (here: RRS) is run for each combination of the       */
/*   parameters passed in 'lList1' and 'lList2'.                               */
/*                                                                             */
/*   The output shows the values of the statistics 'PercBest' and 'AveRank'    */
/*   for each of the parameter combinations.                                   */
/*                                                                             */
/*   This routine is called in 'TuneParameters'.                               */
/*                                                                             */
/*******************************************************************************/

void TuneRefinedRandom(int n, int dim, int nDistribution, int nSim, int nDepth, int nProjections, string lList1, string lList2) {
	clock_t totaltime = 0;
	clock_t starttime;
	double laptime = 0;
	double averagetime;

	double MinDepth;

	int nRefines, nAlpha;
	int* refines = analyzeIntList(lList1.data(), &nRefines);
	double* alphas = analyzeDoubleList(lList2.data(), &nAlpha);

	depthRec* depth = new depthRec[nRefines * nAlpha]{};
	int* iterations = new int[nRefines * nAlpha]{};
	double* isBest = new double[nRefines * nAlpha]{};
	double* rank = new double[nRefines * nAlpha]{};

	cout << setw(13) << " ";
	for (int i = 0; i < nRefines; i++)
		for (int j = 0; j < nAlpha; j++) {
			cout << "(" << refines[i] << "," << alphas[j] << ")   ";
		}
	cout << endl;

	cMatrix x;
	double *z;

	x.SetSize(n, dim);
	z = new double[dim];

	cProjection Depth(x, n, dim, nProjections);
	Depth.SetDepthNotion((eDepth)nDepth);

	mt19937 gen(1234);
	cMNormal normal(dim);
	cMCauchy cauchy(dim);
	cMUniformCube unif(dim);
	cMt tDist(dim, 5);
	cMSkewNormal skewNormal(dim, new double[dim] {5});
	cMExponential exponential(dim);
	uniform_int_distribution<int> rndint(0, n - 1);

	function<void(double* const)> rndVector[] = {
		[&](double* const x) { normal.vector(x, gen);  },
		[&](double* const x) { cauchy.vector(x, gen);  },
		[&](double* const x) { unif.vector(x, gen);  },
		[&](double* const x) { tDist.vector(x, gen);  },
		[&](double* const x) { skewNormal.vector(x, gen);  },
		[&](double* const x) { exponential.vector(x, gen);  }
	};
	function<void(double* const)> rndVec = rndVector[nDistribution];

	for (int cnt = 0; cnt < nSim; cnt++) {
		//cout << cnt << endl;
		for (int i = 0; i < n; i++) rndVec(x[i]);
		for (int j = 0; j < dim; j++) z[j] = 0;
		for (int i = 0; i < 10; i++) {
			int k = rndint(gen);
			for (int j = 0; j < dim; j++) z[j] += x[k][j];
		}
		for (int j = 0; j < dim; j++) z[j] = z[j] / 10.0;

		starttime = clock();
		Depth.SetMethod(eProjMeth::RefinedRandom);
		for (int i = 0; i < nRefines; i++) {
			Depth.SetMaxRefinesRand(refines[i]);
			for (int j = 0; j < nAlpha; j++) {
				Depth.SetAlphaRand(alphas[j]);
				depth[i*nAlpha + j].val = Depth.Depth(z);
				depth[i*nAlpha + j].it = Depth.NProjections();
				depth[i*nAlpha + j].method = i * nAlpha + j;
			}
		}
		MinDepth = DBL_MAX;
		for (int i = 0; i < nRefines * nAlpha; i++) MinDepth = min(MinDepth, depth[i].val);
		for (int i = 0; i < nRefines * nAlpha; i++)
			if (abs(depth[i].val - MinDepth) < 1e-8) isBest[i]++;

		sort(depth, depth + nRefines * nAlpha, [](depthRec& a, depthRec& b) -> bool { return (a.val < b.val); });

		int i{ 0 }, j{ 0 };
		while (i < nRefines * nAlpha) {
			i++;
			while ((i < nRefines * nAlpha) && (depth[i].val < depth[j].val + 1e-8)) i++;
			double mr = (i + j - 1) / 2.0;
			for (; j < i; j++) rank[depth[j].method] += mr;
		}

		laptime = clock() - starttime;
		totaltime += laptime;
	}
	for (int i = 0; i < nRefines * nAlpha; i++) rank[i] /= nSim;
	for (int i = 0; i < nRefines * nAlpha; i++) isBest[i] /= nSim;

	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nRefines * nAlpha; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << rank[i];
	cout << endl;
	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nRefines * nAlpha; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << isBest[i] * 100;
	cout << endl;

	averagetime = (double)totaltime / nSim;
	cout << "% Average time (in s):     " << averagetime / CLOCKS_PER_SEC << endl;
	delete[] z;

}

/*******************************************************************************/
/*                                                                             */
/* 'TuneRefinedGrid' is used to select the 'best' parameters for               */
/*   the refined grid search (RGS).                                            */
/*                                                                             */
/*   For the selected sample size (n), dimension (dim), distribution           */
/*   (nDistribution), depth (nDepth) and number of projections (nProjections)  */
/*   'nSim' samples are simulated. For each of these samples the selected      */
/*   approximation method (here: RGS) is run for each combination of the       */
/*   parameters passed in 'lList1' and 'lList2'.                               */
/*                                                                             */
/*   The output shows the values of the statistics 'PercBest' and 'AveRank'    */
/*   for each of the parameter combinations.                                   */
/*                                                                             */
/*   This routine is called in 'TuneParameters'.                               */
/*                                                                             */
/*******************************************************************************/

void TuneRefinedGrid(int n, int dim, int nDistribution, int nSim, int nDepth, int nProjections, string lList1, string lList2) {
	clock_t totaltime = 0;
	clock_t starttime;
	double laptime = 0;
	double averagetime;

	double MinDepth;

	int nRefines, nAlpha;
	int* refines = analyzeIntList(lList1.data(), &nRefines);
	double* alphas = analyzeDoubleList(lList2.data(), &nAlpha);

	depthRec* depth = new depthRec[nRefines * nAlpha]{};
	int* iterations = new int[nRefines * nAlpha]{};
	double* isBest = new double[nRefines * nAlpha]{};
	double* rank = new double[nRefines * nAlpha]{};

	cout << setw(13) << " ";
	for (int i = 0; i < nRefines; i++)
		for (int j = 0; j < nAlpha; j++) {
			cout << "(" << refines[i] << "," << alphas[j] << ")   ";
		}
	cout << endl;

	cMatrix x;
	double *z;

	x.SetSize(n, dim);
	z = new double[dim];

	cProjection Depth(x, n, dim, nProjections);
	Depth.SetDepthNotion((eDepth)nDepth);

	mt19937 gen(1234);
	cMNormal normal(dim);
	cMCauchy cauchy(dim);
	cMUniformCube unif(dim);
	cMt tDist(dim, 5);
	cMSkewNormal skewNormal(dim, new double[dim] {5});
	cMExponential exponential(dim);
	uniform_int_distribution<int> rndint(0, n - 1);

	function<void(double* const)> rndVector[] = {
		[&](double* const x) { normal.vector(x, gen);  },
		[&](double* const x) { cauchy.vector(x, gen);  },
		[&](double* const x) { unif.vector(x, gen);  },
		[&](double* const x) { tDist.vector(x, gen);  },
		[&](double* const x) { skewNormal.vector(x, gen);  },
		[&](double* const x) { exponential.vector(x, gen);  }
	};
	function<void(double* const)> rndVec = rndVector[nDistribution];

	for (int cnt = 0; cnt < nSim; cnt++) {
		//cout << cnt << endl;
		for (int i = 0; i < n; i++) rndVec(x[i]);
		for (int j = 0; j < dim; j++) z[j] = 0;
		for (int i = 0; i < 10; i++) {
			int k = rndint(gen);
			for (int j = 0; j < dim; j++) z[j] += x[k][j];
		}
		for (int j = 0; j < dim; j++) z[j] = z[j] / 10.0;

		starttime = clock();


		Depth.SetMethod(eProjMeth::RefinedGrid);
		for (int i = 0; i < nRefines; i++) {
			Depth.SetMaxRefinesGrid(refines[i]);
			for (int j = 0; j < nAlpha; j++) {
				Depth.SetAlphaGrid(alphas[j]);
				depth[i*nAlpha + j].val = Depth.Depth(z);
				depth[i*nAlpha + j].it = Depth.NProjections();
				depth[i*nAlpha + j].method = i * nAlpha + j;
			}
		}
		MinDepth = DBL_MAX;
		for (int i = 0; i < nRefines * nAlpha; i++) MinDepth = min(MinDepth, depth[i].val);
		for (int i = 0; i < nRefines * nAlpha; i++)
			if (abs(depth[i].val - MinDepth) < 1e-8) isBest[i]++;

		sort(depth, depth + nRefines * nAlpha, [](depthRec& a, depthRec& b) -> bool { return (a.val < b.val); });

		int i{ 0 }, j{ 0 };
		while (i < nRefines * nAlpha) {
			i++;
			while ((i < nRefines * nAlpha) && (depth[i].val < depth[j].val + 1e-8)) i++;
			double mr = (i + j - 1) / 2.0;
			for (; j < i; j++) rank[depth[j].method] += mr;
		}

		laptime = clock() - starttime;
		totaltime += laptime;
	}
	for (int i = 0; i < nRefines * nAlpha; i++) rank[i] /= nSim;
	for (int i = 0; i < nRefines * nAlpha; i++) isBest[i] /= nSim;

	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nRefines * nAlpha; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << rank[i];
	cout << endl;
	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nRefines * nAlpha; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << isBest[i] * 100;
	cout << endl;

	averagetime = (double)totaltime / nSim;
	cout << "% Average time (in s):     " << averagetime / CLOCKS_PER_SEC << endl;
	delete[] z;
}

/*******************************************************************************/
/*                                                                             */
/* 'TuneRandSimp' is used to select the 'best' parameters for                  */
/*   the random simplices method (RaSi).                                       */
/*                                                                             */
/*   For the selected sample size (n), dimension (dim), distribution           */
/*   (nDistribution), depth (nDepth) and number of projections (nProjections)  */
/*   'nSim' samples are simulated. For each of these samples the selected      */
/*   approximation method (here: RaSi) is run for each of the parameters       */
/*   passed in 'lList1'.                                                       */
/*                                                                             */
/*   The output shows the values of the statistics 'PercBest' and 'AveRank'    */
/*   for each of the parameter combinations.                                   */
/*                                                                             */
/*   This routine is called in 'TuneParameters'.                               */
/*                                                                             */
/*******************************************************************************/

void TuneRandSimp(int n, int dim, int nDistribution, int nSim, int nDepth, int nProjections, string lList1) {
	clock_t totaltime = 0;
	clock_t starttime;
	double laptime = 0;
	double averagetime;

	double MinDepth;

	int nAlpha;
	double* alphas = analyzeDoubleList(lList1.data(), &nAlpha);

	depthRec* depth = new depthRec[nAlpha]{};
	int* iterations = new int[nAlpha] {};
	double* isBest = new double[nAlpha] {};
	double* rank = new double[nAlpha] {};

	cout << setw(13) << " ";
	for (int i = 0; i < nAlpha; i++) cout << "(" << alphas[i] << ")   ";
	cout << endl;

	cMatrix x;
	double* z;

	x.SetSize(n, dim);
	z = new double[dim];

	cProjection Depth(x, n, dim, nProjections);
	Depth.SetDepthNotion((eDepth)nDepth);

	mt19937 gen(1234);
	cMNormal normal(dim);
	cMCauchy cauchy(dim);
	cMUniformCube unif(dim);
	cMt tDist(dim, 5);
	cMSkewNormal skewNormal(dim, new double[dim] {5});
	cMExponential exponential(dim);
	uniform_int_distribution<int> rndint(0, n - 1);

	function<void(double* const)> rndVector[] = {
		[&](double* const x) { normal.vector(x, gen);  },
		[&](double* const x) { cauchy.vector(x, gen);  },
		[&](double* const x) { unif.vector(x, gen);  },
		[&](double* const x) { tDist.vector(x, gen);  },
		[&](double* const x) { skewNormal.vector(x, gen);  },
		[&](double* const x) { exponential.vector(x, gen);  }
	};
	function<void(double* const)> rndVec = rndVector[nDistribution];

	for (int cnt = 0; cnt < nSim; cnt++) {
		//cout << cnt << endl;
		for (int i = 0; i < n; i++) rndVec(x[i]);
		for (int j = 0; j < dim; j++) z[j] = 0;
		for (int i = 0; i < 10; i++) {
			int k = rndint(gen);
			for (int j = 0; j < dim; j++) z[j] += x[k][j];
		}
		for (int j = 0; j < dim; j++) z[j] = z[j] / 10.0;

		starttime = clock();


		Depth.SetMethod(eProjMeth::RandSimp);
		for (int i = 0; i < nAlpha; i++) {
			Depth.SetAlphaRaSi(alphas[i]);
			depth[i].val = Depth.Depth(z);
			depth[i].it = Depth.NProjections();
			depth[i].method = i;
		}
		MinDepth = DBL_MAX;
		for (int i = 0; i < nAlpha; i++) MinDepth = min(MinDepth, depth[i].val);
		for (int i = 0; i < nAlpha; i++)
			if (abs(depth[i].val - MinDepth) < 1e-8) isBest[i]++;

		sort(depth, depth + nAlpha, [](depthRec& a, depthRec& b) -> bool { return (a.val < b.val); });

		int i{ 0 }, j{ 0 };
		while (i < nAlpha) {
			i++;
			while ((i < nAlpha) && (depth[i].val < depth[j].val + 1e-8)) i++;
			double mr = (i + j - 1) / 2.0;
			for (; j < i; j++) rank[depth[j].method] += mr;
		}

		laptime = clock() - starttime;
		totaltime += laptime;
	}
	for (int i = 0; i < nAlpha; i++) rank[i] /= nSim;
	for (int i = 0; i < nAlpha; i++) isBest[i] /= nSim;

	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nAlpha; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << rank[i];
	cout << endl;
	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nAlpha; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << isBest[i] * 100;
	cout << endl;

	averagetime = (double)totaltime / nSim;
	cout << "% Average time (in s):     " << averagetime / CLOCKS_PER_SEC << endl;
	delete[] z;
}

/*******************************************************************************/
/*                                                                             */
/* 'TuneSimAnn' is used to select the 'best' parameters for                    */
/*   simulated annealingh (SA).                                                */
/*                                                                             */
/*   For the selected sample size (n), dimension (dim), distribution           */
/*   (nDistribution), depth (nDepth) and number of projections (nProjections)  */
/*   'nSim' samples are simulated. For each of these samples the selected      */
/*   approximation method (here: SA) is run for each combination of the        */
/*   parameters passed in 'lList1', 'lList2' and 'lList3'.                     */
/*                                                                             */
/*   The output shows the values of the statistics 'PercBest' and 'AveRank'    */
/*   for each of the parameter combinations.                                   */
/*                                                                             */
/*   This routine is called in 'TuneParameters'.                               */
/*                                                                             */
/*******************************************************************************/

void TuneSimAnn(int n, int dim, int nDistribution, int nSim, int nDepth, int nProjections, string lList1, string lList2, string lList3) {
	clock_t totaltime = 0;
	clock_t starttime;
	double laptime = 0;
	double averagetime;

	double MinDepth;

	int nBeta, nAlpha, nStart;
	double* alphas = analyzeDoubleList(lList1.data(), &nAlpha);
	double* betas = analyzeDoubleList(lList2.data(), &nBeta);
	int* starts = analyzeIntList(lList3.data(), &nStart);

	depthRec* depth = new depthRec[nAlpha * nBeta * nStart]{};
	int* iterations = new int[nAlpha * nBeta * nStart]{};
	double* isBest = new double[nAlpha * nBeta * nStart]{};
	double* rank = new double[nAlpha * nBeta * nStart]{};

	cout << setw(13) << " ";
	for (int i = 0; i < nAlpha; i++) {
		for (int j = 0; j < nBeta; j++) {
			for (int k = 0; k < nStart; k++) {
				cout << "(" << alphas[i] << "," << betas[j] << "," << starts[k] << ")   ";
			}
		}
	}
	cout << endl;

	cMatrix x;
	double *z;

	x.SetSize(n, dim);
	z = new double[dim];

	cProjection Depth(x, n, dim, nProjections);
	Depth.SetDepthNotion((eDepth)nDepth);

	mt19937 gen(1234);
	cMNormal normal(dim);
	cMCauchy cauchy(dim);
	cMUniformCube unif(dim);
	cMt tDist(dim, 5);
	cMSkewNormal skewNormal(dim, new double[dim] {5});
	cMExponential exponential(dim);
	uniform_int_distribution<int> rndint(0, n - 1);

	function<void(double* const)> rndVector[] = {
		[&](double* const x) { normal.vector(x, gen);  },
		[&](double* const x) { cauchy.vector(x, gen);  },
		[&](double* const x) { unif.vector(x, gen);  },
		[&](double* const x) { tDist.vector(x, gen);  },
		[&](double* const x) { skewNormal.vector(x, gen);  },
		[&](double* const x) { exponential.vector(x, gen);  }
	};
	function<void(double* const)> rndVec = rndVector[nDistribution];

	for (int cnt = 0; cnt < nSim; cnt++) {
		//cout << cnt << endl;
		for (int i = 0; i < n; i++) rndVec(x[i]);
		for (int j = 0; j < dim; j++) z[j] = 0;
		for (int i = 0; i < 10; i++) {
			int k = rndint(gen);
			for (int j = 0; j < dim; j++) z[j] += x[k][j];
		}
		for (int j = 0; j < dim; j++) z[j] = z[j] / 10.0;

		starttime = clock();


		Depth.SetMethod(eProjMeth::SimAnn);
		for (int i = 0; i < nAlpha; i++) {
			Depth.SetAlphaSA(alphas[i]);
			for (int j = 0; j < nBeta; j++) {
				Depth.SetBetaSA(betas[j]);
				for (int k = 0; k < nStart; k++) {
					Depth.SetStartSA(starts[k]);
					depth[(i*nBeta + j)*nStart + k].val = Depth.Depth(z);
					depth[(i*nBeta + j)*nStart + k].it = Depth.NProjections();
					depth[(i*nBeta + j)*nStart + k].method = (i*nBeta + j)*nStart + k;
				}
			}
		}
		MinDepth = DBL_MAX;
		for (int i = 0; i < nAlpha * nBeta * nStart; i++) MinDepth = min(MinDepth, depth[i].val);
		for (int i = 0; i < nAlpha * nBeta * nStart; i++)
			if (abs(depth[i].val - MinDepth) < 1e-8) isBest[i]++;

		sort(depth, depth + nAlpha * nBeta * nStart, [](depthRec& a, depthRec& b) -> bool { return (a.val < b.val); });

		int i{ 0 }, j{ 0 };
		while (i < nAlpha * nBeta * nStart) {
			i++;
			while ((i < nAlpha * nBeta * nStart) && (depth[i].val < depth[j].val + 1e-8)) i++;
			double mr = (i + j - 1) / 2.0;
			for (; j < i; j++) rank[depth[j].method] += mr;
		}

		laptime = clock() - starttime;
		totaltime += laptime;
	}
	for (int i = 0; i < nAlpha * nBeta * nStart; i++) rank[i] /= nSim;
	for (int i = 0; i < nAlpha * nBeta * nStart; i++) isBest[i] /= nSim;

	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nAlpha * nBeta * nStart; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << rank[i];
	cout << endl;
	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nAlpha * nBeta * nStart; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << isBest[i] * 100;
	cout << endl;

	averagetime = (double)totaltime / nSim;
	cout << "% Average time (in s):     " << averagetime / CLOCKS_PER_SEC << endl;
	delete[] z;
}

/*******************************************************************************/
/*                                                                             */
/* 'TuneCoordinateDescent' is used to select the 'best' parameters for         */
/*   the coordinate descent (CD).                                              */
/*                                                                             */
/*   For the selected sample size (n), dimension (dim), distribution           */
/*   (nDistribution), depth (nDepth) and number of projections (nProjections)  */
/*   'nSim' samples are simulated. For each of these samples the selected      */
/*   approximation method (here: CD) is run for each of the parameters         */
/*   passed in 'lList1'.                                                       */
/*                                                                             */
/*   Note that the two possible values for the parameter 'Space' (Ec,Sp)       */
/*   are not passed as a parameter, but are still considered in the routine.   */
/*                                                                             */
/*   The output shows the values of the statistics 'PercBest' and 'AveRank'    */
/*   for each of the parameter combinations.                                   */
/*                                                                             */
/*   This routine is called in 'TuneParameters'.                               */
/*                                                                             */
/*******************************************************************************/

void TuneCoordinateDescent(int n, int dim, int nDistribution, int nSim, int nDepth, int nProjections, string lList1) {
	clock_t totaltime = 0;
	clock_t starttime;
	double laptime = 0;
	double averagetime;

	double MinDepth;

	int nLineSearch;
	int* LS = analyzeIntList(lList1.data(), &nLineSearch);

	depthRec* depth = new depthRec[2 * nLineSearch]{};
	int* iterations = new int[2 * nLineSearch] {};
	double* isBest = new double[2 * nLineSearch] {};
	double* rank = new double[2 * nLineSearch] {};

	cout << setw(13) << " ";
	for (int i = 0; i < nLineSearch; i++) cout << "(CD," << LS[i] << ")   ";
	for (int i = 0; i < nLineSearch; i++) cout << "(CDGC," << LS[i] << ")   ";
	cout << endl;

	cMatrix x;
	double *z;

	x.SetSize(n, dim);
	z = new double[dim];

	cProjection Depth(x, n, dim, nProjections);
	Depth.SetDepthNotion((eDepth)nDepth);

	mt19937 gen(1234);
	cMNormal normal(dim);
	cMCauchy cauchy(dim);
	cMUniformCube unif(dim);
	cMt tDist(dim, 5);
	cMSkewNormal skewNormal(dim, new double[dim] {5});
	cMExponential exponential(dim);
	uniform_int_distribution<int> rndint(0, n - 1);

	function<void(double* const)> rndVector[] = {
		[&](double* const x) { normal.vector(x, gen);  },
		[&](double* const x) { cauchy.vector(x, gen);  },
		[&](double* const x) { unif.vector(x, gen);  },
		[&](double* const x) { tDist.vector(x, gen);  },
		[&](double* const x) { skewNormal.vector(x, gen);  },
		[&](double* const x) { exponential.vector(x, gen);  }
	};
	function<void(double* const)> rndVec = rndVector[nDistribution];


	for (int cnt = 0; cnt < nSim; cnt++) {
		//cout << cnt << endl;
		for (int i = 0; i < n; i++) rndVec(x[i]);
		for (int j = 0; j < dim; j++) z[j] = 0;
		for (int i = 0; i < 10; i++) {
			int k = rndint(gen);
			for (int j = 0; j < dim; j++) z[j] += x[k][j];
		}
		for (int j = 0; j < dim; j++) z[j] = z[j] / 10.0;

		starttime = clock();


		Depth.SetMethod(eProjMeth::CoordDesc);
		for (int i = 0; i < nLineSearch; i++) {
			Depth.SetLineSearchCD((eLineSearchCD)LS[i]);
			depth[i].val = Depth.Depth(z);
			depth[i].it = Depth.NProjections();
			depth[i].method = i;
		}
		Depth.SetMethod(eProjMeth::CoordDescGC);
		for (int i = 0; i < nLineSearch; i++) {
			Depth.SetLineSearchCDGC((eLineSearchCDGC)LS[i]);
			depth[nLineSearch + i].val = Depth.Depth(z);
			depth[nLineSearch + i].it = Depth.NProjections();
			depth[nLineSearch + i].method = nLineSearch + i;
		}
		MinDepth = DBL_MAX;
		for (int i = 0; i < 2 * nLineSearch; i++) MinDepth = min(MinDepth, depth[i].val);
		for (int i = 0; i < 2 * nLineSearch; i++)
			if (abs(depth[i].val - MinDepth) < 1e-8) isBest[i]++;

		sort(depth, depth + 2 * nLineSearch, [](depthRec& a, depthRec& b) -> bool { return (a.val < b.val); });

		int i{ 0 }, j{ 0 };
		while (i < 2 * nLineSearch) {
			i++;
			while ((i < 2 * nLineSearch) && (depth[i].val < depth[j].val + 1e-8)) i++;
			double mr = (i + j - 1) / 2.0;
			for (; j < i; j++) rank[depth[j].method] += mr;
		}

		laptime = clock() - starttime;
		totaltime += laptime;
	}
	for (int i = 0; i < 2 * nLineSearch; i++) rank[i] /= nSim;
	for (int i = 0; i < 2 * nLineSearch; i++) isBest[i] /= nSim;

	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < 2 * nLineSearch; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << rank[i];
	cout << endl;
	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < 2 * nLineSearch; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << isBest[i] * 100;
	cout << endl;

	averagetime = (double)totaltime / nSim;
	cout << "% Average time (in s):     " << averagetime / CLOCKS_PER_SEC << endl;
	delete[] z;
}

/*******************************************************************************/
/*                                                                             */
/* 'TuneNelderMead' is used to select the 'best' parameters for                */
/*   the Nelder-Mead algorithm (NM).                                           */
/*                                                                             */
/*   For the selected sample size (n), dimension (dim), distribution           */
/*   (nDistribution), depth (nDepth) and number of projections (nProjections)  */
/*   'nSim' samples are simulated. For each of these samples the selected      */
/*   approximation method (here: NM) is run for each combination of the        */
/*   parameters passed in 'lList1', 'lList2' and 'lList3'.                     */
/*                                                                             */
/*   Note that the two possible values for the parameter 'Space' (Ec,Sp)       */
/*   are not passed as a parameter, but are still considered in the routine.   */
/*                                                                             */
/*   The output shows the values of the statistics 'PercBest' and 'AveRank'    */
/*   for each of the parameter combinations.                                   */
/*                                                                             */
/*   This routine is called in 'TuneParameters'.                               */
/*                                                                             */
/*******************************************************************************/

void TuneNelderMead(int n, int dim, int nDistribution, int nSim, int nDepth, int nProjections, string lList1, string lList2, string lList3) {
	clock_t totaltime = 0;
	clock_t starttime;
	double laptime = 0;
	double averagetime;

	double MinDepth;

	int nBeta, nBound, nStart;
	double* betas = analyzeDoubleList(lList1.data(), &nBeta);
	int* bounds = analyzeIntList(lList2.data(), &nBound);
	int* starts = analyzeIntList(lList3.data(), &nStart);

	depthRec* depth = new depthRec[nBeta * nBound * nStart + 1]{};
	int* iterations = new int[nBeta * nBound * nStart + 1]{};
	double* isBest = new double[nBeta * nBound * nStart + 1]{};
	double* rank = new double[nBeta * nBound * nStart + 1]{};

	cout << setw(13) << " " << "(NM)   ";
	for (int i = 0; i < nBeta; i++)
		for (int j = 0; j < nBound; j++)
			for (int k = 0; k < nStart; k++) {
				cout << "(NMGC," << betas[i] << "," << bounds[j] << "," << starts[k] << ")   ";
			}
	cout << endl;

	cMatrix x;
	double *z;

	x.SetSize(n, dim);
	z = new double[dim];

	cProjection Depth(x, n, dim, nProjections);
	Depth.SetDepthNotion((eDepth)nDepth);

	mt19937 gen(1234);
	cMNormal normal(dim);
	cMCauchy cauchy(dim);
	cMUniformCube unif(dim);
	cMt tDist(dim, 5);
	cMSkewNormal skewNormal(dim, new double[dim] {5});
	cMExponential exponential(dim);
	uniform_int_distribution<int> rndint(0, n - 1);

	function<void(double* const)> rndVector[] = {
		[&](double* const x) { normal.vector(x, gen);  },
		[&](double* const x) { cauchy.vector(x, gen);  },
		[&](double* const x) { unif.vector(x, gen);  },
		[&](double* const x) { tDist.vector(x, gen);  },
		[&](double* const x) { skewNormal.vector(x, gen);  },
		[&](double* const x) { exponential.vector(x, gen);  }
	};
	function<void(double* const)> rndVec = rndVector[nDistribution];

	for (int cnt = 0; cnt < nSim; cnt++) {
		//cout << cnt << endl;
		for (int i = 0; i < n; i++) rndVec(x[i]);
		for (int j = 0; j < dim; j++) z[j] = 0;
		for (int i = 0; i < 10; i++) {
			int k = rndint(gen);
			for (int j = 0; j < dim; j++) z[j] += x[k][j];
		}
		for (int j = 0; j < dim; j++) z[j] = z[j] / 10.0;

		starttime = clock();


		Depth.SetMethod(eProjMeth::NelderMead);
		depth[0].val = Depth.Depth(z);
		depth[0].it = Depth.NProjections();
		depth[0].method = 0;
		Depth.SetMethod(eProjMeth::NelderMeadGC);
		for (int i = 0; i < nBeta; i++) {
			Depth.SetBetaNM(betas[i]);
			for (int j = 0; j < nBound; j++) {
				Depth.SetBoundNM(bounds[j]);
				for (int k = 0; k < nStart; k++) {
					Depth.SetStartNM(starts[k]);
					depth[(i*nBound + j)*nStart + k + 1].val = Depth.Depth(z);
					depth[(i*nBound + j)*nStart + k + 1].it = Depth.NProjections();
					depth[(i*nBound + j)*nStart + k + 1].method = (i*nBound + j)*nStart + k + 1;
				}
			}
		}
		/* Used to check the results
		for (int i = 0; i < nBeta * nBound * nStart ; i++) {
			cout << depth[i].val << " ";
		}
		cout << "E:" << Depth.ExactDepth(z) << endl;
		*/
		MinDepth = DBL_MAX;
		for (int i = 0; i < nBeta * nBound * nStart + 1; i++) MinDepth = min(MinDepth, depth[i].val);
		for (int i = 0; i < nBeta * nBound * nStart + 1; i++)
			if (abs(depth[i].val - MinDepth) < 1e-8) isBest[i]++;

		sort(depth, depth + nBeta * nBound * nStart + 1, [](depthRec& a, depthRec& b) -> bool { return (a.val < b.val); });

		int i{ 0 }, j{ 0 };
		while (i < nBeta * nBound * nStart + 1) {
			i++;
			while ((i < nBeta * nBound * nStart + 1) && (depth[i].val < depth[j].val + 1e-8)) i++;
			double mr = (i + j - 1) / 2.0;
			for (; j < i; j++) rank[depth[j].method] += mr;
		}

		laptime = clock() - starttime;
		totaltime += laptime;
	}
	for (int i = 0; i < nBeta * nBound * nStart + 1; i++) rank[i] /= nSim;
	for (int i = 0; i < nBeta * nBound * nStart + 1; i++) isBest[i] /= nSim;

	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nBeta * nBound * nStart + 1; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << rank[i];
	cout << endl;
	cout << setw(7) << DistName[nDistribution] << "   " << setw(3) << DepthName[nDepth];
	for (int i = 0; i < nBeta * nBound * nStart + 1; i++)
		cout << "   " << fixed << setw(7) << setprecision(3) << isBest[i] * 100;
	cout << endl;

	averagetime = (double)totaltime / nSim;
	cout << "% Average time (in s):     " << averagetime / CLOCKS_PER_SEC << endl;
	delete[] z;
}

/*******************************************************************************/
/*                                                                             */
/* 'TuneParameters' is the main routine for the fine-tuning of the parameters. */
/*                                                                             */
/*   The fine-tuning of parameters is one of four tasks that can be done       */
/*   with this program (see documentation of 'main' below).                    */
/*                                                                             */
/*   For the fine-tuning of parameters the program should be called with the   */
/*   following command line:                                                   */
/*                                                                             */
/*   ProgName n d dist nSim sDepth nProj sAlg list1 list2 list3                */
/*                                                                             */
/*     ProgName: Name of the executable                                        */
/*     n: int, size of a sample                                                */
/*     d: int, dimension of the data                                           */
/*     dist: string, possible values are "Normal", "Cauchy", "Uniform",        */
/*           "t-Dist", "SkewNormal", "Exponential"                             */
/*     nSim: int, number of simulations to perform                             */
/*     sDepth: string, possible values are 'MD", "HD", "ZD", "PD", "APD"       */
/*     nProj: int, number of projections, i.e., univariate depth evaluations   */
/*     sAlg: string, possible values are "GS", "RGS", "RS", "RRS", "CC",       */
/*             "RaSi", "NM", "SA", "CCGC", "NMGC"                              */
/*     list1: list of parameters i the form [par_1,...,par_k]                  */
/*     list2: optional list of parameters in the form [par_1,...,par_k]        */
/*     list3: optional list of parameters in the form [par_1,...,par_k]        */
/*                                                                             */
/*   Example: HDEst 1000 5 Uniform 1000 PD 1000 NM [1,2,4] [0,1] [0,1]         */
/*                                                                             */
/*   The number of parameter lists to pass depends on the selected             */
/*   approximation method. See the documantations of the above routines.       */
/*                                                                             */
/*   The output shows the values of the statistics 'PercBest' and 'AveRank'    */
/*   for each of the parameter combinations for the selected method.           */
/*                                                                             */
/*******************************************************************************/

void TuneParameters(int argc, char **argv) {

	int nPunkte, dim, nProjections, nDistribution, nDepth, nSim, nMethod;
	string sDistribution, sMethod, sDepth;
	int prec{ 6 };
	
	string lList1{}, lList2{}, lList3{};   

	for (int i = 1; i < argc; i++) {
		string param = argv[i];
		switch (i) {
		case 1: nPunkte = stoi(param); break;
		case 2: dim = stoi(param); break;
		case 3: sDistribution = param; break;
		case 4: nSim = stoi(param); break;
		case 5: sDepth = param; break;
		case 6: nProjections = stoi(param); break;
		case 7: sMethod = param; break;
		case 8: lList1 = param; break;
		case 9: lList2 = param; break;
		case 10: lList3 = param; break;
		}
	}
	
	for (nDistribution = sizeof(DistName) / sizeof(DistName[0]) - 1; nDistribution >= 0; nDistribution--)
		if (DistName[nDistribution].compare(sDistribution) == 0) break;
	for (nDepth = sizeof(DepthName) / sizeof(DepthName[0]) - 1; nDepth >= 0; nDepth--)
		if (DepthName[nDepth].compare(sDepth) == 0) break;
	for (nMethod = sizeof(Algorithm) / sizeof(Algorithm[0]) - 1; nMethod >= 0; nMethod--)
		if (Algorithm[nMethod].compare(sMethod) == 0) break;
	if ((nDistribution < 0) || (nDepth < 0)) {
		cout << "Illegal parameter!" << endl;
		return;
	}

	cout << "% Number of points:      " << nPunkte << endl
		<< "% Dimension:             " << dim << endl
		<< "% Distribution:          " << sDistribution << endl
		<< "% Depth:                 " << sDepth << endl
		<< "% Number of simulations: " << nSim << endl
		<< "% Number of projections: " << nProjections << endl;

	switch (nMethod) {
	case (int)eProjMeth::RefinedRandom: TuneRefinedRandom(nPunkte, dim, nDistribution, nSim, nDepth, nProjections, lList1, lList2); break;
	case (int)eProjMeth::RefinedGrid: TuneRefinedGrid(nPunkte, dim, nDistribution, nSim, nDepth, nProjections, lList1, lList2); break;
	case (int)eProjMeth::CoordDesc: TuneCoordinateDescent(nPunkte, dim, nDistribution, nSim, nDepth, nProjections, lList1); break;
	case (int)eProjMeth::SimAnn: TuneSimAnn(nPunkte, dim, nDistribution, nSim, nDepth, nProjections, lList1, lList2, lList3); break;
	case (int)eProjMeth::RandSimp: TuneRandSimp(nPunkte, dim, nDistribution, nSim, nDepth, nProjections, lList1); break;
	case (int)eProjMeth::NelderMead: TuneNelderMead(nPunkte, dim, nDistribution, nSim, nDepth, nProjections, lList1, lList2, lList3); break;
	}
}

/*******************************************************************************/
/*                                                                             */
/* 'GenerateData' is used for generating the simulated datasets.               */
/*                                                                             */
/*   The generation of the simulated datasets is one of four tasks that can    */
/*   be done with this program (see documentation of 'main' below).            */
/*                                                                             */
/*   For the generation of simulated datasets the program should be called     */
/*   with the following command line:                                          */
/*                                                                             */
/*   ProgName n d dist fileName startIndex endIndex ext                        */
/*                                                                             */
/*     ProgName: Name of the executable                                        */
/*     n: int, size of a sample                                                */
/*     d: int, dimension of the data                                           */
/*     dist: string, possible values are "Normal", "Cauchy", "Uniform",        */
/*           "t-Dist", "SkewNormal", "Exponential"                             */
/*     fileName: string, template for the filenames of the datasets            */
/*     startIndex: int, index of the first dataset                             */
/*     endIndex: int, index of the last dataset                                */
/*     ext: string, extension for datafiles, must contain the leading dot      */
/*                                                                             */
/*   Example: HDEst 1000 10 Cauchy Cauchy-1000-10- 1 1000 .txt                 */
/*                                                                             */
/*   The number of datasets generated is (endIndex - startIndex + 1).          */
/*   The filenames of the datafiles have the form 'fileName<i>ext', where      */
/*   <i> is an index that runs from startIndex to endIndex.                    */
/*                                                                             */
/*******************************************************************************/

void GenerateData(int argc, char** argv) {
	// Starting settings
	int nPunkte = 1000;
	int dim = 5;
	string sDistribution = "Normal";
	int nDistribution = 0;
	string fileNameTmpl = "SomeData";
	int startFileIndex = 1;
	int endFileIndex = 1000;
	int nSim = endFileIndex - startFileIndex + 1;
	string fileExt = ".txt";
	// Read the input
	for (int i = 1; i < argc; i++) {
		string param = argv[i];
		switch (i) {
		case 1: nPunkte = stoi(param); break;
		case 2: dim = stoi(param); break;
		case 3: sDistribution = param; break;
		case 4: fileNameTmpl = param; break;
		case 5: startFileIndex = stoi(param); break;
		case 6: endFileIndex = stoi(param); break;
		case 7: fileExt = param; break;
		}
	}
	// Parse parameters
	for (nDistribution = sizeof(DistName) / sizeof(DistName[0]) - 1; nDistribution >= 0; nDistribution--)
		if (DistName[nDistribution].compare(sDistribution) == 0) break;
	if (nDistribution < 0) {
		cout << "Illegal parameter!" << endl;
		return;
	}
	nSim = endFileIndex - startFileIndex + 1;
	// Pick distribution
	mt19937 gen(1234);
	cMNormal normal(dim);
	cMCauchy cauchy(dim);
	cMUniformCube unif(dim);
	cMt tDist(dim, 5);
	cMSkewNormal skewNormal(dim, new double[dim] {5});
	cMExponential exponential(dim);
	cHemisphericalShell shell(dim, 0.9, 1.0);
	cBimodalNormal bimodal(dim, 2);
	cMultimodalNormal multimodal(dim, 3);

	function<void(double* const)> rndVector[] = {
		[&](double* const x) { normal.vector(x, gen);  },
		[&](double* const x) { tDist.vector(x, gen);  },
		[&](double* const x) { cauchy.vector(x, gen);  },
		[&](double* const x) { unif.vector(x, gen);  },
		[&](double* const x) { skewNormal.vector(x, gen);  },
		[&](double* const x) { exponential.vector(x, gen);  },
		[&](double* const x) { shell.vector(x, gen);  },
		[&](double* const x) { bimodal.vector(x, gen);  },
		[&](double* const x) { multimodal.vector(x, gen);  }
	};
	function<void(double* const)> rndVec = rndVector[nDistribution];
	// Allocate structures
	cMatrix x;
	x.SetSize(nPunkte, dim);
	// The main loop
	for (int cnt = 0; cnt < nSim; cnt++) {
		// Generate data
		for (int i = 0; i < nPunkte; i++) rndVec(x[i]);
		// Assemble file name
		string fileName(fileNameTmpl);
		fileName = fileName + to_string(startFileIndex + cnt) + fileExt;
		// Write the data to the file
		ofstream fout(fileName);
		fout << nPunkte << " " << dim << " ";
		int tmp = -1;
		for (int j = 2; j < dim - 1; j++) {
			fout << tmp << " ";
		}
		fout << tmp << endl;
		for (int i = 0; i < nPunkte; i++) {
			for (int j = 0; j < dim - 1; j++) {
				fout << setprecision(10) << x[i][j] << " ";
			}
			fout << setprecision(10) << x[i][dim - 1] << endl;
		}
		fout.close();
	}
}

/*******************************************************************************/
/*                                                                             */
/* 'RunSimulation' is used for comparing the different apprimation methods.    */
/*                                                                             */
/*   The comparison of the different approximation methods is one of four      */
/*   tasks that can be done with this program (see documentation of 'main').   */
/*                                                                             */
/*   For the comparison of the different approximation methods the program     */ 
/*   should be called with the following command line:                         */
/*                                                                             */
/*   ProgName sDepth sAlg nProj nProjToSave fileName startIndex endIndex       */
/*            ext fileOut list                                                 */
/*                                                                             */
/*     ProgName: Name of the executable                                        */
/*     sDepth: string, possible values are 'MD", "HD", "ZD", "PD", "APD"       */
/*     sAlg: string, possible values are "GS", "RGS", "RS", "RRS", "CC",       */
/*           "RaSi", "NM", "SA", "CCGC", "NMGC"                                */
/*     nProj: int, number of projections, i.e., univariate depth evaluations   */
/*     nProjToSave: int, number of projections that should be saved in the     */
/*                  file containing the results                                */
/*     fileName: string, template for the filenames of the datasets            */
/*     startIndex: int, index of the first dataset                             */
/*     endIndex: int, index of the last dataset                                */
/*     ext: string, extension for datafiles, must contain the leading dot      */
/*     fileOut: string, name of the output file                                */
/*     list: list of parameters for the selected approximation method          */
/*           in the form [par_1,...,par_k]                                     */
/*                                                                             */
/*   Example: HDEst ZD NMGC 100 100 t-Dist-1000-20- 1 1000 .txt                */
/*                  results-100-t-Dist-ZD-NMGC-1000-20.txt                     */
/*                                                                             */
/*   The datafiles must have been created with the routine 'GenerateData'.     */
/*                                                                             */
/*   The filenames 'fileName<i>ext' with <i> ranging from startIndex to        */
/*   endIndex are processed. Detailed results of the approximations are        */
/*   written to the file 'outFile'.                                            */
/*                                                                             */
/*******************************************************************************/

int RunSimulation(int argc, char **argv) {
	// Starting settings
	int nPunkte = 1000;
	int dim = 5;
	string sDistribution = "Normal";
	int nDistribution = 0;
	string sDepth = "ZD";
	int nDepth = 0;
	string sAlgorithm = "RS";
	int nAlgorithm = 0;
	int nProjections = 1000;
	int nProjToSave = 100;
	string sExact = "";
	bool bExact = false;
	string fileNameTmpl = "SomeData";
	int startFileIndex = 1;
	int endFileIndex = 1000;
	int nSim = endFileIndex - startFileIndex + 1;
	string fileExt = ".txt";
	string fileOut = "SomeOutput.txt";
	string pars;
	// Read the input
	for (int i = 1; i < argc; i++) {
		string param = argv[i];
		switch (i) {
		case 1: sDepth = param; break;
		case 2: sAlgorithm = param; break;
		case 3: nProjections = stoi(param); break;
		case 4: nProjToSave = stoi(param); break;
		case 5: fileNameTmpl = param; break;
		case 6: startFileIndex = stoi(param); break;
		case 7: endFileIndex = stoi(param); break;
		case 8: fileExt = param; break;
		case 9: fileOut = param; break;
		case 10: pars = param; break;
		}
	}
	// Parse the input
	for (nDepth = sizeof(DepthName) / sizeof(DepthName[0]) - 1; nDepth >= 0; nDepth--)
		if (DepthName[nDepth].compare(sDepth) == 0) break;
	for (nAlgorithm = sizeof(Algorithm) / sizeof(Algorithm[0]) - 1; nAlgorithm >= 0; nAlgorithm--)
		if (Algorithm[nAlgorithm].compare(sAlgorithm) == 0) break;
	nSim = endFileIndex - startFileIndex + 1;
	if ((nAlgorithm < 0) || (nDepth < 0)) {
		cout << "Illegal parameter!" << endl;
		return 1;
	}

	cout << nDepth << " " << nAlgorithm << " " << nProjections << " " << fileNameTmpl << " " << startFileIndex << " " << endFileIndex << " " << fileExt << endl;
	double tmpPower = 0;
	cout << endl;
	// Initialize data structures
	int maxDirs = 0;
	vector<vector<double>> depthArrays(nSim);
	vector<double> depthValues(nSim);
	vector<double> depthTimes(nSim);
	// Set depth and algorithm (parameters)
	cMatrix x;
	// The main loop
	for (int cnt = 0; cnt < nSim; cnt++) {
		// Assemble file name
		string fileName(fileNameTmpl);
		fileName = fileName + to_string(startFileIndex + cnt) + fileExt;
		// Read the file
		ifstream fin(fileName);
		fin >> nPunkte >> dim;
		x.SetSize(nPunkte, dim);
		double tmp = 0;
		for (int j = 0; j < dim - 2; j++) {
			fin >> tmp;
		}
		for (int i = 0; i < nPunkte; i++) {
			for (int j = 0; j < dim; j++) {
				fin >> x[i][j];
			}
		}
		fin.close();

		double* z = new double[dim];
		for (int j = 0; j < dim; j++) {
			z[j] = 0;
		}
		for (int i = 0; i < 10; i++) {
			//int k = rndint(gen);
			for (int j = 0; j < dim; j++) {
				z[j] += x[i][j];
			}
		}
		for (int j = 0; j < dim; j++) {
			z[j] = z[j] / 10.0;
		}
		// Set depth and algorithm (parameters)
		cProjection Depth(x, nPunkte, dim, nProjections);
		Depth.SetDepthNotion((eDepth)nDepth);
		Depth.SetMethod((eProjMeth)nAlgorithm);
		SetDepthPars(Depth, pars);
		// Calculate depth with timimng
		clock_t starttime = clock();
		//cout << "Start calculating depth" << endl;
		double depth = Depth.Depth(z);
		//cout << "Depth calculated" << endl;
		clock_t laptime = clock() - starttime;
		// Remember result
		depthArrays[cnt] = Depth.Depths();
		depthValues[cnt] = depth;
		depthTimes[cnt] = Depth.LastDepthDuration();
		if (Depth.NProjections() > maxDirs) {
			maxDirs = Depth.NProjections();
		}
		delete[] z;
		//cout << "File " << fileName << " processed." << endl;
		cout << cnt + 1 << " ";
	}
	cout << endl;
	// Construct the output file
	ofstream fout(fileOut);
	cout << "Writing to file " << fileOut << "output of " << nSim << " simulations..." << endl;
	for (int cnt = 0; cnt < nSim; cnt++) {
		fout << depthTimes[cnt] << " " << depthValues[cnt] << " " << depthArrays[cnt].size() << " " << nProjections << " " << maxDirs;
		if (maxDirs < nProjToSave) {
			// If we needed less projecitons than upper bound, do not waste HDD-spase
			for (int i = 0; i < depthArrays[cnt].size(); i++) {
				fout << " " << depthArrays[cnt][i];
			}
			if (depthArrays[cnt].size() < maxDirs) {
				for (int i = 0; i < maxDirs - depthArrays[cnt].size(); i++) {
					fout << " " << depthArrays[cnt][depthArrays[cnt].size() - 1];
				}
			}
		}
		else {
			// Create the vector of cumulative depths
			for (int i = 1; i < depthArrays[cnt].size(); i++) {
				if (depthArrays[cnt][i] > depthArrays[cnt][i - 1]) {
					depthArrays[cnt][i] = depthArrays[cnt][i - 1];
				}
			}
			// Save (almost) uniformly on the logarithmic scale
			double tmpPower = 0;
			for (int i = 0; i < nProjToSave + 1; i++) {
				if (round(pow(10, tmpPower + i * (log10(nProjections) - tmpPower) / (double)nProjToSave)) - 1 < depthArrays[cnt].size()) {
					fout << " " << depthArrays[cnt][round(pow(10, tmpPower + i * (log10(nProjections) - tmpPower) / (double)nProjToSave)) - 1];
				}
			}
		}
		fout << endl;
	}
	fout.close();
	return 0;
}

/*******************************************************************************/
/*                                                                             */
/* 'RunExact' is used for computing exact depths for the simulated datasets.   */
/*                                                                             */
/*   The computation of exact depth values is one of four tasks that can be    */
/*   done with this program (see documentation of 'main' below).               */
/*                                                                             */
/*   For the computation of exact depth values the program should be called    */
/*   with the following command line:                                          */
/*                                                                             */
/*   ProgName sDepth fileName startIndex endIndex ext fileOut                  */
/*                                                                             */
/*     ProgName: Name of the executable                                        */
/*     sDepth: string, possible values are 'MD", "HD", "ZD", "PD", "APD"       */
/*     fileName: string, template for the filenames of the datasets            */
/*     startIndex: int, index of the first dataset                             */
/*     endIndex: int, index of the last dataset                                */
/*     ext: string, extension for datafiles, must contain the leading dot      */
/*     fileOut: string, name of the output file                                */
/*                                                                             */
/*   Example: HDEst ZD t-Dist-1000-20- 1 1000 .txt                             */
/*                  results-100-t-Dist-ZD-NMGC-1000-20.txt                     */
/*                                                                             */
/*   The datafiles must have been created with the routine 'GenerateData'.     */
/*                                                                             */
/*   The filenames 'fileName<i>ext' with <i> ranging from startIndex to        */
/*   endIndex are processed. The computation time (in seconds) and the         */
/*   exact depth are written to 'outFile'.                                     */
/*                                                                             */
/*******************************************************************************/

int RunExact(int argc, char** argv) {
	// Starting settings
	int nPunkte{ 1000 }, dim{ 5 };
	string sDepth = "ZD";
	int nDepth = 0;
	string fileNameTmpl = "SomeData";
	int startFileIndex = 1;
	int endFileIndex = 1000;
	int nSim = endFileIndex - startFileIndex + 1;
	string fileExt = ".txt";
	string fileOut = "SomeOutput.txt";
	string pars = "[]";
	// Read the input
	for (int i = 1; i < argc; i++) {
		string param = argv[i];
		switch (i) {
		case 1: sDepth = param; break;
		case 2: fileNameTmpl = param; break;
		case 3: startFileIndex = stoi(param); break;
		case 4: endFileIndex = stoi(param); break;
		case 5: fileExt = param; break;
		case 6: fileOut = param; break;
		}
	}
	// Parse the input
	for (nDepth = sizeof(DepthName) / sizeof(DepthName[0]) - 1; nDepth >= 0; nDepth--)
		if (DepthName[nDepth].compare(sDepth) == 0) break;
	nSim = endFileIndex - startFileIndex + 1;
	if (nDepth < 0) {
		cout << "Illegal parameter!" << endl;
		return 1;
	}
	cout << nDepth << " " << fileNameTmpl << " " << startFileIndex << " " << endFileIndex << " " << fileExt << endl;
	double tmpPower = 0;
	// Initialize data structures
	vector<double> depthValues(nSim);
	vector<double> depthTimes(nSim);
	// Set depth and algorithm (parameters)
	cMatrix x;
	// The main loop
	for (int cnt = 0; cnt < nSim; cnt++) {
		// Assemble file name
		string fileName(fileNameTmpl);
		fileName = fileName + to_string(startFileIndex + cnt) + fileExt;
		// Read the file
		ifstream fin(fileName);
		fin >> nPunkte >> dim;
		x.SetSize(nPunkte, dim);
		double tmp = 0;
		for (int j = 0; j < dim - 2; j++) fin >> tmp;
		for (int i = 0; i < nPunkte; i++) {
			for (int j = 0; j < dim; j++) fin >> x[i][j];
		}
		fin.close();
		double* z = new double[dim];
		for (int j = 0; j < dim; j++) z[j] = 0;
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < dim; j++) z[j] += x[i][j];
		}
		for (int j = 0; j < dim; j++) z[j] = z[j] / 10.0;
		// Set depth and algorithm (parameters)
		cProjection Depth(x, nPunkte, dim, 0);
		Depth.SetDepthNotion((eDepth)nDepth);
		SetDepthPars(Depth, pars);
		// Calculate depth with timimng
		clock_t starttime = clock();
		double depth = Depth.ExactDepth(z);
		clock_t laptime = clock() - starttime;
		// Remember result
		depthValues[cnt] = depth;
		depthTimes[cnt] = Depth.LastDepthDuration();
		delete[] z;
		cout << cnt + 1 << " ";
	}
	cout << endl;
	// Construct the output file
	ofstream fout(fileOut);
	cout << "Writing to file " << fileOut << " output of " << nSim << " simulations..." << endl;
	for (int cnt = 0; cnt < nSim; cnt++)
		fout << depthTimes[cnt] << " " << depthValues[cnt] << endl;
	fout.close();
	return 0;
}

/*******************************************************************************/
/*                                                                             */
/* 'makeShortResults' is used for generating the 'results-short' files which   */
/*   contain only the approximated depths.                                     */
/*                                                                             */
/*   This routine assumes that the folders 'results-100', 'results-1000',      */
/*   'results-10000' containing the detailed approximation results as well     */
/*   as the folder 'results-exact' containing the exact depths for the         */
/*   Mahalanobios depth and the zonoid depth have already been generated.      */
/*                                                                             */
/*   The files in the 'results-short' folder are named as follows:             */
/*     results-short-<nProj>-<Dist>-<Depth>-<n>-<d>.txt                        */
/*                                                                             */
/*   Here,                                                                     */
/*     nProj = number of projections used (100,1000,10000)                     */
/*     Dist = distribution (one of the nine used distributions)                */
/*     Depth = depth (MD, HD, ZD, PD, APD)                                     */
/*     n = number of points in the data set (1000)                             */
/*     d = dimension of the data set (5,10,15,20)                              */
/*                                                                             */
/*   Each of these 540 files contains the approximated depth for all of        */
/*   the eight methods and the 1000 simulated data sets for each               */
/*   of the distributional settings                                            */
/*                                                                             */
/*******************************************************************************/

void makeShortResults(int proj, string sDepth, string sDist, int d) {
	const int n{ 1000 };
	string sAlg[] = { "RS", "GS", "RRS", "RGS", "RaSi", "SA", "CD", "NMGC", "Minimum" };
	string inFileName, outFileName, line;
	double time, depth;
	int cnt;

	if ((sDepth == "ZD") || (sDepth == "MD")) sAlg[size(sAlg) - 1] = "Exact";

	unique_ptr<double[]> depths{ new double[maxDatasets * size(sAlg)] };
	for (int i = 0; i < size(sAlg) - 1; i++) {
		inFileName = "results-" + to_string(proj) + "-" + sDist + "-" + sDepth + "-" + sAlg[i] + "-" + to_string(n) + "-" + to_string(d) + ".txt";
		cout << "Reading " << inFileName << endl;
		ifstream input{ "results-" + to_string(proj) + "\\" + inFileName };
		cnt = 0;
		input >> time >> depth;
		while (!input.eof()) {
			depths[cnt * size(sAlg) + i] = depth;
			cnt++;
			getline(input, line);
			input >> time >> depth;
		};
		input.close();
		if (cnt != maxDatasets) {
			cout << "ERROR: cnt <> maxDatasets!" << endl;
			exit(EXIT_FAILURE);
		}
	}
	if ((sDepth == "ZD") || (sDepth == "MD")) { // get exact values
		inFileName = "results-Exact-" + sDist + "-" + sDepth + "-" + to_string(n) + "-" + to_string(d) + ".txt";
		cout << "Reading " << inFileName << endl;
		ifstream input{ "results-exact\\" + inFileName };
		cnt = 0;
		input >> time >> depth;
		while (!input.eof()) {
			depths[(cnt + 1) * size(sAlg) - 1] = depth;
			cnt++;
			getline(input, line);
			input >> time >> depth;
		};
		if (cnt != maxDatasets) {
			cout << "ERROR: cnt <> maxDatasets!" << endl;
			exit(EXIT_FAILURE);
		}
		input.close();
	}
	else { // get minimum
		for (cnt = 0; cnt < maxDatasets; cnt++) {
			double minDepth{ DBL_MAX };
			for (int i = 0; i < size(sAlg) - 1; i++) minDepth = min(minDepth, depths[cnt * size(sAlg) + i]);
			depths[(cnt + 1) * size(sAlg) - 1] = minDepth;
		}
	}

	outFileName = "results-short-" + to_string(proj) + "-" + sDist + "-" + sDepth + "-" + to_string(n) + "-" + to_string(d) + ".txt";
	cout << "Writing data to " << outFileName << endl;
	ofstream out{ "results-short\\" + outFileName };
	for (int j = 0; j < size(sAlg); j++) out << setw(11) << sAlg[j] << " ";
	out << endl;
	for (int i = 0; i < maxDatasets; i++) {
		for (int j = 0; j < size(sAlg); j++) out << setw(11) << depths[i * size(sAlg) + j] << " ";
		out << endl;
	}
	out.close();
	cout << "Job finished!" << endl;
}

void makeShortResults() {
	int n{ 1000 };
	for (int i = 0; i < size(Projections); i++)
		for (int j = 0; j < size(DepthName); j++)
			for (int k = 0; k < size(DistName); k++)
				for (int l = 0; l < size(Dimensions); l++)
					makeShortResults(Projections[i], DepthName[j], DistName[k], Dimensions[l]);
}

/*******************************************************************************/
/*                                                                             */
/* 'computeAllStatistics' is used for computing the four considered statistics */
/*   for evaluationg the performance of the eight considered algorithms.       */
/*                                                                             */
/*   This routine assumes that the 'results-short' files have been generated   */
/*   and stored in the cxorresponding folder. The naming of these files        */
/*   has to follow the scheme described under 'makeShortResults'.              */
/*                                                                             */
/*   This routine generates the statistics files in the folder 'statistics'.   */
/*   The statistics filkes are named as follows:                               */
/*     aveRank-<depth>-<nProj>.txt                                             */
/*     percBest-<depth>-<nProj>.txt                                            */
/*     MAE-<depth>-<nProj>.txt                                                 */
/*     MRE-<depth>-<nProj>.txt                                                 */
/*   Here,                                                                     */
/*     nProj = number of projections used (100,1000,10000)                     */
/*     depth = depth (MD, HD, ZD, PD, APD)                                     */
/*   In each of these 6ß files the repsective statistics are stored for all    */
/*   algorithms and all considered combinations of the nine distributions and  */
/*   the four dimensions.                                                      */
/*                                                                             */
/*   In addition a file 'allStatistics.txt' is generated that contains all     */
/*   statistics. In each row the following data can be found:                  */
/*     nProj        = number of projections (100,1000,10000)                   */ 
/*     Depth        = depth (MD,HD,ZD,PD,APD)                                  */
/*     Distribution = distribution (one of the nine used distributions)        */ 
/*     Dim          = dimension of the data set (5,10,15,20)                   */   
/*     Method       = algorithm (one of the eight considered algorithms)       */
/*     percBest     = percentage of cases where the method was the best        */
/*     aveRank      = average rank of this method                              */
/*     MAE          = mean absolute error                                      */
/*     MRE          = mean reelative error                                     */
/*   Therefore, the file should contain 4320 lines plus the header.            */
/*                                                                             */
/*******************************************************************************/

void computeRanks(double depth[], int n, double ranks[], double bests[]) {
	using sortRec = struct { double depth; int method; };
	unique_ptr<sortRec[]> x{ new sortRec[n] };
	for (int i = 0; i < n; i++) { x[i].depth = depth[i]; x[i].method = i; }
	sort(x.get(), x.get() + n, [](sortRec& a, sortRec& b) -> bool { return (a.depth < b.depth); });

	int i{ 0 }, j{ 0 };
	while (i < n) {
		i++;
		while ((i < n) && (x[i].depth == x[j].depth)) i++;
		double mr = (i + j - 1) / 2.0; //  meanranks;   j;  // minranks 
		for (; j < i; j++) ranks[x[j].method] += (mr + 1);
	}
	for (int i = 0; i < n; i++) bests[x[i].method] += (x[i].depth == x[0].depth) ? 1 : 0;
}

void computeStatistics(string sDepth, int nProj, int d, string sDist, double percBest[], double aveRank[], double MAE[], double MRE[]) {
	const int n{ 1000 };
	const int maxDatasets{ 1000 };
	string sAlg[] = { "RS", "GS", "RRS", "RGS", "RaSi", "SA", "CD", "NMGC" };
	string header;
	string dataPath{}; //  { "D:\\temp\\" };
	unique_ptr<double[]> depth{ new double[size(sAlg)] {} };
	unique_ptr<double[]> absError{ new double[size(sAlg)] {} };
	unique_ptr<double[]> relError{ new double[size(sAlg)] {} };

	double comp;
	int cnt;
	for (int k = 0; k < size(sAlg); k++) percBest[k] = aveRank[k] = MAE[k] = MRE[k] = 0.0;

	string inFileName{ "results-short-" + to_string(nProj) + "-" + sDist + "-" + sDepth + "-" + to_string(n) + "-" + to_string(d) + ".txt" };
	ifstream input{ dataPath + "results-short\\" + inFileName };
	getline(input, header);
	cnt = 0;
	for (int i = 0; i < maxDatasets; i++) {
		for (int j = 0; j < size(sAlg); j++) input >> depth[j];
		input >> comp;

		// update MAE and MRE
		for (int k = 0; k < size(sAlg); k++) MAE[k] += (depth[k] - comp);
		if (comp > 1e-20) {
			for (int k = 0; k < size(sAlg); k++) MRE[k] += (depth[k] - comp) / comp;
			cnt++;
		}
		else cout << "Attention: Outlier!!!" << endl;
		computeRanks(depth.get(), size(sAlg), aveRank, percBest);
	}
	for (int k = 0; k < size(sAlg); k++) {
		percBest[k] *= (100.0 / maxDatasets);
		aveRank[k] /= maxDatasets;
		MAE[k] /= maxDatasets;
		MRE[k] /= cnt;
	}
	input.close();
	if (d > 10) {
		percBest[1] = aveRank[1] = MAE[1] = MRE[1] = nan("");
		percBest[3] = aveRank[3] = MAE[3] = MRE[3] = nan("");
	}
	if (nProj == 100) percBest[3] = aveRank[3] = MAE[3] = MRE[3] = nan("");
}

void computeStatistics(string sDepth, int nProj, ofstream& out) {
	string dataPath{}; // "D:\\temp\\"

	string sAlg[] = { "RS", "GS", "RRS", "RGS", "RaSi", "SA", "CD", "NMGC" };
	unique_ptr<double[]> percBest{ new double[size(sAlg)] {} };
	unique_ptr<double[]> aveRank{ new double[size(sAlg)] {} };
	unique_ptr<double[]> MAE{ new double[size(sAlg)] {} };
	unique_ptr<double[]> MRE{ new double[size(sAlg)] {} };
	string ident{ sDepth + "-" + to_string(nProj) + ".txt" };

	cout << "Processing " << sDepth << "  " << nProj << endl;
	string percBestName{ "percBest-" + ident };
	string aveRankName{ "aveRank-" + ident };
	string MAEName{ "MAE-" + ident };
	string MREName{ "MRE-" + ident };
	ofstream percBestFile{ dataPath + "statistics\\" + percBestName };
	ofstream aveRankFile{ dataPath + "statistics\\" + aveRankName };
	ofstream MAEFile{ dataPath + "statistics\\" + MAEName };
	ofstream MREFile{ dataPath + "statistics\\" + MREName };

	for (int i = 0; i < size(sAlg); i++) {
		percBestFile << setw(6) << sAlg[i];
		aveRankFile << setw(6) << sAlg[i];
		MAEFile << setw(13) << sAlg[i] << " ";
		MREFile << setw(13) << sAlg[i] << " ";
	}
	percBestFile << endl;
	aveRankFile << endl;
	MAEFile << endl;
	MREFile << endl;

	for (int i = 0; i < size(Dimensions); i++) {
		for (int j = 0; j < size(DistName); j++) {
			computeStatistics(sDepth, nProj, Dimensions[i], DistName[j], percBest.get(), aveRank.get(), MAE.get(), MRE.get());
			for (int k = 0; k < size(sAlg); k++) percBestFile << fixed << setw(6) << setprecision(1) << percBest[k];
			percBestFile << endl;
			for (int k = 0; k < size(sAlg); k++) aveRankFile << fixed << setw(6) << setprecision(3) << aveRank[k];
			aveRankFile << endl;
			for (int k = 0; k < size(sAlg); k++) MAEFile << fixed << setw(13) << setprecision(10) << MAE[k] << " ";
			MAEFile << endl;
			for (int k = 0; k < size(sAlg); k++) MREFile << fixed << setw(13) << setprecision(10) << MRE[k] << " ";
			MREFile << endl;
			for (int k = 0; k < size(sAlg); k++) {
				out << setw(6) << nProj << setw(6) << sDepth << setw(15) << DistName[j] << setw(4) << Dimensions[i] << setw(7) << sAlg[k]
					<< fixed << setw(9) << setprecision(1) << percBest[k]
					<< fixed << setw(8) << setprecision(3) << aveRank[k]
					<< fixed << setw(13) << setprecision(10) << MAE[k]
					<< fixed << setw(13) << setprecision(10) << MRE[k] << endl;
			}
		}
	}
	percBestFile.close();
	aveRankFile.close();
	MAEFile.close();
	MREFile.close();
}

void computeAllStatistics() {
	//                                { "MD",  "HD", "ZD",  "PD", "APD" };
	const bool compareWithMinimum[] = { false, true, false, true, false };
	string dataPath{}; // can be used if the files are stored in a different location
	ofstream out{ dataPath + "statistics\\allStatistics.txt" };
	out << setw(6) << "nProj" << setw(6) << "Depth" << setw(15) << "Distribution" << setw(4) << "Dim" << setw(7) << "Method"
		<< setw(9) << "percBest" << setw(8) << "aveRank" << setw(13) << "MAE" << setw(13) << "MRE" << endl;
	for (int i = 0; i < size(DepthName); i++)
		for (int j = 0; j < size(Projections); j++)
			computeStatistics(DepthName[i], Projections[j], out);
	out.close();
}

/*******************************************************************************/
/*                                                                             */
/* 'main'                                                                      */
/*                                                                             */
/* This program can be used for the following possible tasks:                  */
/*   1) Fine-tuning of the parameters of the approximation methods             */
/*   2) Generate simulated datasets for the simulations                        */
/*   3) Run simulations to compare the approximation methods                   */
/*   4) Compute exact depth values for the generated datsets                   */
/*   5) Generation of condensed result files                                   */
/*   6) Computation of statistics for the evaluation of the methods            */
/*                                                                             */
/* Depending on the desired task, the other lines in the main method should    */
/* be commented out.                                                           */
/*                                                                             */
/*******************************************************************************/

int main(int argc, char **argv) {
	TuneParameters(argc, argv);	// Fine-tuning of the parameters of the approximation methods
	GenerateData(argc, argv);	// Generate simulated datasets for the simulations
	RunSimulation(argc, argv);	// Run simulations to compare the approximation methods
	RunExact(argc, argv);		// Compute exact depth values for the generated datsets
	makeShortResults();         // Generation of the 'results-short' files 
	computeAllStatistics();     // Computation of the four performance statistics
}
