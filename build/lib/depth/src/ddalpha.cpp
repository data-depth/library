/*
  File:             ddalpha.cpp
  Created by:       Pavlo Mozharovskyi, Oleksii Pokotylo, Arturo Castellanos
  First published:  28.02.2013
  Last revised:     24.10.2024

  Defines the exported functions for the former 'ddalpha'-package, now 'data-depth'-library.

  For a description of the algorithm, see:
    Lange, T., Mosler, K. and Mozharovskyi, P. (2012). Fast nonparametric classification based on data depth. Statistical Papers.
    Mozharovskyi, P., Mosler, K. and Lange, T. (2013). Classifying real-world data with the DDalpha-procedure. Mimeo.
*/

#include "stdafx.h"

#define EOF (-1)

#ifdef __cplusplus
extern "C" {
#endif

void Sum(double *a, double *b, double *res){
	res[0] = a[0] + b[0];
}

void Det(double *A, int *dim, double *res){
    TDMatrix x = asMatrix(A, *dim, *dim);
    *res = getDet(x, *dim);
    delete[] x;
}

void setSeed(int random_seed){
	if (random_seed != 0) {
		std::seed_seq seq{random_seed};
	}
	else {
		std::seed_seq seq{time(NULL)};
	}
}

void IsInConvexes(double *points, int *dimension, int *cardinalities, int *numClasses, double *objects, int *numObjects, int *seed, int *isInConvexes){
	
	setSeed(*seed);
	int numPoints = 0;for (int i = 0; i < numClasses[0]; i++){numPoints += cardinalities[i];}
	TMatrix x(numPoints);
	for (int i = 0; i < numPoints; i++){x[i] = TPoint(dimension[0]);}
	for (int i = 0; i < numPoints; i++){
		for (int j = 0; j < dimension[0]; j++){
			x[i][j] = points[i * dimension[0] + j];
		}
	}
	TMatrix o(numObjects[0]);
	for (int i = 0; i < numObjects[0]; i++){o[i] = TPoint(dimension[0]);}
	for (int i = 0; i < numObjects[0]; i++){
		for (int j = 0; j < dimension[0]; j++){
			o[i][j] = objects[i * dimension[0] + j];
		}
	}
	TVariables cars(numClasses[0]);
	for (int i = 0; i < numClasses[0]; i++){
		cars[i] = cardinalities[i];
	}
	TIntMatrix answers(o.size());
	int error = 0;
	InConvexes(x, cars, o, error, &answers);
	for (int i = 0; i < numObjects[0]; i++)
    for (int j = 0; j < numClasses[0]; j++){
  		isInConvexes[numClasses[0]*i+j] = answers[i][j];
  	}
}

void ZDepth(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, int *seed, double *depths){
	setSeed(*seed);
	TMatrix x(numPoints[0]);
	for (int i = 0; i < numPoints[0]; i++){x[i] = TPoint(dimension[0]);}
	for (int i = 0; i < numPoints[0]; i++){
		for (int j = 0; j < dimension[0]; j++){
			x[i][j] = points[i * dimension[0] + j];
		}
	}
	TPoint means(*dimension); TPoint sds(*dimension);
	GetMeansSds(x, &means, &sds);
	Standardize(x, means, sds);
	TMatrix z(numObjects[0]);
	for (int i = 0; i < numObjects[0]; i++){z[i] = TPoint(dimension[0]);}
	for (int i = 0; i < numObjects[0]; i++){
		for (int j = 0; j < dimension[0]; j++){
			z[i][j] = objects[i * dimension[0] + j];
		}
		Standardize(z[i], means, sds);
		int error;
		depths[i] = ZonoidDepth(x, z[i], error);
	}
}



void HDepthSpaceEx(double *points, double *objects, int *cardinalities, int *numClasses, int *numObjects,
	int *dimension, int *algNo, double *depths){
	
	double(*func)(double *z, double **xx, int n, int d);
	switch ((HDalgs)*algNo)
	{
	case recursive:
		func = &HD_Rec; break;
	case plane:
		func = &HD_Comb2; break;
	case line:
		func = &HD_Comb; break;
	default:
		func = 0; break;
	}

	TDMatrix x = asMatrix(objects, *numObjects, *dimension);
	int classBegin = 0;

	if (func)
	for (int c = 0; c < *numClasses; c++){
		TDMatrix X = asMatrix(points+classBegin, cardinalities[c], *dimension);
	//	printMatrix(X, cardinalities[c], *dimension);
		for (int i = 0; i < *numObjects; i++){
			depths[c * (*numObjects) + i] = func(x[i], X, cardinalities[c], *dimension);
		}
		classBegin += cardinalities[c]* *dimension;
		delete[] X;
	}
	delete[] x;
}

void HDepthEx(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, int *algNo, double *depths){

	double(*func)(double *z, double **xx, int n, int d);
	switch ((HDalgs)*algNo)
	{
	case recursive:
		func = &HD_Rec; break;
	case plane:
		func = &HD_Comb2; break;
	case line:
		func = &HD_Comb; break;
	default:
		func = 0; break;
	}

	TDMatrix X = asMatrix(points, *numPoints, *dimension);
	TDMatrix x = asMatrix(objects, *numObjects, *dimension);

	if (func)
	for (int i = 0; i < *numObjects; i++){
		depths[i] = func(x[i], X, *numPoints, *dimension);
	}
	delete[] X;
	delete[] x;
}

void MahalanobisDepth(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, double *mat_MCD, double *depths){
	TDMatrix X = asMatrix(points, *numPoints, *dimension);
	TDMatrix x = asMatrix(objects, *numObjects, *dimension);

	MahalanobisDepth(X, x, *dimension, *numPoints, *numObjects, mat_MCD, depths);

	delete[] X;
	delete[] x;
}

void OjaDepth(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, int *seed, int* exact, int *k, int *useCov, double *covEst, double *depths){
	setSeed(*seed);
	TDMatrix X = asMatrix(points, *numPoints, *dimension);
	TDMatrix x = asMatrix(objects, *numObjects, *dimension);
	TDMatrix cov = asMatrix(covEst, *dimension, *dimension);
    
    if (*exact)
        OjaDepthsEx(X, x, *dimension, *numPoints, *numObjects, *useCov, cov, depths);
    else{
		long long K = ((long long)2000000000)*k[0] + k[1];
		OjaDepthsApx(X, x, *dimension, *numPoints, *numObjects, K, *useCov, cov, depths);
	}
	delete[] X;
	delete[] x;
	delete[] cov;
}

void SimplicialDepth(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, int *seed, int* exact, int *k, double *depths){
	setSeed(*seed);
	TDMatrix X = asMatrix(points, *numPoints, *dimension);
	TDMatrix x = asMatrix(objects, *numObjects, *dimension);

	if (*dimension == 2)
		SimplicialDepths2(X, x, *numPoints, *numObjects, depths);
	else if (*exact)
		SimplicialDepthsEx(X, x, *dimension, *numPoints, *numObjects, depths);
	else {
		long long K = ((long long)2000000000)*k[0] + k[1];
		SimplicialDepthsApx(X, x, *dimension, *numPoints, *numObjects, K, depths);
	}
	delete[] X;
	delete[] x;
}




void PotentialDepthsCount(double *points, int *numPoints, int *dimension, int *classes, int *cardinalities, double *testpoints, int *numTestPoints, int* kernelType, double *a, int* ignoreself, double *depths){
	
	TMatrix x(numPoints[0]);
	for (int i = 0; i < numPoints[0]; i++){
		TPoint& curPoint = x[i];
		curPoint.resize(dimension[0]);
		for (int j = 0; j < dimension[0]; j++){
			curPoint[j] = points[i * dimension[0] + j];
		}
	}
	
	TMatrix xt(numTestPoints[0]);
	for (int i = 0; i < numTestPoints[0]; i++){
		TPoint& curPoint = xt[i];
		curPoint.resize(dimension[0]);
		for (int j = 0; j < dimension[0]; j++){
			curPoint[j] = testpoints[i * dimension[0] + j];
		}
	}

	TMatrix d(numTestPoints[0]);
	for (int i = 0; i < numTestPoints[0]; i++){
		d[i].resize(classes[0]);
	}
	TVariables car(classes[0]);
	for (int i = 0; i < classes[0]; i++){
		car[i] = cardinalities[i];
	}

	double (*Kernel) (TPoint& x, TPoint& y, double a) = 0;

	switch (*kernelType){
		case 1: Kernel = EDKernel; break;
		case 2: Kernel = GKernel; break;
		case 3: Kernel = EKernel; break;
		case 4: Kernel = TriangleKernel; break;
		case 5: Kernel = VarGKernel; break;
		default: throw "Unsupported kernel type";
	}
	
	PotentialDepths(x, car, xt, d, Kernel, *a, *ignoreself);

	for (int i = 0; i < numTestPoints[0]; i++){
		for (int j = 0; j < classes[0]; j++){
		//	depths[i * classes[0] + j] = d[i][j];
			depths[j * numTestPoints[0] + i] = d[i][j];
		}
	}
}

void BetaSkeletonDepth(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, double* beta, int* distCode, double* p, double* sigma, double *depths){
  TDMatrix X = asMatrix(points, *numPoints, *dimension);
  TDMatrix x = asMatrix(objects, *numObjects, *dimension);
  TDMatrix s = asMatrix(sigma, *dimension, *dimension);

  LensDepth(X, x, *dimension, *numPoints, *numObjects, *beta, *distCode, *p, s, depths);
  
  delete[] X;
  delete[] x;
  delete[] s;
}

void MinimumCovarianceDeterminantEstim(double *points, int *numPoints, int *dimension, int *hParam, int *seed, double *mat_MCD, double chisqr05, double chisqr0975, int mfull, int nstep, bool hiRegimeCompleteLastComp, bool seeded){
    TDMatrix X = asMatrix(points, *numPoints, *dimension);
    Mcd(X, *numPoints,*dimension, *hParam, mat_MCD, chisqr05, chisqr0975, mfull, nstep, hiRegimeCompleteLastComp, seed, seeded);
    delete[] X;
}

int main() {
    std::cout << random(10);
    return 0;
}
#ifdef __cplusplus
}
#endif
