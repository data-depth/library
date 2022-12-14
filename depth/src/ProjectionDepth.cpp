/*
  File:             ProjectionDepth.cpp
  Created by:       Pavlo Mozharovskyi
  First published:  17.05.2013
  Last revised:     13.11.2015
  
  Computation of the projection depth using random sampling.

  For a description of the method, see:
    Zuo, Y.J. and Serfling, R. (2000). General notions of statistical depth
	  function. Annals of Statistics 28, 461-482.
*/

#include "stdafx.h"



//static int CompareDec(OrderRec x, OrderRec y)
//{
//	return (x.value > y.value);
//}

static void GetMedMad(TPoint &points, double &median, double &mad){
	/* First, determine median */
	int n = points.size();
	//sort(points.begin(), points.end());
  //median = (points[(n + 1)/2 - 1] + points[(n + 2)/2 - 1])/2.;
  nth_element(points.begin(), points.begin() + n/2, points.end());
	median = points[n/2];
	/* Obtain median absolute deviation (from median) (MAD) */
	TPoint deviations(n);
	for (int i = 0; i < n; i++){deviations[i] = fabs(points[i] - median);}
  //sort(deviations.begin(), deviations.end());
  //median = (deviations[(n + 1)/2 - 1] + deviations[(n + 2)/2 - 1])/2.;
  nth_element(deviations.begin(), deviations.begin() + n/2, deviations.end());
	mad = deviations[n/2];
}

void GetPtsPrjDepths(double* projection, int n, double* objectsProjection, int m,
					 TVariables cardinalities, TMatrix *ptsPrjDepths){
	/* Collect basic statistics */
	int q = cardinalities.size();
	for (int i = 0; i < q; i++){
		/* Prepare data and obtain median and mad*/
		int beginIndex = 0;
		for (int j = 0; j < q; j++){
			if (j >= i){break;}
			beginIndex += cardinalities[j];
		}
		int endIndex = beginIndex + cardinalities[i];
		TPoint curClassProjection(projection + beginIndex, projection + endIndex);
		double median, mad;GetMedMad(curClassProjection, median, mad);
		/* Calculate i-class projectional univariate depths */
		for (int j = 0; j < m; j++){
			(*ptsPrjDepths)[i][j] = (objectsProjection[j] - median)/mad;
		}
	}
}

int GetDepthsPrj(TDMatrix points, int n, int d, TDMatrix objects, int m,
				  TVariables cardinalities, int k, bool newDirs, 
				  TDMatrix depths, TDMatrix directions, TDMatrix projections){
	/* 1. Collecting basic statistics */
	int q = cardinalities.size();
	TDMatrix objectsProjections = newM(k,m);
	if (newDirs){
		GetDirections(directions, k, d);
		GetProjections(points, n, d, directions, k, projections);
	}
	GetProjections(objects, m, d, directions, k, objectsProjections);
	/* 2. Calculate projection depths */
	vector<vector<vector<double> > > prjDepths(k, vector<vector<double> >(q, vector<double > (m)));
	for (int i = 0; i < k; i++){
		GetPtsPrjDepths(projections[i], n, objectsProjections[i], m, cardinalities,
			&prjDepths[i]);
	}
	/* 3. Merge depths */
	for (int i = 0; i < m; i++){
		for (int j = 0; j < q; j++){
			depths[i][j] = DBL_MIN;
		}
	}
	for (int i = 0; i < k; i++){
		for (int j = 0; j < q; j++){
			for (int l = 0; l < m; l++){
				if (prjDepths[i][j][l] > depths[l][j]){
					depths[l][j] = prjDepths[i][j][l];
				}
			}
		}
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j < q; j++){		
			depths[i][j] = 1/(1 + depths[i][j]);
		}
	}
	deleteM(objectsProjections);
	return 0;
}
